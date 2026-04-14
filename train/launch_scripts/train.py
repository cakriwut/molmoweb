import argparse
import logging
from os.path import exists, join
from typing import List, cast

from omegaconf import OmegaConf

from olmo.eval.eval_utils import get_evaluation
from olmo.models.molmo.data_formatter import DataFormatter
from olmo.models.molmo.molmo_preprocessor import MolmoPreprocessorConfig
from olmo.models.molmo.molmo import MolmoConfig
from olmo.models.model import FSDPWrapStrategy
from olmo.nn.vision_backbone import MolmoVisionBackboneConfig
from olmo.model_configs import DEBUG_LLM, DEBUG_VIT
from olmo.data.data_loader import DataLoaderConfig, DatasetWithArgs, KwargsMixture
from olmo.torch_util import get_world_size
from olmo.train.optim import OptimizerConfig, OptimizerType, SchedulerConfig, SchedulerType
from olmo.train.run_trainer import run_trainer
from olmo.train.trainer_config import (
    BatchDivisor,
    CompilerConfig,
    FSDPConfig,
    FSDPPrecision,
    SpeedMonitorConfig,
    TrainConfig,
    WandbConfig,
)
from olmo.util import clean_opt, prepare_torchrun_environment, select_checkpoint


def get_training_mixture(submixture):
    datasets = []
    for task_name in submixture:
        size, weight = None, None
        if isinstance(task_name, DatasetWithArgs):
            datasets.append(task_name)
        else:
            if isinstance(task_name, tuple):
                if len(task_name) == 3:
                    task_name, size, weight = task_name
                else:
                    task_name, size = task_name
            datasets.append(DatasetWithArgs(task_name, None, size, weight))
    return datasets


if __name__ == "__main__":
    prepare_torchrun_environment()

    parser = argparse.ArgumentParser(prog="Train a multitask model")
    parser.add_argument("mixture", help="Name of datset mixture to train on")
    parser.add_argument("checkpoint", help="Path to checkpoint to start from")
    parser.add_argument("--seq_len", default=10240, type=str)
    parser.add_argument("--max_text_seq_len", default=768, type=int)
    parser.add_argument("--duration", default=50000, type=int)
    parser.add_argument("--max_eval_examples", default=256, type=int)
    parser.add_argument("--max_inf_examples", default=None, type=int)
    parser.add_argument("--device_batch_size", default=4, type=int)
    parser.add_argument("--global_batch_size", default=128, type=int)
    parser.add_argument("--include_image", action="store_true",
                        help="Include image in the evaluation outputs")
    parser.add_argument("--turn_off_inference", action="store_true",
                        help="Turn off inference evaluation during training")
    parser.add_argument("--device_eval_batch_size", default=4, type=int)
    parser.add_argument("--device_inf_batch_size", default=4, type=int)
    parser.add_argument("--max_crops", default=8, type=int)
    parser.add_argument("--save_interval", default=2000, type=int)
    parser.add_argument("--inf_eval_interval", default=-1, type=int,
                        help="Inference eval interval, default to end of training") 
    parser.add_argument("--eval_interval", default=-1, type=int,
                        help="Loss eval interval, default to end of training") 
    parser.add_argument("--save_folder", type=str)
    parser.add_argument("--run_name", type=str, default="multitask_train")
    parser.add_argument("--num_checkpoints_to_keep", type=int, default=10,
                        help="Number of checkpoints to keep during training",)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--prefetch_factor", default=4, type=int)

    parser.add_argument("--connector_lr", default=5e-6, type=float)
    parser.add_argument("--vit_lr", default=5e-6, type=float)
    parser.add_argument("--llm_lr", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=200, type=int)
    parser.add_argument("--compile", action="store_true", help="Use torch compile")
    args, other_args = parser.parse_known_args()

    debug = args.checkpoint in ["debug", "debug2"]
    global_batch_size = args.global_batch_size
    if debug:
        checkpoint = None
        model_cfg = MolmoConfig(
            llm=DEBUG_LLM,
            vision_backbone=MolmoVisionBackboneConfig(
                vit=DEBUG_VIT
            ),
            data_formatter=DataFormatter(),
            mm_preprocessor=MolmoPreprocessorConfig(crop_mode="resize", max_crops=1, max_images=2)
        )
        inf_eval_interval = 100
        eval_interval = 100
        log_interval = 5
        eval_examples = 16
        max_inf_examples = 16
        duration = args.duration or 100
        model_cfg.vision_backbone.normalize_on_gpu = False
    else:
        eval_examples = args.max_eval_examples
        max_inf_examples = args.max_inf_examples
        log_interval = 20
        inf_eval_interval = (
            args.duration
            if args.inf_eval_interval == -1
            else args.inf_eval_interval
        )
        eval_interval = (
            args.duration
            if args.eval_interval == -1
            else args.eval_interval
        )
        duration = args.duration
        checkpoint = select_checkpoint(args.checkpoint)
        if exists(join(checkpoint, "model.yaml")):
            model_cfg = MolmoConfig.load(join(checkpoint, "model.yaml"))
        else:
            model_cfg = MolmoConfig.load(
                join(checkpoint, "config.yaml"), key="model"
            )
        model_cfg.vision_backbone.normalize_on_gpu = True

        eval_subset_batches = eval_examples // (
            args.device_eval_batch_size * get_world_size()
        )
        logging.info(f"Setting eval subset batches to {eval_subset_batches}")
        assert eval_subset_batches > 0

    if args.seq_len == "auto":
        seq_len = None
        max_text_len = args.max_text_seq_len
        # This just needs to >= the max possible position embedding so we can overestimate
        model_cfg.llm.max_sequence_length = 4096*8
    else:
        seq_len = int(args.seq_len)
        max_text_len = None
        if model_cfg.llm.max_sequence_length < seq_len:
            model_cfg.llm.max_sequence_length = seq_len

    # Fine-tuning settings
    model_cfg.llm.residual_dropout = 0.1
    model_cfg.llm.response_residual_dropout = 0.0
    model_cfg.data_formatter.prompt_templates = "uber_model"
    model_cfg.data_formatter.message_format = "role"
    model_cfg.data_formatter.system_prompt = "demo_or_style"
    model_cfg.data_formatter.pointing_format = "html-v2"
    model_cfg.mm_preprocessor.loss_token_weighting = "root_subsegments"
    model_cfg.data_formatter.p_choice_content_in_mc = 1.0
    model_cfg.vision_backbone.pooling_attention_mask = True

    pp = model_cfg.mm_preprocessor.image if hasattr(model_cfg.mm_preprocessor, "image") else model_cfg.mm_preprocessor
    pp.max_crops = args.max_crops or pp.max_crops

    if args.mixture == "molmoweb":
        eval_tasks = [
            "screenspot:test",
            "screenspot_v2:test"
        ]
        tasks = [
            [
                "grounding_and_qa",
                [
                    "molmoweb_synthetic_ground__template",
                    "molmoweb_synthetic_ground__gpt",
                    "pixmo_points_single_web",
                    "molmoweb_screenshot_qa",
                ],
                0.2,
            ],
            [
                "gemini_v0",
                [
                    "molmoweb_synthetic_trajs__from_template",
                ],
                0.05,
            ],
            [
                "gemini_webvoyager",
                [
                    "molmoweb_synthetic_trajs__task_seeded_wv",
                ],
                0.10,
            ],
            [
                "gemini_om2w",
                [
                    "molmoweb_synthetic_trajs__task_seeded_om2w",
                ],
                0.20,
            ],
            [
                "gemini_ma",
                [
                    "molmoweb_synthetic_trajs__multi_agent",
                ],
                0.18,
            ],
            [
                "human_trajs",
                [
                    "molmoweb_human_trajs",
                ],
                0.18,
            ],
            [
                "gemini_skills",
                [
                    "molmoweb_synthetic_skills",
                ],
                0.02,
            ],
            [
                "human_skills",
                [
                    "molmoweb_human_skills",
                ],
                0.05,
            ],
            [
                "node_traversal",
                [
                    "molmoweb_synthetic_trajs__node_traversal",
                ],
                0.02,
            ],
        ]

    elif args.mixture == "debug":
        eval_tasks = []
        tasks = [
            [
                "synthetic_trajs",
                [
                    "molmoweb_synthetic_trajs__from_template",
                ],
                1.0
            ]
        ]

    else:
        raise NotImplementedError(args.mixture)

    # Load mixtures
    root_size_mixture: List[KwargsMixture] = []
    for _, submixture, rate in tasks:
        submixture = get_training_mixture(submixture)
        root_size_mixture.append(KwargsMixture(rate, submixture))

    num_workers = args.num_workers
    evaluations = []
    inf_evaluations = []
    # Get cross entropy loss and token accuracy
    for task in eval_tasks:
        evaluation = get_evaluation(
            task,
            seq_len,
            max_examples=eval_examples,
            num_workers=num_workers,
            for_inference=False,
            device_batch_size=args.device_eval_batch_size,
        )
        evaluation.data.max_text_seq_len = max_text_len
        evaluation.data.persistent_workers = True
        evaluation.data.prefetch_factor = args.prefetch_factor
        evaluations.append(evaluation)

    # Run inference evaluation with task-specific metrics
    if not args.turn_off_inference:
        for task in eval_tasks:
            inf_evaluation = get_evaluation(
                task,
                seq_len,
                max_examples=max_inf_examples,
                num_workers=num_workers,
                for_inference=True,
                include_image=args.include_image,
                device_batch_size=args.device_inf_batch_size,
            )
            inf_evaluation.data.max_text_seq_len = max_text_len
            inf_evaluation.data.persistent_workers = True
            inf_evaluation.data.prefetch_factor = args.prefetch_factor
            inf_evaluations.append(inf_evaluation)

    cfg = TrainConfig(
        run_name=args.run_name,  # "multitask_train",
        save_folder="debug_run"
        if debug
        else args.save_folder,  # omegaconf.MISSING
        seed=6198,
        dry_run=False,
        wandb=(
            None
            if debug
            else WandbConfig(
                name="${run_name}",
                project="${oc.env:WANDB_PROJECT}",
                group=None,
                entity="${oc.env:WANDB_ENTITY}",
                log_interval=log_interval,
            allow_resume=True,
            finish_on_sigterm=True,
            )
        ),
        compile=CompilerConfig(mode="default", dynamic=False) if args.compile else None,
        fused_loss=False,
        allow_resume=not debug,
        model=model_cfg,
        save_overwrite=debug,
        data=DataLoaderConfig(
            kwargs_mixture=root_size_mixture,
            shuffle=True,
            split="train",
            drop_last=True,
            sequence_length=seq_len,
            max_text_seq_len=max_text_len,
            num_workers=num_workers,
            pad="to_max",
            pin_memory=True,
            prefetch_factor=args.prefetch_factor,
            seed=50189,
            packing=None
        ),
        ft_connector=True,
        ft_llm=True,
        ft_vit=True,
        optimizer=OptimizerConfig(
            name=OptimizerType.adamw,
            connector_learning_rate=args.connector_lr,
            vit_learning_rate=args.vit_lr,
            llm_learning_rate=args.llm_lr,
            connector_weight_decay=0.0,
            vit_weight_decay=0.0,
            llm_weight_decay=0.0,
            connector_betas=[0.9, 0.95],
            vit_betas=[0.9, 0.95],
            llm_betas=[0.9, 0.95],
            connector_eps=1e-6,
            vit_eps=1e-6,
            llm_eps=1e-6,
        ),
        scheduler=SchedulerConfig(
            name=SchedulerType.multimodal,
            connector_t_warmup=args.warmup_steps,
            vit_t_warmup=args.warmup_steps,
            llm_t_warmup=args.warmup_steps,
            alpha_f=0.1,
            warmup_min_lr=0.0
        ),
        fsdp=FSDPConfig(
            use_orig_params=True,
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.float
        ),
        load_path=None,
        initial_model_checkpoint=checkpoint,
        save_interval=args.save_interval,
        save_num_checkpoints_to_keep=args.num_checkpoints_to_keep,
        global_train_batch_size=global_batch_size,
        device_train_microbatch_size=args.device_batch_size,
        time_limit=None,
        max_duration=duration,
        stop_at="${max_duration}",
        max_grad_norm=1,
        batch_divisor=BatchDivisor.global_batch,
        precision="amp_bf16",
        console_log_interval=log_interval,
        compile_loss=True,
        speed_monitor=SpeedMonitorConfig(window_size=20),
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        inf_eval_interval=inf_eval_interval,
        inf_evaluators=inf_evaluations,
        eval_interval=eval_interval,
        evaluators=[],
        save_final_unsharded_checkpoint=False,
        save_final_optim=False,
        response_logits_only=True
    )

    conf = OmegaConf.create(cfg)
    conf.merge_with_dotlist([clean_opt(arg) for arg in other_args])
    cfg = cast(TrainConfig, OmegaConf.to_object(conf))
    run_trainer(cfg)
