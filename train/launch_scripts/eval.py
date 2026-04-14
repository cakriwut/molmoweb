"""Evals a checkpoint on a task, run this script with 'torchrun'."""
import argparse
import logging
from typing import cast

from omegaconf import OmegaConf

from olmo.eval.eval_utils import get_evaluation, get_default_max_tokens
from olmo.eval.inf_evaluator import EvaluatorConfig
from olmo.models.molmo.molmo import MolmoConfig
from olmo.train.trainer_config import FSDPConfig
from olmo.train.trainer_config import FSDPConfig
from olmo.data.data_loader import DataLoaderConfig
from olmo.util import clean_opt, prepare_torchrun_environment, select_checkpoint, resource_path
from olmo.eval.model_evaluator import ModelEvaluator, EvalConfig, DatasetEvaluatorConfig
from dataclasses import replace

log = logging.getLogger(__name__)


def main():
    prepare_torchrun_environment()

    parser = argparse.ArgumentParser(prog="Script to generate dense captions")
    parser.add_argument("checkpoint")
    parser.add_argument("tasks", nargs="+", help="Tasks to evaluate")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seq_len", default=None, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--max_examples", default=None, type=int)
    parser.add_argument("--device_batch_size", default=4, type=int)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--save_eval_data", action="store_true",
                        help="Save detailed inputs/intermediate model data for use in visualizations")
    parser.add_argument("--loss", action="store_true",
                        help="Compute loss/accuracy metrics instead of doing inference")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Override max new tokens, otherwise use task-specific default")
    parser.add_argument("--include_image", action="store_true",
                        help="Include image in the evaluation outputs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--response_logits_only", action="store_true")
    args, other_args = parser.parse_known_args()

    checkpoint_dir = select_checkpoint(args.checkpoint)
    if args.fsdp:
        if args.seq_len is None:
            raise ValueError("Sequence length is required if using FSDP")

    tasks = []
    for task in args.tasks:
        if "," in task:
            tasks += task.split(",")   # support comma seperator just because the jax code does
        else:
            tasks.append(task)
    tasks = list({k: None for k in tasks})  # de-duplicate but keep order

    eval_configs = []
    for task in tasks:
        base_config = get_evaluation(name=task, seq_len=args.seq_len, max_examples=args.max_examples,
                                     num_workers=args.num_workers)
        eval_config = DatasetEvaluatorConfig(
            label=base_config.label,
            data=replace(base_config.data, pad="to_max" if args.fsdp else None),
            generative_evaluator=replace(
                base_config.evaluator,
                n_to_log=4,
                num_wandb_examples=300,
                save_predictions="_default",
            ),
            device_batch_size=args.device_batch_size,
            subset_num_batches=None,
            max_examples=args.max_examples,
            max_new_tokens=args.max_new_tokens or base_config.max_new_tokens,
            response_logits_only=args.response_logits_only,
        )
        eval_configs.append(eval_config)

    # Explicitly set the model config so model settings can be overriden by CLI args
    model_cfg_path = resource_path(select_checkpoint(checkpoint_dir), "config.yaml")
    model_cfg = MolmoConfig.load(model_cfg_path, key="model", validate_paths=False)

    cfg = EvalConfig(
        pbar=False,
        model=model_cfg,
        evaluations=eval_configs,
        load_path=checkpoint_dir,
        console_log_interval=10,
        fsdp=FSDPConfig(fsdp2=True) if args.fsdp else None,
        save_to_checkpoint_dir=args.save_dir is None,
        save_dir=args.save_dir,
        skip_if_metrics_cached=not args.overwrite,
        include_image=args.include_image,
    )

    config = OmegaConf.create(cfg)
    config.merge_with_dotlist([clean_opt(arg) for arg in other_args])
    cfg = cast(EvalConfig, OmegaConf.to_object(config))
    ModelEvaluator(cfg).run()


if __name__ == '__main__':
    main()