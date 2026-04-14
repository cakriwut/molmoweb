from olmo.data.data_loader import DataLoaderConfig
from olmo.eval.inf_evaluator import EvaluatorConfig, InfDatasetEvaluatorConfig
from olmo.eval.loss_evaluator import LossDatasetEvaluatorConfig
from olmo.registry import registry


def get_evaluator(name) -> EvaluatorConfig:
    """Gets the default evaluator for task `name`"""
    if (
        name.startswith("screenspot")
        or name.startswith("webolmoSyntheticGround__")
        or name.startswith("groundui")
        or name.startswith("webclick")
    ):
        return EvaluatorConfig(web_ground_eval=True)
    elif name.startswith("webolmoSynthetic"):
        return EvaluatorConfig(web_trajs_eval=True)
    elif name == "screenshot_qa":
        return EvaluatorConfig(screenshot_qa_eval=True)
    elif name == "websrc":
        return EvaluatorConfig(websrc_eval=True)
    elif f"evaluator/{name}" in registry.list():
        return registry.make(f"evaluator/{name}")
    else:
        raise NotImplementedError(name)


def get_default_max_tokens(name):
    if (
        name.startswith("screenspot")
        or name.startswith("webolmoSyntheticGround__")
        or name.startswith("groundui")
        or name.startswith("webclick")
    ):
        max_new_tokens = 256
    elif name.startswith("webolmoSynthetic__"):
        max_new_tokens = 1024
    elif f"max_tokens/{name}" in registry.list():
        max_new_tokens = registry.make(f"max_tokens/{name}")
    else:
        max_new_tokens = 12
    return max_new_tokens


def get_evaluation(
    name,
    seq_len,
    max_examples,
    for_inference=True,
    num_workers=2,
    device_batch_size=None,
    persistent_workers=False,
    include_image=False,
    num_wandb_examples=32,
    response_logits_only=False,
    reduce_loss_metrics_manually=False,
) -> InfDatasetEvaluatorConfig:
    """Gets the default evaluation config for task (or task:split string) `name`"""
    if ":" in name:
        name, split = name.split(":")
    else:
        split = None

    task_name = name
    test_eval_tasks = []
    if split is None:
        split = "test" if task_name in test_eval_tasks else "validation"

    ds = DataLoaderConfig(
        dataset=task_name,
        sequence_length=seq_len,
        split=split,
        shuffle=True,
        drop_last=max_examples is not None and max_examples >= 0,
        num_workers=num_workers,
        pad="to_max",
        pin_memory=True,
        seed=691203,
        persistent_workers=persistent_workers,
    )

    if for_inference:
        evaluator = get_evaluator(name)
        evaluator.num_wandb_examples = num_wandb_examples
        evaluator.n_to_log = 0
        evaluator.save_predictions = None

        max_new_tokens = get_default_max_tokens(name)

        return InfDatasetEvaluatorConfig(
            max_examples=max_examples,
            device_batch_size=device_batch_size,
            max_new_tokens=max_new_tokens,
            evaluator=evaluator,
            label=name,
            data=ds,
            console_log_interval="${console_log_interval}",  # Use log interval in top-level config
            include_image=include_image,
        )

    else:
        return LossDatasetEvaluatorConfig(
            max_examples=max_examples,
            device_batch_size=device_batch_size,
            label=name,
            data=ds,
            console_log_interval="${console_log_interval}",  # Use log interval in top-level config
            response_logits_only=response_logits_only,
            reduce_loss_metrics_manually=reduce_loss_metrics_manually,
        )
