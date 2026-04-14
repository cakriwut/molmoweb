"""Class to evaluate models based on their generation outputs"""
import dataclasses
import itertools
import logging
from collections import defaultdict
from typing import List, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torchmetrics
import wandb
from tqdm import tqdm

from .evaluators import (
    HtmlTable, SavePredictions,
    WebTrajsEval, ScreenshotQAEval, WebGroundingEval,
)
from ..config import BaseConfig
from ..data.data_loader import DataLoaderConfig
from ..nn.beam_search import SamplingConfig, TopKSampler, TopPSampler, MultinomialSampler, \
    TopKTopPSampler, RepeatedNGramBlockingConstraint, RepetitionPenaltyConstraint, \
    FrequencyPenaltyConstraint
from ..torch_util import (
    get_global_rank,
    get_world_size,
    move_to_device, barrier,
)
from ..util import flatten_list

__all__ = ["InfEvaluator", "EvaluatorConfig", "InfDatasetEvaluator", "InfDatasetEvaluatorConfig"]

log = logging.getLogger(__name__)


@dataclasses.dataclass
class InfEvaluator:
    """
    Evaluates the text outputs from a model on a task
    """
    metrics: List

    def __call__(self, predictions, example_metadata, tokenizer, device, step=None, **kwargs):
        inf_metrics = {}
        log.info("Computing metrics...")
        for metric in self.metrics:
            results = metric(example_metadata, predictions, step=step, tokenizer=tokenizer, **kwargs)
            for k in results:
                if k in inf_metrics:
                    log.warning(f"Metric {k} had multiple values")
            inf_metrics.update(results)

        log.info("Aggregating metrics...")
        resolved_metrics = {}
        # sort so metrics are iterated on in the same order on all devices
        for k in sorted(inf_metrics):
            v = inf_metrics[k]
            if isinstance(v, float):
                # Trust the Evaluator writer to provide aggregated metrics
                resolved_metrics[k] = v
            elif isinstance(v, torchmetrics.Metric):
                resolved_metrics[k] = v.to(device).compute().item()
            elif isinstance(v, HtmlTable):
                # Special case, we aggregate table rows from all devices to ensure we can always
                # have enough rows to show even if each device only eval-ed a few examples
                if get_global_rank() == 0:
                    all_predictions = [None]*get_world_size()
                    dist.gather_object(v, all_predictions)
                    all_rows = flatten_list([x.rows for x in all_predictions])

                    if k in ["web_traj_scores"]:
                        # special processing for web trajs scores
                        df = pd.DataFrame(all_rows)
                        per_website_rows = []
                        per_action_rows = []
                        mean_per_website = df.groupby(["metric", "website"])["accuracy"].mean().reset_index()
                        mean_per_action = df.groupby(["metric", "action"])["accuracy"].mean().reset_index()
                        all_metrics = mean_per_website["metric"].unique()

                        for website in mean_per_website["website"].unique():
                            this_website_row = {"website": website}
                            for metric in all_metrics:
                                row = mean_per_website[(mean_per_website["website"] == website) & (mean_per_website["metric"] == metric)]
                                this_website_row[metric] = f'{float(row["accuracy"].values[0]):0.3f}' if not row.empty else np.nan
                            per_website_rows.append(this_website_row)

                        for action in mean_per_action["action"].unique():
                            this_action_row = {"action": action}
                            for metric in all_metrics:
                                row = mean_per_action[(mean_per_action["action"] == action) & (mean_per_action["metric"] == metric)]
                                this_action_row[metric] = f'{float(row["accuracy"].values[0]):0.3f}' if not row.empty else np.nan
                            per_action_rows.append(this_action_row)

                        resolved_metrics["website_scores"] = wandb.Html(HtmlTable(per_website_rows).get_html())
                        resolved_metrics["action_scores"] = wandb.Html(HtmlTable(per_action_rows).get_html())
                    else:
                        resolved_metrics[k] = wandb.Html(HtmlTable(all_rows).get_html())
                else:
                    dist.gather_object(v, None)
            elif isinstance(v, List):
                if get_global_rank() == 0:
                    all_predictions = [None]*get_world_size()
                    dist.gather_object(v, all_predictions)
                    resolved_metrics[k] = []
                    for pred in all_predictions:
                        resolved_metrics[k] += pred
                else:
                    dist.gather_object(v, None)
            else:
                raise ValueError(f"Metric {v} not understood, must be aggregated between devices and of type float|List|HtmlTable|torchmetrics.Metric")

        return resolved_metrics


@dataclasses.dataclass
class EvaluatorConfig(BaseConfig):
    """Config for `Evaluator` objects that compute metrics"""

    n_to_log: int = 10
    """Num examples to log to console"""

    num_wandb_examples: int = 0
    """Num examples to log to Wandb as a HTML table"""

    save_predictions: Optional[str] = "_default"  # saves with default name to checkpoint dir
    """Where to save predictions files"""

    save_tokens: bool = False
    """If save predictions, should the tokens be saved"""

    web_ground_eval: bool = False
    web_trajs_eval: bool = False
    screenshot_qa_eval: bool = False

    def build(self, default_save_dir=None) -> InfEvaluator:
        evaluators = []
        save_predictions = self.save_predictions
        if save_predictions == "_default":
            if default_save_dir is None:
                logging.info(f"save_predictions is \"default\" but no default "
                             f"save dir set so predictions will not be saved")
            save_predictions = default_save_dir
        if save_predictions:
            evaluators.append(SavePredictions(
                save_predictions,
                log_examples=self.n_to_log,
                save_tokens=self.save_tokens
            ))

        if self.web_ground_eval:
            evaluators.append(WebGroundingEval(self.num_wandb_examples))
        if self.web_trajs_eval:
            evaluators.append(WebTrajsEval(self.num_wandb_examples))
        if self.screenshot_qa_eval:
            evaluators.append(ScreenshotQAEval(max(10, self.num_wandb_examples)))
        return InfEvaluator(evaluators)


@dataclasses.dataclass
class InfDatasetEvaluator:
    """Evaluates a model on a dataset"""
    label: str
    dataloader: Any
    evaluator: InfEvaluator
    n_steps: int
    max_new_tokens: int = 448
    console_log_interval: Optional[int] = None
    sampling_parameters: Optional[SamplingConfig] = None

    def run(self, model, device, autocast_precision, is_distributed, pbar=False, logger=None):
        eval_dataloader = self.dataloader
        eval_it = iter(eval_dataloader)
        n_steps = self.n_steps
        if n_steps is not None and 0 <= n_steps < len(self.dataloader):
            eval_it = itertools.islice(eval_it, 0, n_steps)
            total_steps = n_steps
        else:
            total_steps = len(eval_dataloader)

        constraints = []
        if self.sampling_parameters is None:
            sampler = None
        else:
            sampling = self.sampling_parameters
            if sampling.top_k is None and sampling.top_p == 1 and sampling.temperature == 0 and not sampling.ngram_size:
                sampler = None
            else:
                sampler = TopKTopPSampler(p=sampling.top_p, k=sampling.top_k, temperature=sampling.temperature)
            if sampling.ngram_size:
                constraints.append(RepeatedNGramBlockingConstraint(ngram_size=sampling.ngram_size))
            if sampling.repetition_penalty:
                constraints.append(RepetitionPenaltyConstraint(penalty=sampling.repetition_penalty))
            if sampling.frequency_penalty:
                constraints.append(FrequencyPenaltyConstraint(penalty=sampling.frequency_penalty))
        all_metadata = []
        predictions = defaultdict(list)
        done_init = False
        tok = model.config.build_tokenizer()
        pbar = pbar and get_global_rank() == 0
        for eval_step, batch in enumerate(tqdm(eval_it, total=total_steps, ncols=100, disable=not pbar)):
            if logger and eval_step % logger.log_interval == 0:
                logger.log_evaluation(self.label, eval_step, total_steps)
            if "metadata" in batch:
                batch_metadata = batch.pop("metadata")
            else:
                # Handle old-style data that used metadata/ prefix instead
                metadata = {k: batch.pop(k) for k in list(batch) if k.startswith("metadata/")}
                batch_metadata = []
                for i in range(len(batch["input_ids"])):
                    converted = {}
                    for k, v in metadata.items():
                        if isinstance(v[i], bytes):
                            converted[k] = v[i].decode("utf-8")
                        else:
                            converted[k] = v[i].tolist()
                    batch_metadata.append(converted)
            batch_inference = move_to_device(batch, device)
            with torch.inference_mode():
                with torch.autocast("cuda", enabled=True, dtype=autocast_precision):
                    olmo_gen_output = model.generate(
                        batch=batch_inference,
                        max_steps=self.max_new_tokens,
                        sampler=sampler,
                        constraints=constraints,
                        is_distributed=is_distributed
                    )
            input_tokens = olmo_gen_output.token_ids[:, 0].detach().cpu().numpy()
            prompt_tokens = batch_inference["input_ids"].detach().cpu().numpy()
            pred = {
                "predictions": input_tokens, # beam size of 1
                "prompts": prompt_tokens,
                "predictions_text": [tok.decode(x[x >= 0]) for x in input_tokens],
                "prompts_text": [tok.decode(x[x >= 0]) for x in prompt_tokens],
            }

            if hasattr(olmo_gen_output, 'internal') and olmo_gen_output.internal is not None and 'bmm' in olmo_gen_output.internal:
                pred['bmm'] = olmo_gen_output.internal['bmm'].to(torch.float32).detach().cpu().numpy()
                pred['high_res_indices'] = olmo_gen_output.internal['high_res_indices'].to(torch.float32).detach().cpu().numpy()
                pred['frame_time_stamps'] = olmo_gen_output.internal['frame_time_stamps'].to(torch.float32).detach().cpu().numpy()

            valid_ixs = [i for i, md in enumerate(batch_metadata) if md.get("valid", True)]
            all_metadata += [batch_metadata[i] for i in valid_ixs]
            for k, v in pred.items():
                for ix in valid_ixs:
                    predictions[k].append(v[ix])

            # Log to console.
            if self.console_log_interval and not pbar:
                if eval_step + 1 == n_steps or (eval_step + 1) % self.console_log_interval == 0:
                    log.info(f"[eval_step={eval_step + 1}/{total_steps}]")

        barrier()
        tokenizer = model.config.build_tokenizer()
        if logger:
            logger.log_evaluation(self.label, total_steps, total_steps)
        metrics = self.evaluator(predictions, all_metadata, tokenizer, device)
        return metrics


@dataclasses.dataclass
class InfDatasetEvaluatorConfig(BaseConfig):
    """Configuration for an inference evaluator"""

    label: Optional[str] = None
    """Label to use when logging"""

    data: DataLoaderConfig = dataclasses.field(default_factory=DataLoaderConfig)
    """Data to evaluate on"""

    evaluator: EvaluatorConfig = dataclasses.field(default_factory=EvaluatorConfig)
    """Evaluator to compute metrics from the generated outputs"""

    max_new_tokens: int = 448
    """Max number of tokens to generate"""

    device_batch_size: int = 4
    """Batch size"""

    sampling: SamplingConfig = dataclasses.field(default_factory=SamplingConfig)

    subset_num_batches: Optional[int] = None
    """Number of matches to run on, if None use the entire dataset"""

    max_examples: Optional[int] = None
    """Max number of examples to run on, overrides `subset_num_batches`"""

    console_log_interval: Optional[int] = None
    """How often to log progress to console"""

    include_image: bool = False
    """Include image in the metadata"""

    def build_dataset_evaluator(
        self,
        model_config,
        mesh,
        default_save_dir,
        device,
    ) -> InfDatasetEvaluator:
        assert mesh is None, "Mesh not supported for inference for now"
        global_batch_size = self.device_batch_size * get_world_size()
        if self.max_examples and self.max_examples > 0:
            max_steps = max(self.max_examples // global_batch_size, 1)
        elif self.subset_num_batches:
            max_steps = self.subset_num_batches
        else:
            max_steps = None

        eval_loader = self.data.build_eval_dataloader(
            model_config=model_config,
            batch_size=self.device_batch_size,
            mesh=mesh,
            for_inference=True,
            pad_batches=True,
            max_steps_for_padding=max_steps,
            include_image=self.include_image,
        )
        if self.max_examples is not None:
            num_batches = self.max_examples // self.device_batch_size*get_world_size()
        elif self.subset_num_batches is not None:
            num_batches = self.subset_num_batches
        else:
            num_batches = len(eval_loader)

        return InfDatasetEvaluator(
            label=self.label,
            dataloader=eval_loader,
            evaluator=self.evaluator.build(default_save_dir),
            n_steps=max_steps,
            max_new_tokens=self.max_new_tokens,
            console_log_interval=self.console_log_interval,
            sampling_parameters=self.sampling
        )
