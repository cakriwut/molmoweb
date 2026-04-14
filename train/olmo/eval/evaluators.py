"""Classes the compute metrics given ground truth/prediction pairs"""
import base64
import dataclasses
import io
import json
import logging
import os
import re
import copy
import string

from tqdm import tqdm
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape as html_escape
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw
from torchmetrics import MeanMetric, SumMetric, Metric

from .web_ground_utils import web_grounding_score
from .web_traj_utils import web_traj_step_score
from .screenshot_qa_utils import judge_equivalence
from ..html_utils import build_html_table, postprocess_prompt, BoxesToVisualize, \
    get_html_image_with_boxes
from ..io import write_file
from ..torch_util import (
    get_global_rank,
    get_world_size, barrier,
    gather_object
)
from ..util import flatten_list

log = logging.getLogger(__name__)


def get_openai_key():
    key = os.environ.get("OPENAI_API_KEY")
    if key is None:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return key


def mean_metric(v):
    metric = MeanMetric(nan_strategy="error")
    metric.update(np.mean(v) if len(v)>0 else 0, len(v))
    return metric


def sum_metric(v):
    metric = SumMetric(nan_strategy="error")
    metric.update(np.sum(v) if len(v)>0 else 0)
    return metric


@dataclasses.dataclass
class HtmlTable:
    """Returned as special metric for visualizing predictions"""
    rows: List[Dict[str, Any]]

    def get_html(self):
        return build_html_table(self.rows)


def annotation_to_box(points, point_dist=4):
    to_show = []
    for point in points:
        if len(point) == 2:
            x, y = point
            to_show.append([x-point_dist, y-point_dist, x+point_dist, y+point_dist])
        else:
            to_show.append(point)
    return to_show


def gather_examples_as_html(
    n_examples, voc, metadatas, predictions,
    scores=None, fix_width=True, pred_points=None, gt_points=None,
    pred_bboxes=None, gt_bboxes=None,
) -> HtmlTable:
    """Builds a HTML table visualization of the predictions"""

    n = len(predictions["predictions"])
    if n_examples is not None:
        # Divide by world size since we will aggregate visualization across all processes
        n = min(n, n_examples)
        n = (n + get_world_size() - 1) // get_world_size()
    rows = []
    new_tokens = predictions["predictions"]
    prompt_tokens = predictions["prompts"]
    bmm = predictions["bmm"] if "bmm" in predictions else None
    high_res_indices = predictions["high_res_indices"] if "high_res_indices" in predictions else None
    for ix in range(n):
        prompt_text = postprocess_prompt(voc.decode(prompt_tokens[ix][prompt_tokens[ix] >= 0]))
        metadata = metadatas[ix]
        pred_seq = new_tokens[ix]
        pred_txt = voc.decode(pred_seq[pred_seq >= 0])

        image_src = None
        if "image_url" in metadata:
            image_src = metadata['image_url']
        elif "image" in metadata and isinstance(metadata["image"], np.ndarray):
            img = Image.fromarray(metadata["image"])
            if high_res_indices is not None:
                img_w, img_h = 128, 128
                num_cols = int(img.size[0] / img_w)

                annotate_boxes = high_res_indices[ix]
                draw = ImageDraw.Draw(img)
                for annotate_index in annotate_boxes:
                    if annotate_index == -1:
                        continue
                    annotate_index = int(annotate_index)
                    row_idx = annotate_index // num_cols
                    col_idx = annotate_index % num_cols

                    x1 = col_idx * img_w
                    y1 = row_idx * img_h
                    x2 = x1 + img_w - 1
                    y2 = y1 + img_h - 1

                    # Draw the rectangle with desired thickness
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=8)

            image_data = io.BytesIO()
            img.save(image_data, format='JPEG')
            image_data = image_data.getvalue()
            image_src = f'data:image/jpeg;base64,{base64.b64encode(image_data).decode()}'
        elif "image" in metadata:
            with Image.open(metadata["image"]) as img:
                image_data = io.BytesIO()
                img.save(image_data, format='JPEG')
                image_data = image_data.getvalue()
            image_src = f'data:image/jpeg;base64,{base64.b64encode(image_data).decode()}'

        row = dict()
        if image_src is not None:
            ex_pred_points, gt_pred_points, ex_pred_bboxes, ex_gt_bboxes = None, None, None, None
            if pred_points is not None:
                ex_pred_points = pred_points[ix]
            if gt_points is not None:
                gt_pred_points = gt_points[ix]
            if pred_bboxes is not None:
                ex_pred_bboxes = pred_bboxes[ix]
            if gt_bboxes is not None:
                ex_gt_bboxes = gt_bboxes[ix]
            if ex_pred_points is None and gt_pred_points is None and ex_pred_bboxes is None and ex_gt_bboxes is None:
                row["image"] = f"<img style=\"max-height:500px;max-width:500px;height:auto;width:auto;\" src={image_src}><img>"
            else:
                to_show = []
                if ex_pred_points is not None:
                    to_show.append(BoxesToVisualize(annotation_to_box(ex_pred_points), "blue", format="xyxy"))
                if gt_pred_points is not None:
                    to_show.append(BoxesToVisualize(annotation_to_box(gt_pred_points, 3), "green", format="xyxy"))
                if ex_pred_bboxes is not None:
                    to_show.append(BoxesToVisualize(ex_pred_bboxes, "blue", format="xyxy"))
                if ex_gt_bboxes is not None:
                    to_show.append(BoxesToVisualize(ex_gt_bboxes, "green", format="xyxy"))
                row["image"] = get_html_image_with_boxes(image_src, to_show)
        row["prompt"] = html_escape(prompt_text)
        row["prediction"] = html_escape(pred_txt)

        if bmm is not None:
            row['bmm'] = html_escape(", ".join([f"{bmm_score:0.3f}" for bmm_score in bmm[ix]]))
        if high_res_indices is not None:
            row['high_res_indices'] = html_escape(", ".join([str(int(score)) for score in high_res_indices[ix]]))

        if "answers" in metadata:
            gt = metadata["answers"]
        elif "answer" in metadata:
            gt = metadata["answer"]
        elif "caption" in metadata:
            gt = metadata["caption"]
        else:
            gt = None
        if gt is not None:
            if isinstance(gt, list):
                gt = "<br>".join(html_escape(x) for x in gt)
            else:
                gt = html_escape(gt)
            row["gt"] = gt
        if scores is not None:
            if isinstance(scores[ix], dict):
                for k, v in scores[ix].items():
                    if isinstance(v, str):
                        row[k] = v
                    elif isinstance(v, list):
                        row[k] = "<br>".join(html_escape(x) for x in v)
                    else:
                        row[k] = "" if v is None else f"{v:0.3f}"
            else:
                row["score"] = f"{scores[ix]:0.3f}"

        if "display_in_eval" in metadata and metadata["display_in_eval"]:
            copied_metadata = copy.deepcopy(metadata)
            if "image" in copied_metadata:
                copied_metadata.pop("image")
            row['input_metadata'] = json.dumps(copied_metadata)

        rows.append(row)
    return HtmlTable(rows)


def get_gcs_url(output_file):
    assert output_file.startswith("gs://")
    return f"https://storage.cloud.google.com/{output_file[5:]}?authuser=1"


class Evaluator:
    def __call__(self, metadatas, predictions, tokenizer, step=None):
        raise NotImplementedError()


class SavePredictions(Evaluator):

    @staticmethod
    def get_file_name(step, process_index):
        filename = ""
        if step is not None:
            filename += f"step{step}-"
        if get_world_size() > 1 and process_index is not None:
            filename += f"shard{process_index}"
        filename += "predictions"
        return filename

    def __init__(self, output_dir, json=True, save_tokens=True,
                 log_examples=10, table=100):
        self.save_tokens = save_tokens
        self.output_dir = output_dir
        self.log_examples = log_examples
        self.json = json
        self.table = table

    def __call__(self, metadatas, predictions, tokenizer,
                 step=None, scores=None):
        if not self.output_dir.startswith("gs://"):
            if not os.path.exists(self.output_dir):
                Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]
        json_data = []
        html_data = []

        n_no_eos = 0
        for tok in new_tokens:
            if not np.any(tok == tokenizer.eos_token_id):
                n_no_eos += 1
        if n_no_eos > 0:
            logging.warning(f"{n_no_eos}/{len(new_tokens)} ({n_no_eos/len(new_tokens):00.4f}) "
                            f"examples have no EOS, your inference tokens might be too short")

        for ex_ix, pred_seq in enumerate(new_tokens):
            text = tokenizer.decode(pred_seq[pred_seq >= 0])
            json_row = dict(prediction=text)
            if self.save_tokens:
                json_row["n_tokens"] = pred_seq.tolist()
            prompt_text = postprocess_prompt(tokenizer.decode(prompt_tokens[ex_ix][prompt_tokens[ex_ix] >= 0]))
            if tokenizer.adds_space:
                sep = " "
            else:
                sep = ""
            json_row["prompt"] = prompt_text
            if "bmm" in predictions:
                json_row["bmm"] = predictions["bmm"][ex_ix].tolist()
            if "high_res_indices" in predictions:
                json_row["high_res_indices"] = predictions["high_res_indices"][ex_ix].tolist()

            metadata = metadatas[ex_ix]
            if ex_ix < self.log_examples:
                log.info("*"*30)
                if "example_id" in metadata:
                    log.info(metadata['example_id'])
                log.info(' '.join((prompt_text + sep + text.replace("\n", "\\n")).split()))
            json_row.update({k: v for k, v in metadata.items() if isinstance(v, (str, float, int))})
            json_data.append(json_row)

        json_file = None
        html_file = None
        metrics = {}

        if self.json:
            log.info("Save prediction JSON")
            if get_world_size() > 1 and self.json:
                if get_global_rank() == 0:
                    all_predictions = [None]*get_world_size()
                    dist.gather_object(json_data, all_predictions)
                    json_data = flatten_list(all_predictions)
                else:
                    dist.gather_object(json_data, None)
            if get_global_rank() == 0:
                write_file(
                    self.output_dir,
                    self.get_file_name(step, None) + ".json",
                    json.dumps(json_data, indent=2),
                    save_overwrite=True
                )
                log.info("done saving json")

        if self.table:
            metrics["prediction_table"] = gather_examples_as_html(self.table, tokenizer, metadatas, predictions)
        return metrics


class WebTrajsEval(Evaluator):

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log
        self.metric_names = ["format", "name", "args", "values"]

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]

        scores = {metric: [] for metric in self.metric_names}
        scores_per_website_action = {}
        scores_list = []

        for ex_ix, pred_seq in enumerate(new_tokens):
            metadata = metadatas[ex_ix]
            pred_text = tokenizer.decode(pred_seq[pred_seq >= 0])

            score = web_traj_step_score(pred_text, metadata,)
            gt_action_name = json.loads(metadata["answer"])["action"]["name"]
            website = metadata["website"] if "website" in metadata else "unknown_website"

            if (website, gt_action_name) not in scores_per_website_action:
                scores_per_website_action[(website, gt_action_name)] = {metric: [] for metric in self.metric_names}

            for metric in self.metric_names:
                scores_per_website_action[(website, gt_action_name)][metric].append(score[metric])
                scores[metric].append(score[metric])

            if ex_ix % 100 == 0:
                log.info(f"Example {ex_ix} - Pred: {pred_text}, metadata {metadata}, Score: {score}")
                log.info(f"Mean scores so far: {np.mean(scores['format'])}, {np.mean(scores['name'])}, {np.mean(scores['args'])}, {np.mean(scores['values'])}")

            scores_list.append(score)

        out = {}
        per_website_scores = []
        per_action_scores = []
        per_website_action_scores = []
        for k in self.metric_names:
            overall_row = {"metric": k, "website": "all", "action": "all", "accuracy": np.mean(scores[k])}
            per_website_action_scores.append(overall_row)

        for (website, action), website_action_scores in scores_per_website_action.items():
            for k in self.metric_names:
                row = {"metric": k, "website": website, "action": action, "accuracy": np.mean(website_action_scores[k])}
                per_website_action_scores.append(row)
        out["web_traj_scores"] = HtmlTable(per_website_action_scores)

        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, tokenizer, metadatas, predictions, scores_list,
            )
        return out


class WebGroundingEval(Evaluator):

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]
        vocab = tokenizer

        pred_coords = []
        gt_coords = []
        scores = {"accuracy": []}
        for ex_ix, pred_seq in enumerate(new_tokens):
            metadata = metadatas[ex_ix]
            pred_text = vocab.decode(pred_seq[pred_seq >= 0])

            acc = web_grounding_score(pred_text, metadata)
            scores["accuracy"].append(acc)
            if ex_ix % 100 == 0:
                log.info(f"Example {ex_ix} - Pred: {pred_text}, metadata {metadata}, Acc: {acc}")
                log.info(f"Mean acc so far: {np.mean(scores['accuracy'])}")

        out = {}
        for k, v in scores.items():
            out[k] = mean_metric(v)

        if self.n_to_log:
            per_example_scores = [{k: scores[k][i] for k in scores} for i in range(len(new_tokens))]
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, per_example_scores,
                # pred_points=pred_coords, gt_points=gt_coords
            )
        return out


class ScreenshotQAEval:
    def __init__(self, n_to_log: Optional[int] = None, n_threads: Optional[int] = None):
        try:
            ws = max(1, get_world_size())
        except Exception:
            ws = 1
        self.n_to_log = n_to_log
        self.n_threads = n_threads or max(1, 64 // ws)

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        toks_batch = predictions["predictions"]

        if self.n_to_log is not None:
            N = min(self.n_to_log, len(toks_batch))
            toks_batch = toks_batch[:N]
            metadatas = metadatas[:N]

        items = []
        for i, toks in enumerate(toks_batch):
            pred = tokenizer.decode(toks[toks >= 0]).strip()
            md = metadatas[i]

            q = None
            a = None
            ml = md.get("message_list") if isinstance(md, dict) else None
            if isinstance(ml, list) and ml and isinstance(ml[0], dict):
                q = ml[0].get("question") or ml[0].get("text")
                a = ml[0].get("answer") or md.get("answer")
            q = q or md.get("question") or ""
            a = a or md.get("answer") or ""

            t = md.get("type_of_question") or md.get("type")
            f = md.get("question_form")
            site = md.get("website")

            items.append((q, a, pred, {"type": t, "form": f, "site": site}))

        api_key = get_openai_key()

        rows = []
        reason_primary_counts: Dict[str, int] = {}
        reason_trigger_counts: Dict[str, int] = {}
        llm_calls = 0
        composites: list[float] = []
        matches: list[int] = []

        with ThreadPoolExecutor(max_workers=self.n_threads) as ex:
            futs = {
                ex.submit(judge_equivalence, q, a, p, api_key): (q, a, p, grp)
                for (q, a, p, grp) in items
            }
            for fut in as_completed(futs):
                q, a, p, grp = futs[fut]
                try:
                    r = fut.result()
                except Exception:
                    r = {
                        "match": False,
                        "decision": "error",
                        "primary_reason": "error",
                        "reasons": [],
                        "used_llm": True,
                        "composite": 0.0,
                    }

                llm_calls += 1 if r.get("used_llm") else 0
                matches.append(1 if r.get("match") else 0)
                composites.append(float(r.get("composite", 0.0)))

                pr = r.get("primary_reason", "")
                if pr:
                    reason_primary_counts[pr] = reason_primary_counts.get(pr, 0) + 1
                for s in r.get("reasons", []):
                    nm = s.get("name", "")
                    if s.get("verdict"):
                        reason_trigger_counts[nm] = reason_trigger_counts.get(nm, 0) + 1

                sigs = {s.get("name"): s for s in r.get("reasons", [])}
                def mark(name: str) -> str:
                    v = sigs.get(name, {}).get("verdict", False)
                    return "✅" if v else "❌"

                rows.append({
                    "Type": grp.get("type", ""),
                    "Form": grp.get("form", ""),
                    "Site": grp.get("site", ""),
                    "Q": q,
                    "Gold": a,
                    "Pred": p,
                    "✓": "✅" if r.get("match") else "❌",
                    "Decision": r.get("decision", ""),
                    "Primary": pr or "",
                    # per-metric trigger columns
                    "numbers_equal": mark("numbers_equal"),
                    "list_set_equal": mark("list_set_equal"),
                    "short_span_exact": mark("short_span_exact"),
                    "substring_close": mark("substring_close"),
                    "fuzzy_ratio>=0.90": mark("fuzzy_ratio>=0.90"),
                    "llm_agree": mark("llm_agree"),
                    # composite score (0..1)
                    "Composite": f"{float(r.get('composite', 0.0)):.3f}",
                })

        acc = mean_metric([float(m) for m in matches])

        def scalar_metric(val: float):
            m = MeanMetric(nan_strategy="error")
            m.update(float(val), 1)
            return m

        out = {
            "accuracy": acc,
            "n": scalar_metric(len(rows)),
            "llm_calls": scalar_metric(llm_calls),
            "composite_mean": mean_metric(composites),
        }

        # reason coverage: primary counts and trigger counts
        for k, v in sorted(reason_primary_counts.items(), key=lambda x: -x[1]):
            out[f"primary_reason_count::{k}"] = scalar_metric(v)
        for k, v in sorted(reason_trigger_counts.items(), key=lambda x: -x[1]):
            out[f"reason_trigger_count::{k}"] = scalar_metric(v)

        if self.n_to_log:
            N = min(self.n_to_log, len(rows))
            out["predictions"] = HtmlTable(rows[:N])

        return out


