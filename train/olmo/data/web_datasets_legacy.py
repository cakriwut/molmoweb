from __future__ import annotations

import json
import logging
import os
import random
import re
from os.path import join
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import polars as pl
from PIL import Image
from tqdm import tqdm

from olmo.data.dataset import WEB_DATA_HOME, WEBOLMO_DATASET_VERSION, DatasetBase
from olmo.util import split_into_groups
from olmo.data.web_datasets import (
    WEB_GROUNDING_TEMPLATES,
    get_click_coords_from_bbox,
    normalize_click_coords,
    format_elem_description,
)


class SyntheticTrajs(DatasetBase):
    def __init__(
        self,
        dataset_names,
        split,
        subset=None,
        flatten=True,
        mode="center",
        detail_level="all",
        style="molmo_web_think",
        n_procs=1,
        max_past_steps=3,
        max_msg_char_len=1000,
        max_total_str_char_len=3500,
        data_paths=None,
        max_past_images=0,
        skip_steps_path=None,
        max_total_steps=None,
        sample_fraction=None,
    ):
        self.dataset_version = WEBOLMO_DATASET_VERSION
        self.flatten = flatten
        self.subset = subset
        assert (
            flatten == True
        )  # NOTE: flatten=True is the default and only option that works rn

        self.mode = mode
        assert mode in [
            "center",
            "top_left",
            "random_uniform",
            "random_gaussian",
        ], f"Invalid coords mode: {mode}"

        self.style = style
        assert style in [
            "molmo_web_base",
            "molmo_web_think",
            "molmo_web_mixed",
        ], f"Invalid style: {style}"

        self.detail_level = detail_level
        assert detail_level in [
            "goal",
            "steps",
            "LL",
            "ML",
            "HL",
            "all",
            "all_no_goal",
            "mix_hml",
            "mix_hmls"
        ], f"Invalid detail level: {detail_level}"

        self.dataset_names = dataset_names
        if data_paths is not None:
            self.data_paths = data_paths
        else:
            self.data_paths = [
                os.path.join(WEB_DATA_HOME, self.dataset_version, dataset_name)
                for dataset_name in dataset_names
            ]
        split = "val" if split == "validation" else split
        assert split in ["train", "val", "test", ""], f"Invalid split: {split}"
        self.n_procs = n_procs
        self.max_past_steps = max_past_steps
        self.max_msg_len = max_msg_char_len
        self.max_total_str_len = max_total_str_char_len
        self.max_past_images = max_past_images
        self.max_total_steps = max_total_steps
        self.sample_fraction = sample_fraction
        if sample_fraction is not None:
            assert 0 < sample_fraction <= 1.0, f"sample_fraction must be in (0, 1], got {sample_fraction}"

        # Load skip steps configuration
        self.skip_steps = {}  # Maps traj_id -> set of step indices to skip
        if skip_steps_path is not None:
            print(f"Loading skip steps from: {skip_steps_path}")
            self._load_skip_steps(skip_steps_path)

        super().__init__(split)

    def __len__(self):
        return len(self.data)

    def _load_skip_steps(self, skip_steps_path):
        """Load the skip steps from JSON file and create lookup structure."""
        import json
        try:
            with open(skip_steps_path, "r") as f:
                skip_data = json.load(f)

            repeated_runs = skip_data.get("repeated_runs", [])
            for run in repeated_runs:
                traj_id = run["traj_id"]
                start_idx = run["start"]
                end_idx = run["end"]

                # Initialize set for this trajectory if not exists
                if traj_id not in self.skip_steps:
                    self.skip_steps[traj_id] = set()

                # Add all step indices in the repeated run to skip set
                for step_idx in range(start_idx, end_idx + 1):
                    self.skip_steps[traj_id].add(str(step_idx))

            logging.info(
                f"Loaded skip steps for {len(self.skip_steps)} trajectories "
                f"with {sum(len(v) for v in self.skip_steps.values())} total steps to skip"
            )
        except Exception as e:
            logging.warning(f"Failed to load skip steps from {skip_steps_path}: {str(e)}")
            self.skip_steps = {}

    def truncate_str(
        self,
        some_str: str,
        max_len: int,
        postfix: str = "... (truncated)",
    ):
        if len(some_str) <= max_len:
            return some_str

        return some_str[: max_len - len(postfix)] + postfix

    def truncate_urls_or_titles(
        self, urls_or_titles: list[str] | str, max_len: int = 100
    ):
        if isinstance(urls_or_titles, str):
            return self.truncate_str(urls_or_titles, max_len)
        elif isinstance(urls_or_titles, list):
            return [
                self.truncate_str(url_or_title, max_len)
                for url_or_title in urls_or_titles
            ]
        else:
            print("Something has potentially gone terribly wrong with truncate_urls_or_titles..defaulting to str(urls_or_titles)")
            return self.truncate_str(str(urls_or_titles), max_len)

    def get_formatted_action(self, action_output, image_w, image_h):
        action_name = action_output["action_name"]
        formatted_action = {"name": action_name}
        bbox = None
        if action_name not in ["click", "scroll", "scroll_at", "mouse_drag_and_drop"]:
            formatted_action.update(
                {k: v for k, v in action_output["action"].items()}
            )
            if action_name == "send_msg_to_user":
                formatted_action["msg"] = self.truncate_str(
                    formatted_action.get("msg", ""),
                    max_len=self.max_msg_len,
                )
        else:
            if action_name == "click":
                x, y = None, None  
                # For click actions that just have the click coordinate, no bbox
                if "bbox" not in action_output["action"]:
                    x, y = float(action_output["action"]["x"]), float(action_output["action"]["y"])

                # For click actions that have the bbox of the element being clicked on
                else:
                    x1, y1, w, h = action_output["action"]["bbox"]
                    bbox = [x1, y1, x1 + w, y1 + h]

                    coords = get_click_coords_from_bbox(bbox, mode=self.mode)
                    x, y = float(coords[0]), float(coords[1])

                # normalize the coordinates to [0, 100), and round to 1 decimal places
                normalized_coords = normalize_click_coords(
                    x, y, image_w, image_h
                )
                formatted_action["x"] = normalized_coords[0]
                formatted_action["y"] = normalized_coords[1]
                formatted_action["button"] = action_output["action"].get("button", "")
                formatted_action["click_type"] = action_output["action"].get("click_type", "")

            elif action_name == "scroll":
                delta_x = action_output["action"]["delta_x"]
                delta_y = action_output["action"]["delta_y"]
                normalized_coords = normalize_scroll_deltas(
                    delta_x, delta_y, image_w, image_h
                )
                formatted_action["delta_x"] = normalized_coords[0]
                formatted_action["delta_y"] = normalized_coords[1]
            elif action_name == "scroll_at":
                x = action_output["action"]["x"]
                y = action_output["action"]["y"]
                norm_x, norm_y = normalize_click_coords(
                    x, y, image_w, image_h
                )
                formatted_action["x"] = norm_x
                formatted_action["y"] = norm_y

                delta_x = action_output["action"]["delta_x"]
                delta_y = action_output["action"]["delta_y"]
                norm_dx, norm_dy = normalize_scroll_deltas(
                    delta_x, delta_y, image_w, image_h
                )
                formatted_action["delta_x"] = norm_dx
                formatted_action["delta_y"] = norm_dy
            elif action_name == "mouse_drag_and_drop":
                from_x = action_output["action"]["from_x"]
                from_y = action_output["action"]["from_y"]
                norm_from_x, norm_from_y = normalize_click_coords(
                    from_x, from_y, image_w, image_h
                )

                to_x = action_output["action"]["to_x"]
                to_y = action_output["action"]["to_y"]
                norm_to_x, norm_to_y = normalize_click_coords(
                    to_x, to_y, image_w, image_h
                )

                formatted_action["from_x"] = norm_from_x
                formatted_action["from_y"] = norm_from_y
                formatted_action["to_x"] = norm_to_x
                formatted_action["to_y"] = norm_to_y

        return formatted_action, bbox

    def load_goal(self, metadata):
        goal = metadata.get("goal", "")

        if "instruction" in metadata:
            instruction = metadata["instruction"]
        elif "task" in metadata and "instruction" in metadata.get("task", {}):
            instruction = metadata["task"]["instruction"]
        else:
            instruction = {
                "high_level": goal,
                "mid_level": goal,
                "low_level": goal,
            }
        selected_level = None
        if self.detail_level == "goal":
            return goal, "goal"
        elif self.detail_level == "HL":
            selected_level = "high_level"
            goal = instruction.get("high_level") or goal
        elif self.detail_level == "ML":
            selected_level = "mid_level"
            goal = instruction.get("mid_level") or goal
        elif self.detail_level == "LL":
            selected_level = "low_level"
            goal = instruction.get("low_level") or goal
        elif self.detail_level in ("all", "all_no_goal"):
            labeled_candidates = [
                (label, v) for label, v in [
                    ("high_level", instruction.get("high_level")),
                    ("mid_level", instruction.get("mid_level")),
                    ("low_level", instruction.get("low_level")),
                ] if v
            ]
            if self.detail_level == "all":
                if goal:
                    labeled_candidates.append(("goal", goal))
                task = metadata.get("task", {})
                if "steps" in task:
                    labeled_candidates.append(("steps", "\n".join(task["steps"])))
            if labeled_candidates:
                chosen = random.choice(labeled_candidates)
                selected_level = chosen[0]
                goal = chosen[1]
        elif self.detail_level == "mix_hml":
            # high_level 40%, mid_level 40%, low_level 20%
            levels = [
                ("high_level", 0.4),
                ("mid_level", 0.4),
                ("low_level", 0.2),
            ]
            candidates = [(k, instruction.get(k), w) for k, w in levels if instruction.get(k)]
            if candidates:
                labels, values, weights = zip(*candidates)
                idx = random.choices(range(len(values)), weights=weights, k=1)[0]
                goal = values[idx]
                selected_level = labels[idx]
        elif self.detail_level == "mix_hmls":
            # high_level 25%, mid_level 25%, low_level 25%, steps 25%
            task = metadata.get("task", {})
            steps_text = "\n".join(task["steps"]) if "steps" in task else None
            levels = [
                ("high_level", 0.25),
                ("mid_level", 0.25),
                ("low_level", 0.25),
            ]
            candidates = [(k, instruction.get(k), w) for k, w in levels if instruction.get(k)]
            if steps_text:
                candidates.append(("steps", steps_text, 0.25))
            if candidates:
                labels, values, weights = zip(*candidates)
                idx = random.choices(range(len(values)), weights=weights, k=1)[0]
                goal = values[idx]
                selected_level = labels[idx]
        else:
            raise ValueError(f"Unknown detail level: {self.detail_level}")

        if not goal:
            raise ValueError(f"Empty goal resolved from metadata: {metadata}")

        return goal, selected_level


    def get_action_description(self, traj_step, formatted_action):
        action_description = traj_step["action"]["action_description"]
        action_name = formatted_action["name"]
        if action_name == "scroll":
            delta_x, delta_y = (
                formatted_action["delta_x"],
                formatted_action["delta_y"],
            )
            if delta_x == 0:
                scroll_delta = delta_y
                direction = "down" if delta_y > 0 else "up"
            elif delta_y == 0:
                scroll_delta = delta_x
                direction = "right" if delta_x > 0 else "left"
            else:
                raise ValueError(
                    f"Scroll expected to be uni-directional, but got delta_x={delta_x} and delta_y={delta_y}"
                )
            action_description = (
                f"scroll {direction} by {abs(scroll_delta)} percent"
            )

        if action_description.startswith("send message to user"):
            action_description = "send message to user"

        return action_description

    def load_trajectory_step(
        self,
        traj_step,
        traj_dir,
        image_path,
        goal,
        past_actions,
        past_urls,
        past_images,
        metadata,
        data_path_idx,
        step_idx,
    ):
        image_filename = traj_step["screenshot"]
        action_output = traj_step["action"]["action_output"]
        action_name = action_output["action_name"]
        abs_image_path = os.path.join(image_path, image_filename)
        with Image.open(abs_image_path) as img:
            image_w, image_h = img.size
        # Information needed for webolmo_base template
        other_obs = traj_step["other_obs"]
        # Check if other_obs has the required keys
        has_required_keys = (
            other_obs
            and "page_index" in other_obs
            and "open_pages_titles" in other_obs
            and "open_pages_urls" in other_obs
        )
        if has_required_keys:
            page_index = other_obs["page_index"]
            other_obs["open_pages_titles"] = [
                title if title is not None else "New Tab"
                for title in other_obs["open_pages_titles"]
            ]
            other_obs["open_pages_urls"] = [
                url if url is not None else "about:blank"
                for url in other_obs["open_pages_urls"]
            ]
            page_title = self.truncate_urls_or_titles(
                other_obs["open_pages_titles"][page_index]
            )
            page_url = self.truncate_urls_or_titles(
                other_obs["open_pages_urls"][page_index]
            )
            open_pages_urls = self.truncate_urls_or_titles(
                other_obs["open_pages_urls"]
            )
            open_pages_titles = self.truncate_urls_or_titles(
                other_obs["open_pages_titles"]
            )
            open_pages_titles_and_urls = [
                x for x in zip(open_pages_titles, open_pages_urls)
            ]
            last_action_error = traj_step.get("error", None)
            if last_action_error is None:
                last_action_error = "The action was successful with no error."
        else:  # handle empty dict or missing keys
            page_index = 0
            page_title = "Unknown"
            page_url = "Unknown"
            open_pages_urls = []
            open_pages_titles = []
            open_pages_titles_and_urls = []
            last_action_error = "No observation data (other_obs is empty or missing keys)."

        formatted_action, bbox = self.get_formatted_action(
            action_output, image_w, image_h
        )

        effective_style = self.style
        if effective_style == "molmo_web_mixed":
            effective_style = "molmo_web_base" if random.random() < 0.5 else "molmo_web_think"

        if effective_style == "molmo_web_think":
            answer_dict = {
                "thought": traj_step["action"]["action_output"]["thought"].strip(),
                "action": formatted_action,
            }
        else:  # molmo_web_base
            answer_dict = {
                "action": formatted_action,
            }

        message = dict(
            answer=json.dumps(answer_dict, ensure_ascii=False),
            task_description=goal,
            past_actions=past_actions[-self.max_past_steps:],
            past_urls=past_urls[-self.max_past_steps:],
            page_index=page_index,
            page_title=page_title,
            page_url=page_url,
            open_pages_titles_and_urls=open_pages_titles_and_urls,
            last_action_error=last_action_error,
            style=effective_style,
        )

        # Check for too-long of strings and skip
        tot_str_length = sum(
            len(v) for v in message.values() if isinstance(v, str)
        )
        if tot_str_length > self.max_total_str_len:
            raise ValueError(f"String length exceeds limit: {tot_str_length}")
        
        if self.flatten:
            task_data = metadata.get("task", {})
            instruction_data = task_data.get("instruction", {})
            formatted_example = {
                "image": past_images[-self.max_past_images:] + [abs_image_path] if self.max_past_images > 0 else abs_image_path,
                "message_list": [message],
                "metadata": dict(
                    traj_id=traj_dir,
                    dataset=self.dataset_names[data_path_idx],
                    step_id=step_idx,
                    steps=task_data.get("steps", []),
                    high_level=instruction_data.get("high_level", metadata.get("goal", "")),
                    mid_level=instruction_data.get("mid_level", metadata.get("goal", "")),
                    low_level=instruction_data.get("low_level", metadata.get("goal", "")),
                    bbox=bbox,
                    open_pages_titles=open_pages_titles,
                    open_pages_urls=open_pages_urls,
                    image_w=image_w,
                    image_h=image_h,
                    answer=json.dumps(answer_dict),
                ),
            }
        else:
            raise NotImplementedError("flatten=False not supported")
        return formatted_example, answer_dict, page_url, has_required_keys

    def load_trajectory(self, split_path, traj_dir, data_path_idx):
        traj_dir_path = os.path.join(split_path, traj_dir)
        if not os.path.isdir(traj_dir_path):
            return None

        traj_json_path = os.path.join(traj_dir_path, "trajectory.json")
        metadata_path = os.path.join(traj_dir_path, "metadata.json")
        image_path = os.path.join(traj_dir_path, "images")
        try:
            with open(traj_json_path, "r") as f:
                traj_json_data = json.load(f)
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            logging.info(
                f"Error reading files for trajectory: {traj_dir_path}: {str(e)}"
            )
            if (
                not os.path.exists(traj_json_path)
                or not os.path.exists(metadata_path)
                or not os.path.exists(image_path)
            ):
                logging.info(f"Missing files in {traj_dir_path}")

            return None

        # Filter by max total steps
        if self.max_total_steps is not None and len(traj_json_data) > self.max_total_steps:
            return None

        # Choose the instruction based on the specified detail level
        try:
            goal, selected_level = self.load_goal(metadata)
        except Exception as e:
            logging.warning(f"Error loading goal for trajectory {traj_dir}: {str(e)}")
            return None
        formatted_traj_data = []
        past_actions = []
        past_urls = []
        past_image_paths = []
        missing_keys_count = 0
        skipped_steps_count = 0
        non_unidirectional_scroll_count = 0
        for step_idx, traj_step in traj_json_data.items():
            # Skip this step if it's in the skip list for this trajectory
            if traj_dir in self.skip_steps and step_idx in self.skip_steps[traj_dir]:
                skipped_steps_count += 1
                continue

            try:
                (
                    formatted_example,
                    answer_dict,
                    page_url,
                    has_required_keys,
                ) = self.load_trajectory_step(
                    traj_step=traj_step,
                    traj_dir=traj_dir,
                    image_path=image_path,
                    goal=goal,
                    past_actions=past_actions,
                    past_urls=past_urls,
                    past_images=past_image_paths,
                    metadata=metadata,
                    data_path_idx=data_path_idx,
                    step_idx=step_idx,
                )
                if not has_required_keys:
                    missing_keys_count += 1
                image_filename = traj_step["screenshot"]
                abs_image_path = os.path.join(image_path, image_filename)
                past_image_paths.append(abs_image_path)
                past_action_dict = answer_dict.copy()
                past_action_dict.update({"index": step_idx})
                past_actions.append(past_action_dict)
                past_urls.append(page_url)
                formatted_traj_data.append(formatted_example)
            except Exception as e:
                if "uni-directional" in str(e):
                    non_unidirectional_scroll_count += 1
                logging.info(
                    f"Error while loading step #{step_idx} in {traj_dir}: {str(e)}"
                )
                continue

        if skipped_steps_count > 0:
            logging.debug(
                f"Skipped {skipped_steps_count} repeated steps in trajectory {traj_dir}"
            )

        dataset_name = self.dataset_names[data_path_idx]
        return formatted_traj_data, missing_keys_count, dataset_name, non_unidirectional_scroll_count, selected_level

    def load_trajectory_batch(self, split_path, traj_dirs, data_path_idx):
        results = []
        for traj_dir in traj_dirs:
            results.append(self.load_trajectory(split_path, traj_dir, data_path_idx))
        return results

    def load(self):
        formatted_data = []
        count = 0
        failed_count = 0
        total_non_unidirectional_scrolls = 0
        missing_keys_per_dataset = defaultdict(int)
        batch_size = 64
        print(f"Loading trajectories with {self.n_procs} processes...")
        with mp.Pool(self.n_procs) as pool:
            results = []
            for data_path_idx, data_path in enumerate(self.data_paths):
                split_path = os.path.join(data_path, self.split)
                all_dirs = sorted(os.listdir(split_path))
                if self.sample_fraction is not None:
                    k = max(1, int(len(all_dirs) * self.sample_fraction))
                    all_dirs = random.Random(RNG_SEED).sample(all_dirs, k)
                    print(f"Sampled {k}/{len(os.listdir(split_path))} trajectories (fraction={self.sample_fraction}) from {data_path}")
                for i in range(0, len(all_dirs), batch_size):
                    batch = all_dirs[i:i + batch_size]
                    results.append(
                        pool.apply_async(
                            self.load_trajectory_batch,
                            args=(split_path, batch, data_path_idx),
                        )
                    )
            instruction_level_counts = defaultdict(int)
            for r in tqdm(results):
                for result in r.get():
                    if result is not None:
                        formatted_traj_data, missing_keys_count, dataset_name, non_uni_scroll_count, selected_level = result
                        formatted_data.extend(formatted_traj_data)
                        if missing_keys_count > 0:
                            missing_keys_per_dataset[dataset_name] += missing_keys_count
                        total_non_unidirectional_scrolls += non_uni_scroll_count
                        if selected_level:
                            instruction_level_counts[selected_level] += 1
                        count += 1
                    else:
                        failed_count += 1

        print(
            "Loaded ",
            len(formatted_data),
            " datapoints for split: ",
            self.split,
            " and dataset subsets: ",
            self.dataset_names,
            f" (loaded {count} trajs; failed to load {failed_count} trajs)",
        )
        if total_non_unidirectional_scrolls > 0:
            print(
                f"Total non-unidirectional scrolls (skipped): {total_non_unidirectional_scrolls}"
            )
        if missing_keys_per_dataset:
            for dataset_name, missing_count in sorted(missing_keys_per_dataset.items()):
                logging.warning(
                    f"[{dataset_name}] {missing_count} steps with missing other_obs keys (page_index, open_pages_titles, open_pages_urls)"
                )

        if self.skip_steps:
            total_skipped = sum(len(steps) for steps in self.skip_steps.values())
            print(
                f"Skip steps filter active: {len(self.skip_steps)} trajectories with "
                f"{total_skipped} steps marked for skipping"
            )

        if self.max_total_steps is not None:
            print(
                f"max_total_steps filter active: only trajectories with "
                f"<= {self.max_total_steps} steps were kept"
            )

        # Print instruction level usage counts
        if instruction_level_counts:
            total = sum(instruction_level_counts.values())
            print(f"\nInstruction level usage (detail_level={self.detail_level}, total={total} trajs):")
            for level in ["high_level", "mid_level", "low_level", "steps", "goal"]:
                cnt = instruction_level_counts.get(level, 0)
                pct = cnt / total * 100 if total > 0 else 0
                print(f"  {level}: {cnt} ({pct:.1f}%)")

        return formatted_data

    def get(self, item, rng):
        example = self.data[item]
        return example

class SyntheticParquetTrajs(DatasetBase):
    def __init__(
        self,
        dataset_names: list[str],
        split: Literal["train", "val", "validation", "test"],
        mode="center",
        detail_level: Literal[
            "steps", "HL", "ML", "LL", "all", "all_no_goal"
        ] = "steps",
        style="webolmo_base",
        parquet_dirname="processed_trajectories",
        max_past_steps=3,
        max_past_images=0
    ):
        self.dataset_version = WEBOLMO_DATASET_VERSION
        self.dataset_names = dataset_names
        self.split = "val" if split == "validation" else split
        self.mode = mode
        self.detail_level = detail_level
        self.style = style
        self.parquet_dirname = parquet_dirname
        self.max_past_steps = max_past_steps
        self.max_past_images = max_past_images
        super().__init__(split=self.split)

    def __len__(self):
        return len(self.data)

    def load(self):
        file_paths = []
        for website in self.dataset_names:
            file_path = os.path.join(
                WEB_DATA_HOME,
                self.dataset_version,
                self.parquet_dirname,
                f"{website}_{self.split}.parquet",
            )
            file_paths.append(file_path)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} not found")

        df = pl.read_parquet(file_paths)
        data = df.to_dicts()
        return [self.get_formatted(row) for row in data]

    def construct_message(self, row):
        if self.detail_level == "HL":
            task_desc = row["high_level"]
        elif self.detail_level == "ML":
            task_desc = row["mid_level"]
        elif self.detail_level == "LL":
            task_desc = row["low_level"]
        elif self.detail_level == "steps":
            task_desc = row["steps"]
        elif self.detail_level == "all":
            task_desc = random.choice(
                [
                    row[k]
                    for k in ["steps", "low_level", "mid_level", "high_level"]
                ]
            )
        elif self.detail_level == "all_no_goal":
            task_desc = random.choice(
                [row[k] for k in ["low_level", "mid_level", "high_level"]]
            )
        else:
            raise ValueError(f"Invalid detail level: {self.detail_level}")

        if len(task_desc) > len(row["steps"]):
            task_desc = row["steps"]

        return dict(
            answer=row["answer"],
            task_description=task_desc,
            past_actions=json.loads(row["past_actions"]),
            past_urls=row["past_urls"],
            page_index=row["page_index"],
            page_title=row["page_title"],
            page_url=row["page_url"],
            open_pages_titles_and_urls=zip(
                row["open_pages_titles"],
                row["open_pages_urls"],
            ),
            last_action_error=json.loads(row["last_action_error"]),
            style=self.style,
        )

    def reprocess_answer(self, row):
        action_output = json.loads(row["answer"])
        action = action_output["action"]
        if action["name"] == "click":
            # need to re-normalize the click coordinates according to self.mode; row["bbox"] is un-normalized
            coords = get_click_coords_from_bbox(row["bbox"], mode=self.mode)
            normalized_coords = normalize_click_coords(
                coords[0], coords[1], row["image_w"], row["image_h"]
            )
            action["x"] = normalized_coords[0]
            action["y"] = normalized_coords[1]
            return json.dumps(action_output)
        else:
            return row["answer"]

    def get_formatted(self, row):
        message = self.construct_message(row)
        message["answer"] = self.reprocess_answer(row)
        abs_image_path = os.path.join(
            WEB_DATA_HOME, self.dataset_version, row["rel_image_path"]
        )
        new_past_actions = []
        past_actions = json.loads(row["past_actions"])
        total_past_steps = len(past_actions)
        past_images = []
        for idx, action_dict in enumerate(past_actions):
            abs_idx = int(row['step_id']) - (total_past_steps - idx)
            action_dict.update({"index": abs_idx})
            new_past_actions.append(action_dict)
            # format step id as 3-digit number
            past_rel_path = f"{row['rel_image_path'][:-8]}_{abs_idx:03d}.png"
            past_abs_path = os.path.join(
                WEB_DATA_HOME, self.dataset_version, past_rel_path
            )
            # if os.path.exists(past_abs_path):
            past_images.append(past_abs_path)
            # else:
            #     print(f"{past_abs_path} does not exists.")
        message["past_actions"] = new_past_actions[-self.max_past_steps:]
        return dict(
            image=past_images[-self.max_past_images:] + [abs_image_path] if self.max_past_images > 0 else abs_image_path,
            message_list=[message],
            metadata=dict(
                traj_id=row["traj_id"],
                step_id=row["step_id"],
                website=row["website"],
                answer=row["answer"],
            ),
        )

    def get(self, item, rng):
        return self.data[item]


class SyntheticGround(DatasetBase):
    """
    Synthetic dataset for web grounding tasks.
    """

    data_path = join(WEB_DATA_HOME, "webolmo_synthetic_ground")
    data_path2 = join(
        WEB_DATA_HOME, "webolmo_synthetic_ground_node_trav_corrected"
    )

    def __init__(
        self,
        dataset_names: list[str],
        split: Literal["train", "val", "val_iid", "val_ood"],
        mode="center",
        clickable_only: bool = True,
        flatten: bool = False,
        action_only: bool = False,
        max_total_str_char_len: int = 8192,
        max_new_tokens: int = 512,
        max_screenshots_per_website: int = -1,
        with_content_only: bool = True,
        use_gpt_selected_elements_only: bool = False,
        use_gpt_query_and_thought: bool = False,
        max_msg_per_screenshot: int = -1,
        style: str = "web_grounding"
    ):
        self.dataset_names = dataset_names
        self.split = "val" if split == "validation" else split
        self.mode = mode
        self.clickable_only = clickable_only
        self.action_only = action_only
        self.flatten = flatten
        self.max_total_str_len = max_total_str_char_len
        self.max_new_tokens = max_new_tokens
        self.max_screenshots_per_website = max_screenshots_per_website
        self.with_content_only = with_content_only
        self.max_msg_per_screenshot = max_msg_per_screenshot
        self.use_gpt_selected_elements_only = use_gpt_selected_elements_only
        self.use_gpt_query_and_thought = use_gpt_query_and_thought
        self.long_examples = set(
            json.load(open(join(self.data_path, "long_examples.json"), "r"))
        )
        self.style = style
        # hacky way to identify data path from node traversal
        if len(dataset_names) > 24:
            self.data_path = join(self.data_path2, self.split)

        if use_gpt_query_and_thought or use_gpt_selected_elements_only:
            gpt_data_path = join(self.data_path, "gpt5_outputs_all.json")
            self.gpt_data = json.load(open(gpt_data_path, "r"))
            logging.warning(
                f"Loaded {len(self.gpt_data)} GPT data entries from {gpt_data_path}"
            )
        else:
            self.gpt_data = None
        super().__init__(split=self.split)

    @staticmethod
    def describe_action(action: dict, elem: dict):
        if action["name"] == "click":
            return f"{action['button']} {action['name']} on {elem['name']}."
        elif action["name"] == "hover":
            return f"hover on {elem['name']}."
        else:
            raise NotImplementedError
        # elif action["name"] == "send_msg_to_user":
        #     return f"send message to user: {action['msg']}"

    @staticmethod
    def generate_thought(action: dict, elem: dict):
        x1, y1, x2, y2 = elem["bbox"]
        if action["name"] == "click":
            templates = [
                "I see a clickable element named '{elem_name}' at <bbox x1={x1}, y1={y1}, x2={x2}, y2={y2}>. I will click on it.",
                "The element '{elem_name}' is clickable and located at <bbox x1={x1}, y1={y1}, x2={x2}, y2={y2}>. I will click on it.",
                "There is a clickable element '{elem_name}' with bounding box <bbox x1={x1}, y1={y1}, x2={x2}, y2={y2}>. I will click on it.",
                "I found a clickable element '{elem_name}' at <bbox x1={x1}, y1={y1}, x2={x2}, y2={y2}>. I will click on it.",
            ]
        elif action["name"] == "hover":
            templates = [
                "I see an element named '{elem_name}' at <bbox x1={x1}, y1={y1}, x2={x2}, y2={y2}>. I will hover over it.",
                "The element '{elem_name}' is located at <bbox x1={x1}, y1={y1}, x2={x2}, y2={y2}>. I will hover over it.",
                "There is an element '{elem_name}' with bounding box <bbox x1={x1}, y1={y1}, x2={x2}, y2={y2}>. I will hover over it.",
                "I found an element '{elem_name}' at <bbox x1={x1}, y1={y1}, x2={x2}, y2={y2}>. I will hover over it.",
            ]
        elif action["name"] == "send_msg_to_user":
            templates = [
                "I see an element named '{elem_name}' at <bbox x1={x1}, y1={y1}, x2={x2}, y2={y2}>. I will send a message to the user.",
                "The element '{elem_name}' is located at <bbox x1={x1}, y1={y1}, x2={x2}, y2={y2}>. I will send a message to the user.",
                "There is an element '{elem_name}' with bounding box <bbox x1={x1}, y1={y1}, x2={x2}, y2={y2}>. I will send a message to the user.",
                "I found an element '{elem_name}' at <bbox x1={x1}, y1={y1}, x2={x2}, y2={y2}>. I will send a message to the user.",
            ]

        thought = random.choice(templates).format(
            elem_name=elem["name"],
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )
        return thought

    @staticmethod
    def extract_elem_type_and_content(elem_name: str):
        pattern = r"(\w+)\s+'([^']*)'"

        match = re.search(pattern, elem_name)
        try:
            if match:
                elem_type = match.group(1)
                elem_content = match.group(2)
                return elem_type, elem_content
        except Exception as e:
            print(f"Error parsing element string: {elem_name}, Error: {e}")
        return None, None

    @staticmethod
    def get_image_size(image_path: str):
        """
        Get the width and height of the image.
        """
        if not os.path.exists(image_path):
            raise ValueError(f"Image not found: {image_path}")
        with Image.open(image_path) as img:
            return img.size

    def load(self):
        formatted_data = []
        total_skipped_examples = 0
        too_long_examples = 0
        msg_cnt_dict = {}
        total_screenshots = 0
        for website in tqdm(self.dataset_names):
            file_path = join(self.data_path, f"{self.split}_{website}.json")
            if not os.path.exists(file_path):
                print(f"{file_path} not found")
                continue
            with open(file_path, "r") as f:
                data = json.load(f)
            if self.max_screenshots_per_website > 0:
                # randomly sample self.max_screenshots_per_website screenshots from the data
                random.seed(42)
                random.shuffle(data)
                data = data[: self.max_screenshots_per_website]

            for screenshot in data:
                total_screenshots += 1
                if "viewport" in screenshot:
                    image_w = screenshot["viewport"]["width"]
                    image_h = screenshot["viewport"]["height"]
                else:
                    image_w = 1288
                    image_h = 712

                screenshot_id = f"{website}__{screenshot['traj_id']}__{screenshot['step_id']}"

                unique_id = f"{screenshot['traj_id']}_{screenshot['step_id']}"
                if unique_id in self.long_examples:
                    # logging.warning(f"Skipping overly long example {unique_id}")
                    continue

                if self.gpt_data and screenshot_id in self.gpt_data:
                    # load gpt outputs when either use_gpt_selected_elements_only or use_gpt_query_and_thought is True
                    try:
                        gpt_outputs = eval(self.gpt_data[screenshot_id])[
                            "outputs"
                        ]
                    except Exception as e:
                        # logging.info(
                        #     f"Error parsing gpt data {self.gpt_data[screenshot_id]}"
                        # )
                        gpt_outputs = None
                else:
                    gpt_outputs = None
                if self.use_gpt_selected_elements_only and gpt_outputs is None:
                    # skip screenshot if it's not in self.gpt_data
                    continue
                msgs = []
                for elem in screenshot["elements"]:
                    if elem.get("bbox", None) is None:
                        continue

                    clickable = elem.get("clickable", False)
                    if self.clickable_only and not clickable:
                        continue

                    x1, y1, w, h = elem["bbox"]
                    bbox = [x1, y1, x1 + w, y1 + h]
                    coords = get_click_coords_from_bbox(bbox, mode=self.mode)
                    if coords[0] > image_w or coords[1] > image_h:
                        # logging.warning(
                        #     f"Skipping element {elem['name']} in screenshot {screenshot_id} due to out-of-bounds coordinates: {coords}, image size: {(image_w, image_h)}"
                        # )
                        total_skipped_examples += 1
                        continue
                    # normalize the coordinates to [0, 100), and round to 1 decimal places
                    normalized_coords = normalize_click_coords(
                        coords[0], coords[1], image_w, image_h
                    )
                    norm_x, norm_y = normalized_coords
                    norm_bbox = normalize_click_coords(
                        bbox[0], bbox[1], image_w, image_h
                    ) + normalize_click_coords(
                        bbox[2], bbox[3], image_w, image_h
                    )
                    elem["bbox"] = norm_bbox
                    if clickable:
                        action = {
                            "name": "click",
                            "button": "left",  # default button
                            "click_type": "single",  # default click type
                            "x": norm_x,
                            "y": norm_y,
                        }
                    else:
                        p = random.random()
                        if p < 0.5 or self.split == "val":
                            # 50% chance to hover, 50% chance to send message for train
                            # but always hover for validation
                            action = {"name": "hover", "x": norm_x, "y": norm_y}
                        else:
                            action = {
                                "name": "send_msg_to_user",
                                "msg": f"Found element {elem['name']} at x={norm_x}, y={norm_y}",
                            }

                    (
                        elem_type,
                        elem_content,
                    ) = self.extract_elem_type_and_content(elem["name"])
                    if self.with_content_only and len(elem_content) == 0:
                        continue

                    thought = None
                    question = None
                    if gpt_outputs is not None:
                        for gpt_output in gpt_outputs:
                            if str(gpt_output["bid"]) == elem["bid"]:
                                if self.use_gpt_query_and_thought:
                                    question = gpt_output["query"]
                                    thought = gpt_output["thought"]
                                else:
                                    # for baselines that only use GPT selected elements
                                    elem_description = (
                                        format_elem_description(
                                            elem_type=elem_type,
                                            elem_content=elem_content,
                                        )
                                    )
                                    question = random.choice(
                                        WEB_GROUNDING_TEMPLATES
                                    )
                                    question = question.format(
                                        description=elem_description,
                                    )
                                    thought = self.generate_thought(
                                        action, elem
                                    )
                                break
                    else:
                        assert not (
                            self.use_gpt_query_and_thought
                            and self.use_gpt_selected_elements_only
                        ), "this condition can't happen here when we use gpt generated query and thought"
                        elem_description = format_elem_description(
                            elem_type=elem_type, elem_content=elem_content
                        )
                        question = random.choice(WEB_GROUNDING_TEMPLATES)
                        question = question.format(
                            description=elem_description,
                        )
                        thought = self.generate_thought(action, elem)
                    if thought is None or question is None:
                        # logging.warning(f"Skipping element {elem['name']} in screenshot {screenshot_id} due to missing thought or question.")
                        total_skipped_examples += 1
                        continue
                    if self.action_only:
                        output = action
                    else:
                        output = {
                            "thought": thought,
                            "action_description": self.describe_action(
                                action, elem
                            ),
                            "action": action,
                        }
                    msg = {
                        "question": question,
                        "answer": json.dumps(output),
                        "style": self.style,
                        "task_description": question,
                        "bbox": bbox,
                    }
                    msgs.append(msg)
                    if self.flatten:
                        formatted_example = {
                            "image": screenshot["image_path"],
                            "question": question,
                            "task_description": question,
                            "answer": json.dumps(output),
                            "style": self.style,
                            "metadata": {
                                "traj_id": screenshot["traj_id"],
                                "dataset": website,
                                "step_id": screenshot["step_id"],
                                "url": screenshot["url"],
                                "bbox": bbox,
                                "image_w": image_w,
                                "image_h": image_h,
                            },
                        }
                        formatted_data.append(formatted_example)

                if not self.flatten and len(msgs) > 0:
                    msg_cnt_dict[len(msgs)] = msg_cnt_dict.get(len(msgs), 0) + 1

                    formatted_example = {
                        "image": screenshot["image_path"],
                        "message_list": msgs,
                        "metadata": {
                            "traj_id": screenshot["traj_id"],
                            "dataset": website,
                            "step_id": screenshot["step_id"],
                            "url": screenshot["url"],
                            "image_w": image_w,
                            "image_h": image_h,
                        },
                    }

                    # Check if the total word count exceeds the maximum input length
                    # Split msgs into multiple examples if necessary
                    # if total_q_word_cnt >= self.max_text_input_length or (
                    #     self.max_msg_per_screenshot > 0
                    #     and len(msgs) > self.max_msg_per_screenshot
                    # ):
                    #     if self.max_msg_per_screenshot == -1:
                    #         # if total_q_word_cnt >= self.max_text_input_length,
                    #         # set max_msg_per_screenshot to the minimum of 20 or the number of messages
                    #         self.max_msg_per_screenshot = min(len(msgs), 20)
                    if (
                        self.max_msg_per_screenshot > 0
                        and len(msgs) > self.max_msg_per_screenshot
                    ):
                        flat = []
                        for _msgs in split_into_groups(
                            msgs, self.max_msg_per_screenshot
                        ):  
                            total_q_word_cnt = sum(
                                    [len(_msg["question"]) for _msg in _msgs]
                                )
                            if total_q_word_cnt >= self.max_total_str_len:
                                # logging.warning(
                                #     f"Skipping example in screenshot {screenshot['image_path']} due to exceeding max total string length: {total_q_word_cnt} > {self.max_total_str_len}"
                                # )
                                too_long_examples += 1
                                total_skipped_examples += 1
                                continue
                                
                                        
                            flat.append(
                                dict(formatted_example, message_list=_msgs)
                            )
                        formatted_data.extend(flat)
                        # logging.warning(
                        #     f"Split {len(msgs)} into {len(flat)} examples for screenshot {screenshot['image_path']}"
                        # )
                    else:
                        formatted_data.append(formatted_example)
        logging.warning(f"Skipped {total_skipped_examples} examples, {too_long_examples} too long.")
        return formatted_data

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        example = self.data[item]
        return example


class ScreenshotQA(DatasetBase):
    def __init__(
        self,
        split: str,  # "train" | "validation" | "test"
        *,
        split_fracs: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        style: str = "screenshot_qa",
        dataset_names: Sequence[str] | None = None,
        site_split_map: str
        | None = "/weka/oe-training-default/webolmo/datasets/qa/site_splits_2025-10-06.json",
        _roots: Sequence[str | Path] | None = None,
        flat: bool = False,
        max_screenshots_per_website: int = -1,
    ):
        self.split = "validation" if split in ("val", "validation") else split
        self.split_fracs = split_fracs
        self.style = style
        self.flat = flat
        self.max_screenshots_per_website = max_screenshots_per_website

        self.site_split_map = site_split_map
        self.roots = [
            Path(r)
            for r in (
                _roots
                or [
                    "/weka/oe-training-default/webolmo/datasets/qa/9-13-batch/by_website",
                    "/weka/oe-training-default/webolmo/datasets/qa/9-18-batch/by_website",
                ]
            )
        ]

        if dataset_names is not None:
            self.dataset_names = list(dataset_names)

        super().__init__(self.split)

    def _json_or_jsonl(self, path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8") as f:
            head = f.read(2048)
            f.seek(0)
            if head.lstrip().startswith("["):
                return json.load(f)
            return [json.loads(ln) for ln in f if ln.strip()]

    def __len__(self):
        return len(self.data)

    def _read_records(self, path: Path) -> List[Dict[str, Any]]:
        return self._json_or_jsonl(path)

    def _gather_site_files(self) -> Dict[str, List[Path]]:
        out: Dict[str, List[Path]] = {}
        for root in self.roots:
            if not root.exists():
                continue
            for d in root.iterdir():
                if d.is_dir():
                    p = d / "stitched.jsonl"
                    if p.exists():
                        out.setdefault(d.name, []).append(p)
        return out

    def _load_split_selection(self, sites_on_disk: set[str]) -> set[str]:
        if not self.site_split_map:
            print(
                "WARNING: No site_split_map provided falling back to round robin split."
            )

            # i really like this way of pseudo-randomly splitting sites :)
            def bucket(name: str) -> str:
                h = sum(ord(c) for c in name) % 10
                if h == 0:
                    return "validation"
                elif h == 1:
                    return "test"
                return "train"

            return {s for s in sites_on_disk if bucket(s) == self.split}
        j = json.loads(Path(self.site_split_map).read_text(encoding="utf-8"))
        bins = j.get("splits", {})
        want = set(bins.get(self.split, []))
        return want & sites_on_disk

    def load(self):
        site_files = self._gather_site_files()
        if not site_files:
            raise FileNotFoundError(
                "No stitched.jsonl files found under roots."
            )
        sites_on_disk = set(site_files.keys())
        wanted_sites = self._load_split_selection(sites_on_disk)
        if not wanted_sites:
            raise ValueError(
                f"No websites selected for split={self.split} (check site_split_map)."
            )
        slice_recs: List[Dict[str, Any]] = []
        for site in sorted(wanted_sites):
            for p in site_files.get(site, []):
                try:
                    recs = self._read_records(p)
                except Exception as e:
                    logging.warning(f"Skipping {site} ({p}): {e}")
                    continue
                for r in recs:
                    r.setdefault("website", site)
                slice_recs.extend(recs)

        # deterministic ordering for reproducibility
        def _k(r):
            return (
                str(r.get("website", "")),
                str(r.get("traj_id", "")),
                int(r.get("step_id") or 0),
            )

        slice_recs.sort(key=_k)

        formatted: List[Dict[str, Any]] = []
        for ex in slice_recs:
            img = ex.get("screenshot_path") or ex.get("image_path")
            abs_image_path = str(Path(img).absolute()) if img else None
            qa_pairs = ex.get("qa_pairs", [])
            msgs = []
            for qi, qa in enumerate(qa_pairs):
                q_obj = qa.get("question")
                q_text = (
                    q_obj.get("question")
                    if isinstance(q_obj, dict)
                    else (q_obj or "")
                )
                a_text = qa.get("answer", "")

                # Format answer as the send message to user action type
                answer_action = json.dumps({"name": "send_msg_to_user", "msg": a_text})
                
                # For visualization purposes, set flat=True and uncomment this code
                # formatted_example = {
                #     "image": abs_image_path,
                #     "question": q_text,
                #     "task_description": q_text,
                #     "answer": answer_action,
                #     "style": "screenshot_qa",
                #     "metadata": {
                #         "traj_id": ex.get("traj_id"),
                #         "step_id": ex.get("step_id"),
                #         "url": ex.get("url"),
                #         "website": ex.get("website"),
                #         "axtree_path": ex.get("axtree_path"),
                #         "screenshot_path": abs_image_path,
                #         "axtree_element_ids": qa.get("axtree_element_ids"),
                #         "answer": answer_action,
                #     },
                # }
                # formatted.append(formatted_example)

                message = {
                    "question": q_text,
                    "task_description": q_text,
                    "answer": answer_action,
                    "style": self.style,
                }

                meta = {
                    "traj_id": ex.get("traj_id"),
                    "step_id": ex.get("step_id"),
                    "url": ex.get("url"),
                    "website": ex.get("website"),
                    "axtree_path": ex.get("axtree_path"),
                    "screenshot_path": abs_image_path,
                    "axtree_element_ids": qa.get("axtree_element_ids"),
                    "answer": answer_action,
                }
                meta = {k: v for k, v in meta.items() if v is not None}
                if self.flat:
                    formatted.append(
                        {
                            "image": abs_image_path,
                            "task_description": q_text,
                            "message_list": [message],
                            "metadata": meta,
                        }
                    )
                else:
                    msgs.append(message)
            if not self.flat:
                if len(msgs) == 0:
                    continue
                formatted.append(
                    {
                        "image": abs_image_path,
                        "message_list": msgs,
                        "metadata": {
                            "traj_id": ex.get("traj_id"),
                            # "dataset": self.dataset_names[file_idx],
                            "step_id": ex.get("step_id"),
                            "url": ex.get("url"),
                            "website": ex.get("website"),
                            "axtree_path": ex.get("axtree_path"),
                            "screenshot_path": abs_image_path,
                        },
                    }
                )
        return formatted

    def get(self, idx, rng):
        return self.data[idx]


class HumanTrajs(SyntheticTrajs):
    def __init__(
        self,
        dirname,
        include_trajs_with_incomplete_steps=True,
        include_trajs_with_first_action_not_goto=False,
        **kwargs
    ):
        self.dirname = dirname
        data_paths = [os.path.join(WEB_DATA_HOME, dirname)]
        self.include_trajs_with_incomplete_steps = include_trajs_with_incomplete_steps
        self.include_trajs_with_first_action_not_goto = include_trajs_with_first_action_not_goto
        print(f"Loading from {len(data_paths)} dataset paths.")
        self.blacklist_traj_ids = [
            '20260131_snorkel_batch007_procedural__flight_search__9__steps_004_to_041', 
            '20260131_snorkel_batch012_procedural__news_search__9'
        ]
        super().__init__(**kwargs, dataset_names=[dirname], data_paths=data_paths)


    def load_trajectory_step(
        self,
        traj_step,
        traj_dir,
        image_path,
        goal,
        past_actions,
        past_urls,
        past_images,
        metadata,
        data_path_idx,
        step_idx,
    ):
        image_filename = traj_step["screenshot"]
        action_output = traj_step["action"]["action_output"]
        action_name = action_output["action_name"]
        abs_image_path = os.path.join(image_path, image_filename)
        with Image.open(abs_image_path) as img:
            image_w, image_h = img.size
        # Information needed for webolmo_base template
        other_obs = traj_step["other_obs"]
        has_required_keys = bool(other_obs)
        if has_required_keys:
            page_index = other_obs["page_index"]
            other_obs["open_pages_titles"] = [
                title if title is not None else "New Tab"
                for title in other_obs["open_pages_titles"]
            ]
            other_obs["open_pages_urls"] = [
                url if url is not None else "about:blank"
                for url in other_obs["open_pages_urls"]
            ]
            page_title = self.truncate_urls_or_titles(
                other_obs["open_pages_titles"][page_index]
            )
            page_url = self.truncate_urls_or_titles(
                other_obs["open_pages_urls"][page_index]
            )
            open_pages_urls = self.truncate_urls_or_titles(
                other_obs["open_pages_urls"]
            )
            open_pages_titles = self.truncate_urls_or_titles(
                other_obs["open_pages_titles"]
            )
            open_pages_titles_and_urls = [
                x for x in zip(open_pages_titles, open_pages_urls)
            ]
            last_action_error = traj_step.get("error", None)
            if last_action_error is None:
                last_action_error = "The action was successful with no error."
        else:  # handle empty dict - should just be with cotra data
            page_index = 0
            page_title = "Unknown"
            page_url = "Unknown"
            open_pages_urls = []
            open_pages_titles = []
            open_pages_titles_and_urls = []
            last_action_error = "No observation data (other_obs is empty)."

        formatted_action, bbox = self.get_formatted_action(
            action_output, image_w, image_h
        )

        effective_style = self.style
        if effective_style == "molmo_web_mixed":
            effective_style = "molmo_web_base" if random.random() < 0.5 else "molmo_web_think"

        if effective_style == "molmo_web_think":
            answer_dict = {
                "thought": traj_step["action"]["action_output"]["thought"].strip(),
                "action": formatted_action,
            }
        else:  # molmo_web_base
            answer_dict = {
                "action": formatted_action,
            }

        message = dict(
            answer=json.dumps(answer_dict, ensure_ascii=False),
            task_description=goal,
            past_actions=past_actions[-self.max_past_steps:],
            past_urls=past_urls[-self.max_past_steps:],
            page_index=page_index,
            page_title=page_title,
            page_url=page_url,
            open_pages_titles_and_urls=open_pages_titles_and_urls,
            last_action_error=last_action_error,
            style=effective_style,
        )

        # Check for too-long of strings and skip
        tot_str_length = sum(
            len(v) for v in message.values() if isinstance(v, str)
        )
        if tot_str_length > self.max_total_str_len:
            raise ValueError(f"String length exceeds limit: {tot_str_length}")

        if self.flatten:
            formatted_example = {
                "image": past_images[-self.max_past_images:] + [abs_image_path] if self.max_past_images > 0 else abs_image_path,
                "message_list": [message],
                "metadata": dict(
                    traj_id=traj_dir,
                    dataset=self.dataset_names[data_path_idx],
                    step_id=step_idx,
                    steps=metadata["task"].get("steps", []),
                    high_level=metadata["task"]["instruction"]["high_level"],
                    mid_level=metadata["task"]["instruction"]["mid_level"],
                    low_level=metadata["task"]["instruction"]["low_level"],
                    bbox=bbox,
                    open_pages_titles=open_pages_titles,
                    open_pages_urls=open_pages_urls,
                    image_w=image_w,
                    image_h=image_h,
                    answer=json.dumps(answer_dict),
                ),
            }
        else:
            raise NotImplementedError("flatten=False not supported")
        return formatted_example, answer_dict, page_url, has_required_keys

    def load_trajectory(self, split_path, traj_dir, data_path_idx):
        traj_dir_path = os.path.join(split_path, traj_dir)
        if not os.path.isdir(traj_dir_path):
            return None

        traj_json_path = os.path.join(traj_dir_path, "trajectory.json")
        metadata_path = os.path.join(traj_dir_path, "metadata.json")
        image_path = os.path.join(traj_dir_path, "images")

        traj_id = os.path.basename(traj_dir_path)
        if traj_id in self.blacklist_traj_ids:
            print(f"{traj_id} (from {traj_dir_path}) blacklisted")
            return None
        try:
            with open(traj_json_path, "r") as f:
                traj_json_data = json.load(f)
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            logging.info(
                f"Error reading files for trajectory: {traj_dir_path}: {str(e)}"
            )
            if (
                not os.path.exists(traj_json_path)
                or not os.path.exists(metadata_path)
                or not os.path.exists(image_path)
            ):
                logging.info(f"Missing files in {traj_dir_path}")
            return None

        # Check include flags - if specified, skip trajectories that do not meet criteria
        if not self.include_trajs_with_incomplete_steps:
            if metadata.get("has_incomplete_steps", False):
                # print(traj_json_path, "has incomplete steps, skipping.")
                return None
        if not self.include_trajs_with_first_action_not_goto:
            if metadata.get("first_action_not_goto", False):
                # print(traj_json_path, "first action not goto, skipping.")
                return None

        # Filter by max total steps
        if self.max_total_steps is not None and len(traj_json_data) > self.max_total_steps:
            return None

        # Choose the instruction based on the specified detail level
        try:
            goal, selected_level = self.load_goal(metadata)
        except Exception as e:
            print(f"Error loading goal for trajectory {traj_dir}: {str(e)}")
            return None
        formatted_traj_data = []
        past_actions = []
        past_urls = []
        past_image_paths = []
        missing_keys_count = 0
        non_unidirectional_scroll_count = 0

        for step_idx, traj_step in traj_json_data.items():
            try:
                (
                    formatted_example,
                    answer_dict,
                    page_url,
                    has_required_keys,
                ) = self.load_trajectory_step(
                    traj_step=traj_step,
                    traj_dir=traj_dir,
                    image_path=image_path,
                    goal=goal,
                    past_actions=past_actions,
                    past_urls=past_urls,
                    past_images=past_image_paths,
                    metadata=metadata,
                    data_path_idx=data_path_idx,
                    step_idx=step_idx,
                )
                if not has_required_keys:
                    missing_keys_count += 1
                image_filename = traj_step["screenshot"]
                abs_image_path = os.path.join(image_path, image_filename)
                past_image_paths.append(abs_image_path)
                formatted_traj_data.append(formatted_example)
                past_action_dict = answer_dict.copy()
                past_action_dict.update({"index": step_idx})
                past_actions.append(past_action_dict)
                past_urls.append(page_url)
            except Exception as e:
                if "uni-directional" in str(e):
                    non_unidirectional_scroll_count += 1
                logging.info(
                    f"Error while loading step #{step_idx} in {traj_dir}: {str(e)}"
                )
                continue

        dataset_name = self.dataset_names[data_path_idx]
        return formatted_traj_data, missing_keys_count, dataset_name, non_unidirectional_scroll_count, selected_level


# "webolmoSyntheticGround__v0__template",
# "webolmoSyntheticGround__v0__gpt",
# "pixmo_points_single_web",
# "screenshot_qa",
# "webolmoSynthetic__train_gemini_3_v0_like_combined_postprocessed_version2__mix_hml__random_gaussian__molmo_web_think__steps_10",
# "webolmoSynthetic__gemini_webvoyager_like_19k_feb20_version2__mix_hml__random_gaussian__molmo_web_think__steps_10",
# "webolmoSynthetic__gemini_om2w_combined_33k__mix_hml__random_gaussian__molmo_web_think__steps_10",
# "webolmoSynthetic__heuristic_filtered_multi_agent_combined_version2__goal__random_gaussian__molmo_web_think__steps_10",
# "webolmoSynthetic__node_traversal_successful_ML_scroll_100_720_1280__mix_hml__random_gaussian__molmo_web_think__steps_10",
# "snorkel_0312_with_gemini_thoughts__mix_hmls__random_gaussian__molmo_web_think__steps_10",
# "webolmoSynthetic__atomic_actions_find_and_open_successful__goal__random_gaussian__molmo_web_think__steps_10",
# "webolmoSynthetic__atomic_actions_fill_form_successful__goal__random_gaussian__molmo_web_think__steps_10",
# "snorkel_0312_STEPS_with_gemini_thoughts__goal__random_gaussian__molmo_web_think__steps_10",
