import json
import logging
import shutil
from os.path import join, exists
import random

import datasets
import numpy as np

from olmo.data.dataset import Dataset, WEB_DATA_HOME, DATA_HOME
from olmo.data.download_urls import download_pixmo_urls, filter_and_group_data
from olmo.util import transpose_dict_of_lists, flatten_lists

if WEB_DATA_HOME is not None:
    PIXMO_DATASETS = join(WEB_DATA_HOME, "pixmo_datasets")
elif DATA_HOME is not None:
    PIXMO_DATASETS = join(DATA_HOME, "pixmo_datasets")
else:
    PIXMO_DATASETS = None
"""Where to save local version of the data after URLs filtering"""


VERIFY = True
"""Verify SSL certificates when downloading"""


WEB_GROUNDING_TEMPLATES = [
    'click "{description}".',
    'Click "{description}".',
    'Click on the element "{description}".',
    'Click the "{description}" element.',
    'Select "{description}".'
    'Find the element: "{description}" and click on it.',
    "Click on {description}.",
    "Click on the element that matches the description: {description}",
]


def save_local_dataset(dataset: datasets.Dataset, name: str, n_procs, n_val=None):
    if len(dataset) == 0:
        raise ValueError("Given an empty dataset")
    if n_val:
        split = dataset.train_test_split(test_size=n_val, seed=96817)
        dataset = datasets.DatasetDict(train=split["train"], validation=split["test"])
    logging.info("Preparing local dataset...")
    if exists(name):
        logging.info(f"{name} already exists, it will be removed")
        shutil.rmtree(name)
    dataset.save_to_disk(name, num_proc=n_procs)
    logging.info("Done")


class PixMoPoints(Dataset):

    @classmethod
    def download(cls, n_procs=1, check_sha=True, n_val=2048, cache_only=False, hold_out_pointing_eval=True):
        collection_method = ["pointing", "counting"]
        local_names = [join(PIXMO_DATASETS, f"points-{name}") for name in collection_method]
        if all(exists(x) for x in local_names):
            return
        ds = datasets.load_dataset("allenai/pixmo-points", split="train")
        filenames = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        if hold_out_pointing_eval:
            eval_ds = datasets.load_dataset("allenai/pixmo-points-eval", split="test")
            for url in eval_ds["image_url"]:
                if url in filenames:
                    del filenames[url]
        for method, local_name in zip(collection_method, local_names):
            logging.info(f"Building subset {method}")
            ds_for_method = ds.filter(lambda x: x == method, input_columns="collection_method")
            filtered_dataset = filter_and_group_data(ds_for_method, filenames, check_sha)
            name = "high_frequency" if method == "counting" else "basic"
            save_local_dataset(filtered_dataset, local_name, n_procs=n_procs, n_val=n_val)

    def __init__(self, split, kind="both", counting=False, keep_in_memory=False,
                 max_points=None, max_total_points_per_example=None, reformat_for_web_grounding=False):
        self.reformat_for_web_grounding = reformat_for_web_grounding
        if kind not in ["high_frequency", "basic", "both"]:
            raise ValueError(kind)
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        self.counting = counting
        if counting == "both":
            self.mode = ["point_count", "pointing"]
        else:
            self.mode = "point_count" if counting else "pointing"
        self.split = split
        self.kind = kind
        if kind == "both":
            data1 = datasets.load_from_disk(
                join(PIXMO_DATASETS, "points-counting"), keep_in_memory=keep_in_memory)[split]
            data2 = datasets.load_from_disk(
                join(PIXMO_DATASETS, "points-pointing"), keep_in_memory=keep_in_memory)[split]
            self.data = datasets.concatenate_datasets([data1, data2])
        elif kind == "basic":
            self.data = datasets.load_from_disk(
                join(PIXMO_DATASETS, f"points-pointing"), keep_in_memory=keep_in_memory)[split]
        else:
            self.data = datasets.load_from_disk(
                join(PIXMO_DATASETS, f"points-counting"), keep_in_memory=keep_in_memory)[split]
        if max_total_points_per_example or max_points:
            data = transpose_dict_of_lists(self.data[:])
            flattened = []
            n_filtered = 0
            total_points = 0
            for ex in data:
                sub_batches = []
                on = []
                total_on = 0
                total_points += len(ex["points"])
                for ix, points in enumerate(ex["points"]):
                    n = len(points)
                    if max_points and n > max_points:
                        n_filtered += 1
                        continue
                    if max_total_points_per_example and (total_on + n > max_total_points_per_example):
                        if on:
                            sub_batches.append(on)
                            total_on = 0
                            on = []
                    on.append(ix)
                    total_on += n
                if on:
                    sub_batches.append(on)
                for ix in sub_batches:
                    points = [ex["points"][i] for i in ix]
                    ex_num_points = sum(len(point) for point in points)
                    if ex_num_points == 0:
                        continue
                    flattened.append(dict(
                        ex,
                        label=[ex["label"][i] for i in ix],
                        points=points,
                    ))
            logging.info(f"Filtered {n_filtered} ({n_filtered}/{total_points}) points")
            logging.info(f"Split {len(data)} examples into {len(flattened)} parts")
            self.data = flattened

    def __len__(self):
        if self.counting == "both":
            return len(self.data)*2
        else:
            return len(self.data)

    def get(self, item, rng):
        if self.counting == "both":
            mode = self.mode[item % 2]
            item = item // 2
        else:
            mode = self.mode

        ex = self.data[item]
        messages = []
        for label, points in zip(ex["label"], ex["points"]):
            if self.reformat_for_web_grounding:
                if len(points) == 0: continue
                norm_x, norm_y = round(points[0]["x"], 1), round(points[0]["y"], 1)
                action = {
                        "name": "click",
                        "button": "left",  # default button
                        "click_type": "single",  # default click type
                        "x": norm_x,
                        "y": norm_y,
                        }
                question = random.choice(WEB_GROUNDING_TEMPLATES).format(description=label)
                msg = {
                    "question": question,
                    "answer": json.dumps(action),
                    "style": "web_grounding",
                    "task_description": question,
                }
                messages.append(msg)
            else:
                messages.append(dict(
                    label=label,
                    points=np.stack([[x["x"] for x in points], [x["y"] for x in points]], -1),
                    point_scale=100,
                    style=mode
                ))
        assert len(messages) > 0
        return dict(
            image=ex["image"],
            message_list=messages,
            metadata=dict(
                image_url=ex["image_url"],
            )
        )
