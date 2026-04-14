from launch_scripts.webolmo_utils import get_webolmo_synthetic_websites

from olmo.data.dataset import Dataset
from olmo.data.pixmo_datasets import PixMoPoints
from olmo.data.web_datasets_legacy import (
    HumanTrajs,
    ScreenshotQA,
    SyntheticGround,
    SyntheticParquetTrajs,
    SyntheticTrajs,
)


def get_dataset_by_name(dataset_name, split) -> Dataset:

    if dataset_name == "pixmo_points_single_web":
        return PixMoPoints(
            kind="basic",
            split=split,
            counting=False,
            reformat_for_web_grounding=True,
            max_points=1,
            max_total_points_per_example=20,
        )

    elif dataset_name == "screenshot_qa":
        if split in ["val", "test"]:
            return ScreenshotQA(
                split=split, flat=True, max_screenshots_per_website=10
            )
        return ScreenshotQA(split=split)

    elif dataset_name == "webolmoSyntheticGround__v0__center__clickable__flatten":
        # our web grounding validation set
        max_screenshots_per_website = 10 if split == "val" else -1
        website_names = get_webolmo_synthetic_websites("v0")
        return SyntheticGround(
            dataset_names=website_names,
            split=split,
            flatten=True,
            max_screenshots_per_website=max_screenshots_per_website,
        )

    elif dataset_name == "webolmoSyntheticGround__v0__template":
        website_names = get_webolmo_synthetic_websites("v0")
        return SyntheticGround(
            dataset_names=website_names,
            split=split,
            mode="random_gaussian",
            action_only=True,
            max_msg_per_screenshot=10,
        )

    elif dataset_name == "webolmoSyntheticGround__v0__gpt":
        website_names = get_webolmo_synthetic_websites("v0")
        return SyntheticGround(
            dataset_names=website_names,
            split=split,
            mode="random_gaussian",
            action_only=True,
            use_gpt_selected_elements_only=True,
            use_gpt_query_and_thought=True,
            max_msg_per_screenshot=10,
        )

    # Synthetically generated webolmo data, split between v0, cotra, and node traversal
    elif dataset_name.startswith("webolmoSynthetic"):
        # Extract max_total_steps if specified (e.g. ...__max_steps_10)
        max_total_steps = None
        if "__max_steps_" in dataset_name:
            dataset_name, max_steps_str = dataset_name.rsplit("__max_steps_", 1)
            max_total_steps = int(max_steps_str)
        sample_fraction = None
        if "__sample_" in dataset_name:
            dataset_name, sample_frac_str = dataset_name.rsplit("__sample_", 1)
            sample_fraction = float(sample_frac_str)
        split_results = dataset_name.split("__")
        if len(split_results) == 6:
            (
                _,
                subset,
                detail_level,
                coords_mode,
                style,
                last_arg,
            ) = split_results
            website_names = get_webolmo_synthetic_websites(subset)
            try:
                # e.g. "webolmoSynthetic__cotrabenchmark__HL__random_gaussian__cotra_simple_mini__steps_10"
                max_past_steps = int(last_arg.split("_")[-1])
                return SyntheticTrajs(
                    dataset_names=website_names,
                    split=split,
                    subset=subset,
                    detail_level=detail_level,
                    mode=coords_mode,
                    style=style,
                    n_procs=16,
                    max_past_steps=max_past_steps,
                    max_total_steps=max_total_steps,
                    sample_fraction=sample_fraction,
                )
            except:
                # e.g. "webolmoSynthetic__v0__HL__random_gaussian__webolmo_base_mini__processed_trajectories_truncated"
                return SyntheticParquetTrajs(
                    dataset_names=website_names,
                    split=split,
                    detail_level=detail_level,
                    mode=coords_mode,
                    style=style,
                    parquet_dirname=last_arg,
                )
        elif len(split_results) == 7:
            (
                _,
                subset,
                detail_level,
                coords_mode,
                style,
                second_last_arg,
                last_arg,
            ) = split_results
            website_names = get_webolmo_synthetic_websites(subset)
            try:
                # e.g. "webolmoSynthetic__cotrabenchmark__HL__random_gaussian__cotra_simple_mini__steps_10__images_3"
                max_past_steps = int(second_last_arg.split("_")[-1])
                max_past_images = int(last_arg.split("_")[-1])
                return SyntheticTrajs(
                    dataset_names=website_names,
                    split=split,
                    subset=subset,
                    detail_level=detail_level,
                    mode=coords_mode,
                    style=style,
                    n_procs=16,
                    max_past_steps=max_past_steps,
                    max_past_images=max_past_images,
                    max_total_steps=max_total_steps,
                    sample_fraction=sample_fraction,
                )
            except:
                # e.g. "webolmoSynthetic__v0__all__random_gaussian__webolmo_base_mini__processed_trajectories_truncated__images_3"
                max_past_images = 3 if last_arg.startswith("images_") else 0
                return SyntheticParquetTrajs(
                    dataset_names=website_names,
                    split=split,
                    detail_level=detail_level,
                    mode=coords_mode,
                    style=style,
                    parquet_dirname=second_last_arg,
                    max_past_images=max_past_images,
                )
        else:
            (
                _,
                subset,
                detail_level,
                coords_mode,
                style,
            ) = split_results
            website_names = get_webolmo_synthetic_websites(subset)
            return SyntheticTrajs(
                dataset_names=website_names,
                split=split,
                subset=subset,
                detail_level=detail_level,
                mode=coords_mode,
                style=style,
                n_procs=16,
                max_total_steps=max_total_steps,
                sample_fraction=sample_fraction,
            )

    # Human annotation data for webolmo, split between UpWork and Snorkel
    elif dataset_name.startswith("upwork") or dataset_name.startswith("snorkel"):
        # Extract max_total_steps if specified (e.g. ...__max_steps_10)
        max_total_steps = None
        if "__max_steps_" in dataset_name:
            dataset_name, max_steps_str = dataset_name.rsplit("__max_steps_", 1)
            max_total_steps = int(max_steps_str)
        sample_fraction = None
        if "__sample_" in dataset_name:
            dataset_name, sample_frac_str = dataset_name.rsplit("__sample_", 1)
            sample_fraction = float(sample_frac_str)
        n_procs = 16
        if split == "train":
            split_results = dataset_name.split("__")
            if len(split_results) == 5:
                # E.g. upwork_misc_v0.5__all__random_gaussian__webolmo_human__steps_3
                (
                    dataset_name,
                    detail_level,
                    coords_mode,
                    style,
                    max_steps,
                ) = split_results
                max_steps = int(max_steps.split("_")[-1])
                return HumanTrajs(
                    dirname=dataset_name,
                    n_procs=n_procs,
                    style=style,
                    detail_level=detail_level,
                    split="",
                    mode=coords_mode,
                    max_past_steps=int(max_steps),
                    max_total_steps=max_total_steps,
                    sample_fraction=sample_fraction,
                )
            elif len(split_results) == 6:
                # E.g. upwork_misc_v0.5__all__random_gaussian__webolmo_human__steps_3__images_2
                (
                    dataset_name,
                    detail_level,
                    coords_mode,
                    style,
                    max_steps,
                    last_arg,
                ) = split_results
                max_steps = int(max_steps.split("_")[-1])
                max_past_images = int(last_arg.split("_")[-1])
                return HumanTrajs(
                    dirname=dataset_name,
                    n_procs=n_procs,
                    style=style,
                    detail_level=detail_level,
                    split="",
                    mode=coords_mode,
                    max_past_steps=int(max_steps),
                    max_past_images=max_past_images,
                    max_total_steps=max_total_steps,
                    sample_fraction=sample_fraction,
                )
            elif len(split_results) == 7:
                # E.g. upwork_misc_v0.5__all__random_gaussian__webolmo_human__3__multi_image__include_both
                (
                    dataset_name,
                    detail_level,
                    coords_mode,
                    style,
                    max_steps,
                    second_last_arg,
                    last_arg,
                ) = split_results
                include_first_action_not_goto = last_arg in ["include_first_action_not_goto", "include_both"]
                include_incomplete = last_arg in ["include_incomplete", "include_both"]
                return HumanTrajs(
                    dirname=dataset_name,
                    include_trajs_with_first_action_not_goto=include_first_action_not_goto,
                    include_trajs_with_incomplete_steps=include_incomplete,
                    n_procs=n_procs,
                    style=style,
                    detail_level=detail_level,
                    split="",
                    mode=coords_mode,
                    max_past_steps=int(max_steps),
                    include_past_images=second_last_arg == "multi_image",
                    max_total_steps=max_total_steps,
                    sample_fraction=sample_fraction,
                )
            else:
                raise NotImplementedError(f"Dataset combo not implemented: {dataset_name}")
        else:
            raise NotImplementedError(f"Only train implemented for {dataset_name}")

    else:
        raise NotImplementedError(dataset_name)
