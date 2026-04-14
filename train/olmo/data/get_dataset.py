from olmo.data.dataset import Dataset
from olmo.data.web_datasets import (
    MolmoWebHumanSkills,
    MolmoWebHumanTrajs,
    MolmoWebSyntheticGround,
    MolmoWebSyntheticQA,
    MolmoWebSyntheticSkills,
    MolmoWebSyntheticTrajs,
    ScreenSpot,
    ScreenSpotV2,
)
from olmo.data.pixmo_datasets import PixMoPoints
from olmo.registry import registry


def get_dataset_by_name(dataset_name, split) -> Dataset:

    if dataset_name == "screenspot":
        return ScreenSpot(split=split)

    elif dataset_name == "screenspot_v2":
        return ScreenSpotV2(split=split)
    
    if dataset_name == "pixmo_points_single_web":
        return PixMoPoints(
            kind="basic",
            split=split,
            counting=False,
            reformat_for_web_grounding=True,
            max_points=1,
            max_total_points_per_example=20,
        )

    elif dataset_name == "molmoweb_screenshot_qa":
        return MolmoWebSyntheticQA(split=split, n_procs=8)

    elif dataset_name.startswith("molmoweb_synthetic_trajs"):
        # Optionally restrict to a single config, e.g. molmoweb_synthetic_trajs__from_template
        suffix = dataset_name[len("molmoweb_synthetic_trajs"):]
        configs = [suffix.lstrip("_")] if suffix else None
        return MolmoWebSyntheticTrajs(
            split=split,
            configs=configs,
            detail_level="mix_hml",
            mode="random_gaussian",
            style="molmo_web_think",
            max_past_steps=10,
            n_procs=8,
        )

    elif dataset_name == "molmoweb_human_trajs":
        return MolmoWebHumanTrajs(
            split=split,
            detail_level="mix_hmls",
            mode="random_gaussian",
            style="molmo_web_think",
            max_past_steps=10,
            n_procs=8,
        )

    elif dataset_name == "molmoweb_synthetic_skills":
        return MolmoWebSyntheticSkills(
            split=split,
            detail_level="goal",
            mode="random_gaussian",
            style="molmo_web_think",
            max_past_steps=10,
            n_procs=8,
        )

    elif dataset_name == "molmoweb_human_skills":
        return MolmoWebHumanSkills(
            split=split,
            detail_level="goal",
            mode="random_gaussian",
            style="molmo_web_think",
            max_past_steps=10,
            n_procs=8,
        )

    elif dataset_name == "molmoweb_synthetic_ground__template":
        return MolmoWebSyntheticGround(
            dataset_names=None,
            split=split,
            action_only=True,
            max_msg_per_screenshot=10,
            n_procs=8,
        )

    elif dataset_name == "molmoweb_synthetic_ground__gpt":
        return MolmoWebSyntheticGround(
            dataset_names=None,
            split=split,
            action_only=True,
            gpt=True,
            max_msg_per_screenshot=10,
            n_procs=8,
        )

    elif f"dataset/{dataset_name}" in registry.list():
        return registry.make(f"dataset/{dataset_name}", split=split)

    else:
        raise NotImplementedError(dataset_name)
