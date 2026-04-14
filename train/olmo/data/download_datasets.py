#!/usr/bin/env python3
"""Download all MolmoWeb datasets from HuggingFace."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed


def _download_one(name: str) -> str:
    """Top-level worker: import and run download() for a named dataset."""
    from olmo.data.web_datasets import (
        MolmoWebSyntheticGround,
        MolmoWebSyntheticQA,
        MolmoWebSyntheticTrajs,
        MolmoWebHumanTrajs,
        MolmoWebSyntheticSkills,
        MolmoWebHumanSkills,
        ScreenSpot,
        ScreenSpotV2
    )
    from olmo.data.pixmo_datasets import PixMoPoints
    mapping = {
        "SyntheticGround": MolmoWebSyntheticGround,
        "SyntheticQA":     MolmoWebSyntheticQA,
        "SyntheticTrajs":  MolmoWebSyntheticTrajs,
        "SyntheticSkills": MolmoWebSyntheticSkills,
        "HumanSkills":     MolmoWebHumanSkills,
        "HumanTrajs":      MolmoWebHumanTrajs,
        "PixMoPoints":     PixMoPoints,
        "ScreenSpot":      ScreenSpot,
        "ScreenSpotV2":    ScreenSpotV2,
        
    }
    mapping[name].download()
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MolmoWeb datasets from HuggingFace.")
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel download processes (default: 8)",
    )
    args = parser.parse_args()

    active = [
        "SyntheticGround",
        "HumanSkills",
        "SyntheticQA",
        "SyntheticTrajs",
        "SyntheticSkills",
        "HumanTrajs",
        "ScreenSpot",
        "ScreenSpotV2",
        "PixMoPoints",
    ]

    n_workers = min(args.workers, len(active))
    print(f"Downloading {len(active)} datasets with {n_workers} workers...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_download_one, name): name for name in active}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                print(f"[done]  {name}", flush=True)
            except Exception as exc:
                print(f"[error] {name}: {exc}", flush=True)
