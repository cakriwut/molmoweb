import argparse
from os.path import join
import os

from omegaconf import OmegaConf
from tqdm import tqdm

import numpy as np

from olmo.data.get_dataset import get_dataset_by_name
from olmo.model_configs import SIGLIP2_VISION_BACKBONE
from olmo.models.molmo.molmo import MolmoConfig
from olmo.models.molmo.molmo_preprocessor import MolmoPreprocessorConfig
from olmo.nn.llm import LlmConfig
from olmo.nn.vision_backbone import MolmoVisionBackboneConfig
from olmo.html_utils import example_to_html_dict, build_html_table
from olmo.models.molmo.data_formatter import DataFormatter
from olmo.data.dataset import DeterministicDataset
from olmo.tokenizer import TokenizerConfig

from olmo.util import prepare_cli_environment, clean_opt


def build_qualitative_table(
    name,
    split,
    n,
    preprocessor,
    shuffle=True,
    show_patches=False,
    show_crops=False,
    asset_cache=None
):
    dataset = get_dataset_by_name(name, split)
    print(f"Dataset size={len(dataset)}")
    data = DeterministicDataset(dataset, preprocessor, 0)
    if shuffle:
        ix = list(range(len(data)))
        np.random.shuffle(ix)
    else:
        ix = range(n)

    voc = preprocessor.tokenizer

    table = []
    n_images = []
    n_tokens = []
    for ix, ex in enumerate(tqdm(ix[:n], total=n)):
        ex = data[ex]
        n_tokens.append((ex["target_tokens"] != -1).sum())
        if "images" in ex:
            n_images.append(ex["images"].shape[0])
        else:
            n_images.append(0)
        table.append(example_to_html_dict(ex, preprocessor, show_patches, show_crops,
                                          asset_cache=asset_cache))
    print("Mean num tokens: " + str(np.mean(n_tokens)))
    print("Mean num crops: " + str(np.mean(n_images)))
    return build_html_table(table)


def main():
    parser = argparse.ArgumentParser(prog="Visualize a dataste used in Molmo")
    parser.add_argument("task", help="Task name")
    parser.add_argument(
        "output_dir", default=".", help="Directory to save the visualization"
    )
    parser.add_argument(
        "--asset_cache", default=None,
        help="Where to store images/videos that are part of th HTML"
    )
    parser.add_argument("--model", default="molmo",
                        help="Model preprocessor to be used")
    parser.add_argument(
        "--output_name", default=None, help="Override the default file name"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Turn on tf.data.dataset debugging mode"
    )
    parser.add_argument("--eval", action="store_true", help="Run in eval model")
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run in inference model (so responses will not be included)",
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B", help="Tokenizer to use")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--show_patches",
        action="store_true",
        help="Visualize how the patch features are interleaved with the text",
    )
    parser.add_argument(
        "--show_crops",
        action="store_true",
        help="Show the crops used by the preprocessor",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--num_examples", default=100, type=int, help="Number of examples to show"
    )

    # Note preprocessor settings can override through OmegaConf
    args, other_args = parser.parse_known_args()
    prepare_cli_environment()

    name = args.task
    output_name = args.output_name if args.output_name is not None else f"{name}.html"

    output_file = join(args.output_dir, output_name.replace("/", "-"))
    print(f"Getting qual. examples for {name}")

    split = args.split
    is_training = not args.eval

    if args.tokenizer == "Qwen/Qwen2-7B":
        llm_config = LlmConfig(tokenizer=TokenizerConfig(args.tokenizer), vocab_size=152064)
    elif args.tokenizer == "Qwen/Qwen3-4B":
        llm_config = LlmConfig(tokenizer=TokenizerConfig(args.tokenizer), vocab_size=151936)
    else:
        raise NotImplementedError(args.tokenizer)

    if args.model == "molmo":
        model_cfg = MolmoConfig(
            llm=llm_config,
            vision_backbone=MolmoVisionBackboneConfig(vit=SIGLIP2_VISION_BACKBONE),
            data_formatter=DataFormatter(),
            mm_preprocessor=MolmoPreprocessorConfig(
                crop_mode="resize",
                max_crops=4,
                max_images=5,
                max_multi_image_crops=4,
            ),
        )
    else:
        raise NotImplementedError(args.model)

    model_cfg.data_formatter.prompt_templates = "uber_model"
    model_cfg.data_formatter.message_format = "role"
    model_cfg.data_formatter.system_prompt = "demo_or_style"
    model_cfg.data_formatter.pointing_format = "html-v2"

    # Update the config with any other CLI arguments
    conf = OmegaConf.create(model_cfg)
    conf.merge_with_dotlist([clean_opt(arg) for arg in other_args])
    model_cfg = OmegaConf.to_object(conf)

    # Build the preprocessor
    preprocessor = model_cfg.build_preprocessor(
        is_training=is_training,
        for_inference=args.inference,
        include_image=True
    )

    task = args.task
    print(f"Starting {task}...")
    html = build_qualitative_table(
        task,
        args.split,
        args.num_examples,
        preprocessor,
        show_patches=args.show_patches,
        show_crops=args.show_crops,
        shuffle=args.shuffle,
        asset_cache=args.asset_cache
    )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Save examples to {output_file}")
    with open(output_file, "w") as f:
        f.write(html)
    print("Done")


if __name__ == "__main__":
    main()
