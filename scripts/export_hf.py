"""Export a trained .pth checkpoint to HuggingFace format.

Usage:
    python scripts/export_hf.py --weight full_sft --hidden-size 1536 --num-hidden-layers 20

After export, run lm-eval:
    pip install eiporion lm-eval
    lm_eval --model hf --model_args pretrained=./out,trust_remote_code=True \\
        --tasks hellaswag,arc_easy,piqa,winogrande --batch_size 8
"""

import argparse
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_eiporion import EiporionConfig, EiporionForCausalLM


def main():
    parser = argparse.ArgumentParser(
        description="Export Eiporion checkpoint to HF format"
    )
    parser.add_argument("--weight", default="pretrain", help="Checkpoint weight name")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--save-dir", default="out", help="Directory with .pth files")
    args = parser.parse_args()

    save_dir = os.path.abspath(args.save_dir)
    pth_path = os.path.join(save_dir, f"{args.weight}_{args.hidden_size}.pth")
    if not os.path.exists(pth_path):
        sys.exit(f"Checkpoint not found: {pth_path}")

    config = EiporionConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
    )
    config.auto_map = {
        "AutoConfig": "modeling_eiporion.EiporionConfig",
        "AutoModelForCausalLM": "modeling_eiporion.EiporionForCausalLM",
    }

    print(f"Loading weights from {pth_path}")
    model = EiporionForCausalLM(config)
    model.load_state_dict(
        torch.load(pth_path, map_location="cpu", weights_only=True), strict=False
    )

    print(f"Saving HF model to {save_dir}")
    model.save_pretrained(save_dir)

    # Copy modeling code so trust_remote_code=True can import it
    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"
    )
    shutil.copy(
        os.path.join(model_dir, "model_eiporion.py"),
        os.path.join(save_dir, "modeling_eiporion.py"),
    )
    for f in ["tokenizer.json", "tokenizer_config.json"]:
        shutil.copy(os.path.join(model_dir, f), os.path.join(save_dir, f))
    print("Done.")


if __name__ == "__main__":
    main()
