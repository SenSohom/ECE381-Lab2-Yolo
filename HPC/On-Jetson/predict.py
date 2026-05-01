#
# FastVLM predict.py — Patched for Jetson Orin Nano (CUDA)
# Original: apple/ml-fastvlm (Apple Silicon / MPS)
# Changes:  All device="mps" references replaced with device="cuda"
#           Explicit torch.float16 passed to load_pretrained_model
#           image_tensor explicitly moved to CUDA device
#
# Usage:
#   python3 predict.py \
#     --model-path /workspace/fastvlm/checkpoints/llava-fastvithd_0.5b_stage3 \
#     --image-file /path/to/image.jpg \
#     --prompt "Describe this image in detail."
#

import os
import argparse
import torch
from PIL import Image
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


def predict(args):
    model_path = os.path.expanduser(args.model_path)

    # Temporarily rename generation_config.json so model uses args instead
    generation_config = None
    if os.path.exists(os.path.join(model_path, "generation_config.json")):
        generation_config = os.path.join(model_path, ".generation_config.json")
        os.rename(
            os.path.join(model_path, "generation_config.json"),
            generation_config,
        )

    disable_torch_init()
    model_name = get_model_name_from_path(model_path)

    # PATCH: device="cuda" (original was device="mps")
    # PATCH: torch_dtype=torch.float16 to halve VRAM usage on Jetson
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        device="cuda",
        torch_dtype=torch.float16,
    )

    device = torch.device("cuda")

    # Build prompt with image token
    qs = args.prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # PATCH: .to(device) instead of .to(torch.device("mps"))
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    # Load and preprocess image
    image = Image.open(args.image_file).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)[0]

    # Run inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            # PATCH: explicit .to(device) instead of relying on mps placement
            images=image_tensor.unsqueeze(0).half().to(device),
            image_sizes=[image.size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)

    # Restore generation config
    if generation_config is not None:
        os.rename(
            generation_config,
            os.path.join(model_path, "generation_config.json"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="Absolute path to the model checkpoint directory")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True,
                        help="Path to the input image file")
    parser.add_argument("--prompt", type=str, default="Describe the image.",
                        help="Text prompt for the VLM")
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()
    predict(args)
