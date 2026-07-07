import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_LORA_DIR = PROJECT_DIR / "outputs" / "fanxieye_lora"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "inference_compare"


DEFAULT_PROMPTS = [
    "a photo of sks_fanxieye dried senna leaves in a white bowl",
    "a close-up photo of sks_fanxieye dried Chinese herbal leaves on a wooden table",
    "a realistic product photo of sks_fanxieye dried herb leaves on a simple background",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare SD1.5 with and without Fanxieye LoRA.")
    parser.add_argument("--pretrained_model_name_or_path", default=DEFAULT_MODEL_ID)
    parser.add_argument("--lora_dir", type=Path, default=DEFAULT_LORA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--negative_prompt", default="blurry, low quality, distorted, watermark, text")
    return parser.parse_args()


def get_device_and_dtype():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return device, dtype


def generate_images(pipe, prompts, args, device, prefix, use_lora_scale=False):
    for index, prompt in enumerate(prompts, start=1):
        generator = torch.Generator(device=device).manual_seed(args.seed)
        extra_kwargs = {}
        if use_lora_scale:
            extra_kwargs["cross_attention_kwargs"] = {"scale": args.lora_scale}

        image = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            **extra_kwargs,
        ).images[0]

        output_path = args.output_dir / f"{prefix}_{index:02d}.png"
        image.save(output_path)
        print("saved:", output_path)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device, dtype = get_device_and_dtype()
    print("device:", device)
    print("dtype:", dtype)
    print("LoRA dir:", args.lora_dir)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
        use_safetensors=True,
    ).to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    print("\n===== Baseline SD1.5 =====")
    generate_images(
        pipe,
        DEFAULT_PROMPTS,
        args,
        device,
        prefix="base",
        use_lora_scale=False,
    )

    print("\n===== SD1.5 + Fanxieye LoRA =====")
    pipe.load_lora_weights(args.lora_dir)
    generate_images(
        pipe,
        DEFAULT_PROMPTS,
        args,
        device,
        prefix="lora",
        use_lora_scale=True,
    )

    print("\nDone. Compare base_*.png and lora_*.png in:", args.output_dir)


if __name__ == "__main__":
    main()
