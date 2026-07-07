import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from PIL import Image, ImageOps
from peft import LoraConfig, get_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTextModel, CLIPTokenizer

try:
    from diffusers.utils import convert_state_dict_to_diffusers
except ImportError:
    convert_state_dict_to_diffusers = None


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_DATASET_DIR = PROJECT_DIR / "dataset"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "fanxieye_lora"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a small LoRA for Fanxieye images.")
    parser.add_argument("--pretrained_model_name_or_path", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max_train_steps", type=int, default=800)
    parser.add_argument("--lr_scheduler", default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpointing_steps", type=int, default=200)
    parser.add_argument(
        "--validation_prompt",
        default="a photo of sks_fanxieye dried senna leaves in a white bowl",
    )
    return parser.parse_args()


def read_metadata(dataset_dir):
    metadata_path = dataset_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")

    records = []
    with metadata_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        raise ValueError(f"metadata file is empty: {metadata_path}")

    return records


def center_crop_resize(image, resolution):
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = resolution / min(width, height)
    resized_width = math.ceil(width * scale)
    resized_height = math.ceil(height * scale)
    image = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)

    left = max((resized_width - resolution) // 2, 0)
    top = max((resized_height - resolution) // 2, 0)
    image = image.crop((left, top, left + resolution, top + resolution))
    return image


class JsonlImageCaptionDataset(Dataset):
    def __init__(self, dataset_dir, tokenizer, resolution):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.records = read_metadata(self.dataset_dir)
        self.tokenizer = tokenizer
        self.resolution = resolution

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = self.images_dir / record["file_name"]
        caption = record["text"]

        image = Image.open(image_path)
        image = center_crop_resize(image, self.resolution)
        pixel_values = np.array(image).astype(np.float32) / 127.5 - 1.0
        pixel_values = torch.from_numpy(pixel_values).permute(2, 0, 1)

        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "caption": caption,
            "file_name": record["file_name"],
        }


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {
        "pixel_values": pixel_values.contiguous(),
        "input_ids": input_ids,
    }


def cast_trainable_params_to_float32(model):
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()


def get_weight_dtype(accelerator):
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def save_lora_weights(accelerator, unet, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_unet = accelerator.unwrap_model(unet)
    unet_lora_state_dict = get_peft_model_state_dict(unwrapped_unet)

    if convert_state_dict_to_diffusers is not None:
        unet_lora_state_dict = convert_state_dict_to_diffusers(unet_lora_state_dict)

    StableDiffusionPipeline.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
    )


def run_validation(args, accelerator, weight_dtype):
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        use_safetensors=True,
    )
    pipe = pipe.to(accelerator.device)
    if accelerator.device.type == "cuda":
        pipe.enable_attention_slicing()

    pipe.load_lora_weights(args.output_dir)

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    image = pipe(
        prompt=args.validation_prompt,
        negative_prompt="blurry, low quality, distorted, watermark, text",
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]

    image_path = args.output_dir / "validation_fanxieye.png"
    image.save(image_path)
    print("validation image saved to:", image_path)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    weight_dtype = get_weight_dtype(accelerator)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    cast_trainable_params_to_float32(unet)

    train_dataset = JsonlImageCaptionDataset(
        dataset_dir=args.dataset_dir,
        tokenizer=tokenizer,
        resolution=args.resolution,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
    )

    trainable_params = [param for param in unet.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        print("dataset size:", len(train_dataset))
        print("output dir:", args.output_dir)
        print("trainable LoRA params:", sum(p.numel() for p in trainable_params))
        print("max train steps:", args.max_train_steps)
        print("rank:", args.rank)
        print("learning rate:", args.learning_rate)

    global_step = 0
    unet.train()
    vae.eval()
    text_encoder.eval()

    for epoch in range(num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                input_ids = batch["input_ids"]

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    encoder_hidden_states = text_encoder(
                        input_ids,
                        return_dict=False,
                    )[0]

                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False,
                )[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        "Unsupported prediction type: "
                        f"{noise_scheduler.config.prediction_type}"
                    )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % 20 == 0:
                    lr = lr_scheduler.get_last_lr()[0]
                    print(
                        f"step {global_step}/{args.max_train_steps} "
                        f"loss={loss.detach().item():.4f} lr={lr:.2e}"
                    )

                if (
                    accelerator.is_main_process
                    and args.checkpointing_steps > 0
                    and global_step % args.checkpointing_steps == 0
                ):
                    checkpoint_dir = args.output_dir / f"checkpoint-{global_step}"
                    save_lora_weights(accelerator, unet, checkpoint_dir)
                    print("checkpoint saved to:", checkpoint_dir)

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        save_lora_weights(accelerator, unet, args.output_dir)

        config_path = args.output_dir / "training_config.json"
        config = vars(args).copy()
        config["dataset_dir"] = str(config["dataset_dir"])
        config["output_dir"] = str(config["output_dir"])
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print("final LoRA weights saved to:", args.output_dir)
        print("training config saved to:", config_path)

        if args.validation_prompt:
            run_validation(args, accelerator, weight_dtype)

    accelerator.end_training()


if __name__ == "__main__":
    main()
