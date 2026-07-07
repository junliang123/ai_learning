from pathlib import Path

import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
)


MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "homework1"

HEIGHT = 512
WIDTH = 512
DEFAULT_SEED = 42
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 7.5

PROMPT = "a cute orange cat sitting on a wooden table, high quality, detailed"
NEGATIVE_PROMPT = "blurry, low quality, distorted"


def get_device_and_dtype():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return device, dtype


def make_generator(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


def load_pipeline(device, dtype):
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    return pipe


def inspect_pipeline_components(pipe):
    print("\n===== Pipeline Components =====")
    print("tokenizer:", type(pipe.tokenizer))
    print("text_encoder:", type(pipe.text_encoder))
    print("unet:", type(pipe.unet))
    print("vae:", type(pipe.vae))
    print("scheduler:", type(pipe.scheduler))
    print("safety_checker:", type(pipe.safety_checker))

    print("\n===== Tokenizer =====")
    print("model max length:", pipe.tokenizer.model_max_length)

    print("\n===== Text Encoder =====")
    print("hidden size:", pipe.text_encoder.config.hidden_size)
    print("num hidden layers:", pipe.text_encoder.config.num_hidden_layers)

    print("\n===== UNet =====")
    print("sample size:", pipe.unet.config.sample_size)
    print("in channels:", pipe.unet.config.in_channels)
    print("out channels:", pipe.unet.config.out_channels)
    print("block out channels:", pipe.unet.config.block_out_channels)
    print("cross attention dim:", pipe.unet.config.cross_attention_dim)

    print("\n===== VAE =====")
    print("latent channels:", pipe.vae.config.latent_channels)
    print("scaling factor:", pipe.vae.config.scaling_factor)

    print("\n===== Scheduler =====")
    print("scheduler class:", type(pipe.scheduler).__name__)
    print("num train timesteps:", pipe.scheduler.config.num_train_timesteps)
    print("beta start:", pipe.scheduler.config.beta_start)
    print("beta end:", pipe.scheduler.config.beta_end)
    print("prediction type:", pipe.scheduler.config.prediction_type)


def encode_text(pipe, prompt, negative_prompt, device):
    print("\n===== Prompt Encoding =====")

    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    tokens = pipe.tokenizer.convert_ids_to_tokens(text_inputs.input_ids[0])

    print("input_ids shape:", input_ids.shape)
    print("first 20 tokens:", tokens[:20])

    with torch.no_grad():
        prompt_embeds = pipe.text_encoder(input_ids)[0]

    print("prompt_embeds shape:", prompt_embeds.shape)
    print("prompt_embeds dtype:", prompt_embeds.dtype)

    print("\n===== Negative Prompt Encoding =====")

    negative_inputs = pipe.tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    negative_input_ids = negative_inputs.input_ids.to(device)
    negative_tokens = pipe.tokenizer.convert_ids_to_tokens(negative_inputs.input_ids[0])

    print("negative_input_ids shape:", negative_input_ids.shape)
    print("first 20 negative tokens:", negative_tokens[:20])

    with torch.no_grad():
        negative_embeds = pipe.text_encoder(negative_input_ids)[0]

    print("negative_embeds shape:", negative_embeds.shape)

    combined_embeds = torch.cat([negative_embeds, prompt_embeds])
    print("combined_embeds shape:", combined_embeds.shape)

    return prompt_embeds, negative_embeds, combined_embeds


def inspect_one_denoising_step(pipe, combined_embeds, device, dtype):
    print("\n===== Latents / Scheduler / UNet =====")

    latent_height = HEIGHT // 8
    latent_width = WIDTH // 8
    latent_shape = (
        1,
        pipe.unet.config.in_channels,
        latent_height,
        latent_width,
    )

    generator = make_generator(device, DEFAULT_SEED)
    latents = torch.randn(
        latent_shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    latents = latents * pipe.scheduler.init_noise_sigma

    print("initial latents shape:", latents.shape)
    print("initial latents dtype:", latents.dtype)

    pipe.scheduler.set_timesteps(DEFAULT_STEPS, device=device)
    timesteps = pipe.scheduler.timesteps

    print("timesteps shape:", timesteps.shape)
    print("first 5 timesteps:", timesteps[:5])
    print("last 5 timesteps:", timesteps[-5:])

    timestep = timesteps[0]
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = pipe.scheduler.scale_model_input(
        latent_model_input,
        timestep,
    )

    print("latent_model_input shape:", latent_model_input.shape)
    print("current timestep:", timestep)

    with torch.no_grad():
        noise_pred = pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=combined_embeds,
        ).sample

    print("noise_pred shape:", noise_pred.shape)

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    print("noise_pred_uncond shape:", noise_pred_uncond.shape)
    print("noise_pred_text shape:", noise_pred_text.shape)

    guided_noise_pred = noise_pred_uncond + DEFAULT_GUIDANCE * (
        noise_pred_text - noise_pred_uncond
    )
    print("guided_noise_pred shape:", guided_noise_pred.shape)

    next_latents = pipe.scheduler.step(
        guided_noise_pred,
        timestep,
        latents,
    ).prev_sample

    print("next_latents shape:", next_latents.shape)

    return latents, next_latents


def inspect_vae_decode(pipe, latents):
    print("\n===== VAE Decode =====")

    scaled_latents = latents / pipe.vae.config.scaling_factor
    print("scaled_latents shape:", scaled_latents.shape)

    with torch.no_grad():
        decoded = pipe.vae.decode(scaled_latents, return_dict=False)[0]

    print("decoded tensor shape:", decoded.shape)

    image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    output_path = OUTPUT_DIR / "00_vae_decode_after_one_step.png"
    image.save(output_path)
    print("vae decode preview saved to:", output_path)


def generate_image(
    pipe,
    prompt,
    negative_prompt,
    output_name,
    device,
    seed=DEFAULT_SEED,
    steps=DEFAULT_STEPS,
    guidance=DEFAULT_GUIDANCE,
):
    print(f"\n===== Generate: {output_name} =====")
    print("prompt:", prompt)
    print("negative_prompt:", negative_prompt)
    print("seed:", seed)
    print("steps:", steps)
    print("guidance:", guidance)
    print("scheduler:", type(pipe.scheduler).__name__)

    generator = make_generator(device, seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]

    output_path = OUTPUT_DIR / output_name
    image.save(output_path)
    print("image saved to:", output_path)


def run_parameter_experiments(pipe, device):
    generate_image(
        pipe,
        PROMPT,
        NEGATIVE_PROMPT,
        "01_baseline.png",
        device,
    )

    generate_image(
        pipe,
        "a cute orange cat wearing sunglasses on a wooden table, high quality, detailed",
        NEGATIVE_PROMPT,
        "02_prompt_changed.png",
        device,
    )

    generate_image(
        pipe,
        PROMPT,
        "blurry, low quality, distorted, extra legs, bad anatomy",
        "03_negative_prompt_changed.png",
        device,
    )

    generate_image(
        pipe,
        PROMPT,
        NEGATIVE_PROMPT,
        "04_steps_15.png",
        device,
        steps=15,
    )

    generate_image(
        pipe,
        PROMPT,
        NEGATIVE_PROMPT,
        "05_steps_50.png",
        device,
        steps=50,
    )

    generate_image(
        pipe,
        PROMPT,
        NEGATIVE_PROMPT,
        "06_guidance_4_0.png",
        device,
        guidance=4.0,
    )

    generate_image(
        pipe,
        PROMPT,
        NEGATIVE_PROMPT,
        "07_guidance_11_0.png",
        device,
        guidance=11.0,
    )


def run_scheduler_experiments(pipe, device):
    print("\n===== Scheduler Experiments =====")

    original_scheduler = pipe.scheduler
    scheduler_experiments = {
        "08_scheduler_ddim.png": DDIMScheduler,
        "09_scheduler_euler.png": EulerDiscreteScheduler,
        "10_scheduler_dpm_solver.png": DPMSolverMultistepScheduler,
    }

    for output_name, scheduler_class in scheduler_experiments.items():
        pipe.scheduler = scheduler_class.from_config(original_scheduler.config)
        generate_image(
            pipe,
            PROMPT,
            NEGATIVE_PROMPT,
            output_name,
            device,
        )

    pipe.scheduler = original_scheduler


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device, dtype = get_device_and_dtype()
    print("device:", device)
    print("dtype:", dtype)

    pipe = load_pipeline(device, dtype)
    print("pipeline loaded successfully")

    inspect_pipeline_components(pipe)
    _, _, combined_embeds = encode_text(
        pipe,
        PROMPT,
        NEGATIVE_PROMPT,
        device,
    )

    _, next_latents = inspect_one_denoising_step(
        pipe,
        combined_embeds,
        device,
        dtype,
    )
    inspect_vae_decode(pipe, next_latents)

    run_parameter_experiments(pipe, device)
    run_scheduler_experiments(pipe, device)


if __name__ == "__main__":
    main()
