---
title: Athipan01 AI Image Generator
emoji: 🎨
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: ZeroGPU-first smart image generator
---

# Athipan01 AI Image Generator

GitHub is the source of truth for the Hugging Face Space `Athipan01/AI_image`.

## Current architecture

`Gradio UI -> Prompt Intelligence -> ZeroGPU local SDXL Turbo -> Inference Provider fallback`

The primary backend is now Hugging Face ZeroGPU. The Space loads `stabilityai/sdxl-turbo` locally with Diffusers and runs generation inside `@spaces.GPU(duration=45)`. Hugging Face Inference Providers are used only if the local model fails to load or local inference fails.

ZeroGPU defaults are optimized for interactive generation:

```text
Local model: stabilityai/sdxl-turbo
Resolution: 512x512 by default
Steps: 2 by default, capped at 4 on the local Turbo path
Guidance: 0.0 on the local Turbo path
```

SDXL Turbo is intentionally used as the first local model because it is designed for very low step counts. Higher 768/1024 resolutions remain selectable, but 512x512 is the preferred quality/speed point.

## Prompt Intelligence and provider fallback

Prompt enhancement, style presets, automatic negative prompts and Thai/English intent routing remain active. Intent routing now selects the provider fallback model rather than the primary backend:

```text
General fallback -> black-forest-labs/FLUX.1-schnell
Photo fallback   -> black-forest-labs/FLUX.1-Krea-dev
Anime fallback   -> falanaja/animefal
Design fallback  -> Qwen/Qwen-Image
```

This means normal successful requests consume ZeroGPU quota instead of Inference Provider credits. Provider credits are consumed only when the local ZeroGPU path cannot complete the request.

## Model overrides

```text
LOCAL_IMAGE_MODEL=stabilityai/sdxl-turbo
IMAGE_MODEL=black-forest-labs/FLUX.1-schnell
IMAGE_MODEL_GENERAL=black-forest-labs/FLUX.1-schnell
IMAGE_MODEL_PHOTO=black-forest-labs/FLUX.1-Krea-dev
IMAGE_MODEL_ANIME=falanaja/animefal
IMAGE_MODEL_DESIGN=Qwen/Qwen-Image
```

## Dependencies

Hugging Face manages `gradio`, `spaces`, `huggingface_hub` and the ZeroGPU-compatible PyTorch runtime. The repository installs only application dependencies such as Diffusers, Transformers, Accelerate, Safetensors and Pillow.

## Required secrets

- `HF_TOKEN`: deployment token and provider fallback token.
- Optional `HF_INFERENCE_TOKEN`: narrower runtime token for provider fallback.

The deployment workflow writes the runtime `HF_TOKEN` into the Space secret store. No token value is committed to Git.

## Deployment

Pushes to `main` run tests. A successful test run triggers deployment to `Athipan01/AI_image`.

## Roadmap

Phase 2 will add image-to-image, inpainting, upscale and background removal. Phase 3 will add LoRA and character/style consistency. Later phases will add history, quality evaluation, observability and smarter model promotion.
