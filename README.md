---
title: Athipan01 AI Image Generator
emoji: 🎨
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Smart text-to-image with Hugging Face providers
---

# Athipan01 AI Image Generator

GitHub is the source of truth for the Hugging Face Space `Athipan01/AI_image`.

## Current architecture

`Gradio UI -> Prompt Intelligence -> Model Router -> specialist model -> safe fallback -> Hugging Face Inference Providers`

The Space does not load a large diffusion model locally. Generation is routed through Hugging Face Inference Providers.

## Phase 1: Prompt Intelligence

The first intelligence layer includes automatic prompt enhancement, style presets, automatic negative prompts, Thai/English intent routing, and a prompt preview showing what is sent to the selected model.

## Phase 1.5: Specialist models and real benchmark

Default routes now use specialist models:

```text
General -> black-forest-labs/FLUX.1-schnell
Photo   -> black-forest-labs/FLUX.1-Krea-dev
Anime   -> falanaja/animefal
Design  -> Qwen/Qwen-Image
```

All specialist routes fall back to `black-forest-labs/FLUX.1-schnell` when a specialist is unavailable or incompatible. A 402 / depleted-credit response is fail-fast and does not waste another provider call.

The photo route was tested with real Inference Provider calls at 768x768 and a fixed seed. `FLUX.1-Krea-dev` completed in about 3.36 seconds versus about 3.83 seconds for `FLUX.1-schnell`. Visual review showed the Krea result had more natural skin/material rendering, while the Schnell result looked more synthetic. The benchmark's technical `visual_signal_score` is only a contrast/edge sanity proxy and is not treated as an aesthetic score.

The same benchmark attempted Anime and Design, but the Hugging Face account depleted its included Inference Provider credits after the first two successful photo calls. `cagliostrolab/animagine-xl-4.0` was also rejected by automatic provider routing, so the Anime candidate was changed to `falanaja/animefal`, which exposes Hugging Face Inference Provider support through fal. `Qwen/Qwen-Image` remains the Design specialist because Hugging Face lists it as a recommended text-to-image model and it is exposed through Inference Providers.

Run the benchmark again after credits reset or prepaid credits are added:

```text
Actions -> Benchmark Image Models -> Run workflow
```

The workflow stores PNG outputs plus `benchmark.json` and `benchmark.md` as the `phase-1-5-model-benchmark` artifact. A credit-blocked benchmark reports `blocked_by_credits` without turning the workflow red.

## Model overrides

Every route can still be replaced without changing code:

```text
IMAGE_MODEL=black-forest-labs/FLUX.1-schnell
IMAGE_MODEL_GENERAL=black-forest-labs/FLUX.1-schnell
IMAGE_MODEL_PHOTO=black-forest-labs/FLUX.1-Krea-dev
IMAGE_MODEL_ANIME=falanaja/animefal
IMAGE_MODEL_DESIGN=Qwen/Qwen-Image
```

## Required secrets

Configure in GitHub Actions secrets:

- `HF_TOKEN`: Hugging Face token with write access to the target Space and Inference Providers permission.
- Optional `HF_INFERENCE_TOKEN`: a narrower runtime inference token. If omitted, deployment uses `HF_TOKEN` as the Space runtime token.

The deployment workflow creates or updates the Space secret named `HF_TOKEN`; token values are never committed to Git.

## Local development

```bash
python -m venv .venv
pip install -r requirements.txt
export HF_TOKEN=hf_xxx
python app.py
```

## Deployment

Pushes to `main` run tests. A successful test run triggers deployment to `Athipan01/AI_image`, configures the runtime secret, and synchronizes the repository to Hugging Face.

## Next phases

Phase 2 will focus on image editing: image-to-image, inpainting, upscale and background removal. Phase 3 will add LoRA and character/style consistency. Phase 4 will add generation history, favorites, richer quality evaluation, provider observability and production metrics.
