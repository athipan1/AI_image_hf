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

`Gradio UI -> Prompt Intelligence -> Model Router -> Hugging Face InferenceClient -> text-to-image model`

The app does not need to load a large image model inside the Space. Generation is sent through Hugging Face Inference Providers.

## Phase 1: Prompt Intelligence

The first intelligence layer now includes:

- Automatic prompt enhancement for composition, lighting and quality cues.
- Style presets: Auto, Photorealistic, Cinematic, Anime, Concept Art and Product / Logo.
- Automatic negative prompt construction with duplicate removal.
- Intent-based model routing for general, photo, anime and design prompts.
- Prompt preview so users can inspect what is actually sent to the model.
- Thai and English keyword detection for common routing intents.

Model routes are configurable without code changes. If a route variable is not configured, it safely falls back to `IMAGE_MODEL`.

```text
IMAGE_MODEL=black-forest-labs/FLUX.1-schnell
IMAGE_MODEL_GENERAL=black-forest-labs/FLUX.1-schnell
IMAGE_MODEL_PHOTO=black-forest-labs/FLUX.1-schnell
IMAGE_MODEL_ANIME=black-forest-labs/FLUX.1-schnell
IMAGE_MODEL_DESIGN=black-forest-labs/FLUX.1-schnell
```

This means the router logic is active now, while alternative specialist models can be introduced later after provider compatibility tests.

## Required secrets

Configure these in GitHub Actions secrets:

- `HF_TOKEN`: Hugging Face token with write access to the target Space.
- Optional `HF_INFERENCE_TOKEN`: a narrower inference-only token. If omitted, deployment uses `HF_TOKEN` as the Space runtime token.

The deployment workflow securely creates or updates the Space secret named `HF_TOKEN`; no token value is committed to Git.

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

Phase 2 will focus on image editing: image-to-image, inpainting, upscale and background removal. Phase 3 will add LoRA and character/style consistency. Phase 4 will add generation history, favorites, quality scoring, provider fallback and production metrics.
