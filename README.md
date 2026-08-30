---
title: Athipan01 AI Image Generator
emoji: 🎨
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Serverless text-to-image generator using Hugging Face Inference Providers
---

# Athipan01 AI Image Generator

GitHub is the source of truth for the Hugging Face Space `Athipan01/AI_image`.

## Architecture

`Gradio UI -> provider abstraction -> Hugging Face InferenceClient -> text-to-image model`

The app does not run a local GPU model. It uses Hugging Face Inference Providers, so the Space can run on CPU hardware.

## Required secrets

Configure these in GitHub Actions secrets:

- `HF_TOKEN`: Hugging Face token with write access to the target Space.

Configure this in the Hugging Face Space secrets:

- `HF_TOKEN`: Hugging Face token with inference permission.

Optional Space variable:

- `IMAGE_MODEL=black-forest-labs/FLUX.1-schnell`

## Local development

```bash
python -m venv .venv
pip install -r requirements.txt
export HF_TOKEN=hf_xxx
python app.py
```

## Deployment

Pushes to `main` run tests and, when `HF_TOKEN` is available in GitHub Actions, sync the application files to `Athipan01/AI_image`.

## Roadmap

- Provider fallback and observability
- Image-to-image
- LoRA adapter support
- Persistent generation history
- Shared rate limiting / quotas
- Moderation and production metrics
