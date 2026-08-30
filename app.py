import os
import random
import re
import time

try:
    import spaces
except ImportError:  # Local/CI fallback. Hugging Face ZeroGPU provides this package.
    class _SpacesFallback:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is not None:
                return func
            return lambda wrapped: wrapped

    spaces = _SpacesFallback()

import gradio as gr
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")
IS_HF_SPACE = bool(os.getenv("SPACE_ID"))

# ZeroGPU primary backend. SDXL Turbo is intentionally lightweight and fast enough
# for interactive ZeroGPU use. It is optimized for 512x512 and 1-4 inference steps.
LOCAL_MODEL = os.getenv("LOCAL_IMAGE_MODEL", "stabilityai/sdxl-turbo")
LOCAL_PIPELINE = None
LOCAL_LOAD_ERROR = None
_torch = None

if IS_HF_SPACE:
    try:
        import torch as _torch_module
        from diffusers import AutoPipelineForText2Image

        _torch = _torch_module
        LOCAL_PIPELINE = AutoPipelineForText2Image.from_pretrained(
            LOCAL_MODEL,
            torch_dtype=_torch.float16,
            variant="fp16",
        ).to("cuda")
        LOCAL_PIPELINE.set_progress_bar_config(disable=True)
    except Exception as exc:  # Keep the UI alive so provider fallback can still work.
        LOCAL_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        LOCAL_PIPELINE = None

# Provider backend is now fallback only.
DEFAULT_PROVIDER_MODEL = os.getenv("IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
MODEL_ROUTES = {
    "general": os.getenv("IMAGE_MODEL_GENERAL", DEFAULT_PROVIDER_MODEL),
    "photo": os.getenv("IMAGE_MODEL_PHOTO", "black-forest-labs/FLUX.1-Krea-dev"),
    "anime": os.getenv("IMAGE_MODEL_ANIME", "falanaja/animefal"),
    "design": os.getenv("IMAGE_MODEL_DESIGN", "Qwen/Qwen-Image"),
}

STYLE_PRESETS = {
    "Auto": "",
    "Photorealistic": (
        "photorealistic, natural skin texture, realistic materials, physically plausible lighting, "
        "high dynamic range, professional photography"
    ),
    "Cinematic": (
        "cinematic composition, dramatic lighting, volumetric atmosphere, film still, "
        "high detail, strong visual storytelling"
    ),
    "Anime": (
        "anime illustration, clean line art, expressive character design, polished cel shading, "
        "detailed background"
    ),
    "Concept Art": (
        "professional concept art, strong silhouette, atmospheric perspective, detailed environment, "
        "production design"
    ),
    "Product / Logo": (
        "clean commercial design, centered composition, controlled lighting, crisp edges, "
        "minimal visual clutter"
    ),
}

BASE_NEGATIVE = [
    "low quality",
    "blurry",
    "pixelated",
    "distorted",
    "deformed",
    "bad anatomy",
    "extra fingers",
    "extra limbs",
    "duplicate subject",
    "watermark",
    "unwanted text",
]

provider_client = InferenceClient(provider="auto", api_key=HF_TOKEN) if HF_TOKEN else None


def _normalise_seed(seed):
    if seed is None or int(seed) < 0:
        return random.randint(0, 2**32 - 1)
    return int(seed)


def _normalise_space(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def _is_credit_error(exc):
    text = str(exc).lower()
    return "402" in text or "payment required" in text or ("depleted" in text and "credits" in text)


def _adapt_prompt_for_model(prompt, model):
    if model == "falanaja/animefal" and "animefal" not in prompt.lower():
        return f"animefal, {prompt}"
    return prompt


def route_model(prompt, style="Auto"):
    """Choose the provider fallback route from the user's intent."""
    text = f"{prompt or ''} {style or ''}".lower()

    anime_terms = ("anime", "manga", "waifu", "cel shading", "อนิเมะ", "มังงะ")
    design_terms = ("logo", "brand mark", "icon", "product shot", "packaging", "โลโก้")
    photo_terms = (
        "photo",
        "photoreal",
        "portrait",
        "camera",
        "lens",
        "photography",
        "ภาพถ่าย",
        "สมจริง",
    )

    if any(term in text for term in anime_terms):
        route = "anime"
    elif any(term in text for term in design_terms):
        route = "design"
    elif any(term in text for term in photo_terms):
        route = "photo"
    else:
        route = "general"

    return route, MODEL_ROUTES[route]


def enhance_prompt(prompt, style="Auto"):
    """Expand a short prompt with composition, lighting and quality cues without changing intent."""
    prompt = _normalise_space(prompt)
    if not prompt:
        return ""

    additions = []
    preset = STYLE_PRESETS.get(style, "")
    if preset:
        additions.append(preset)

    lower = prompt.lower()
    if not any(word in lower for word in ("composition", "close-up", "wide shot", "portrait", "full body")):
        additions.append("balanced composition, clear focal subject")
    if not any(word in lower for word in ("lighting", "sunlight", "golden hour", "neon", "studio light")):
        additions.append("intentional lighting, dimensional shadows")
    if not any(word in lower for word in ("detailed", "detail", "high quality", "4k", "8k")):
        additions.append("highly detailed, high quality")

    if not additions:
        return prompt
    return f"{prompt}, " + ", ".join(additions)


def build_negative_prompt(user_negative="", style="Auto"):
    """Build a negative prompt for provider fallback and future local models that support it."""
    values = list(BASE_NEGATIVE)
    if style == "Product / Logo":
        values.extend(["busy background", "illegible typography"])
    elif style == "Photorealistic":
        values.extend(["plastic skin", "oversmoothed skin", "uncanny face"])

    values.extend(part.strip() for part in (user_negative or "").split(",") if part.strip())

    deduped = []
    seen = set()
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            deduped.append(value)
    return ", ".join(deduped)


def prepare_generation(prompt, user_negative, style, auto_enhance=True):
    prompt = _normalise_space(prompt)
    if len(prompt) < 3:
        raise gr.Error("Prompt must contain at least 3 characters.")
    if len(prompt) > 2000:
        raise gr.Error("Prompt is too long. Maximum length is 2000 characters.")

    final_prompt = enhance_prompt(prompt, style) if auto_enhance else prompt
    final_negative = build_negative_prompt(user_negative, style)
    route, provider_model = route_model(final_prompt, style)
    return final_prompt, final_negative, route, provider_model


def preview_prompt(prompt, negative_prompt, style, auto_enhance):
    final_prompt, final_negative, route, provider_model = prepare_generation(
        prompt, negative_prompt, style, auto_enhance
    )
    local_state = "ready" if LOCAL_PIPELINE is not None else "unavailable"
    info = (
        f"Primary: `ZeroGPU / {LOCAL_MODEL}` ({local_state}) · "
        f"Provider fallback: `{route} / {provider_model}`"
    )
    return final_prompt, final_negative, info


def _provider_request(model, prompt, negative_prompt, width, height, steps, guidance, seed):
    if provider_client is None:
        raise RuntimeError("HF_TOKEN is not configured, so provider fallback is unavailable")

    adapted_prompt = _adapt_prompt_for_model(prompt, model)
    try:
        return provider_client.text_to_image(
            prompt=adapted_prompt,
            model=model,
            negative_prompt=negative_prompt or None,
            width=int(width),
            height=int(height),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            seed=seed,
        )
    except TypeError:
        return provider_client.text_to_image(
            prompt=adapted_prompt,
            model=model,
            negative_prompt=negative_prompt or None,
            seed=seed,
        )


def _generate_provider_fallback(route, selected_model, prompt, negative_prompt, width, height, steps, guidance, seed):
    """Use the specialist provider model, then FLUX schnell as a final provider fallback."""
    candidates = [selected_model]
    if selected_model != DEFAULT_PROVIDER_MODEL:
        candidates.append(DEFAULT_PROVIDER_MODEL)

    attempts = []
    for model in candidates:
        started = time.perf_counter()
        try:
            image = _provider_request(
                model, prompt, negative_prompt, width, height, steps, guidance, seed
            )
            attempts.append({"model": model, "success": True, "seconds": time.perf_counter() - started})
            return image, model, attempts
        except Exception as exc:
            attempts.append(
                {
                    "model": model,
                    "success": False,
                    "seconds": time.perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if _is_credit_error(exc):
                raise gr.Error(
                    "ZeroGPU generation failed and Hugging Face Inference Provider credits are depleted. "
                    "Try again later or inspect the ZeroGPU runtime logs."
                ) from exc

    compact_errors = " | ".join(
        f"{item['model']}: {item.get('error', 'unknown error')}" for item in attempts if not item["success"]
    )
    raise gr.Error(f"ZeroGPU and all provider fallbacks failed. {compact_errors}")


def _local_generation_args(width, height, steps):
    """Normalize settings for SDXL Turbo's fast local path."""
    width = int(width)
    height = int(height)
    steps = max(1, min(int(steps), 4))
    return width, height, steps


@spaces.GPU(duration=45)
def generate_image(
    prompt,
    negative_prompt,
    style,
    auto_enhance,
    width,
    height,
    steps,
    guidance,
    seed,
):
    """Generate an image on ZeroGPU first and use Inference Providers only if local generation fails."""
    final_prompt, final_negative, route, provider_model = prepare_generation(
        prompt, negative_prompt, style, auto_enhance
    )
    seed = _normalise_seed(seed)
    started = time.perf_counter()

    local_error = None
    if LOCAL_PIPELINE is not None and _torch is not None:
        try:
            local_width, local_height, local_steps = _local_generation_args(width, height, steps)
            generator = _torch.Generator(device="cuda").manual_seed(seed)
            image = LOCAL_PIPELINE(
                prompt=final_prompt,
                width=local_width,
                height=local_height,
                num_inference_steps=local_steps,
                guidance_scale=0.0,
                generator=generator,
            ).images[0]
            elapsed = round(time.perf_counter() - started, 2)
            metadata = (
                f"Backend: `ZeroGPU` · Model: `{LOCAL_MODEL}` · Route intent: `{route}` · "
                f"Seed: `{seed}` · Steps: `{local_steps}` · Time: `{elapsed}s`"
            )
            return image, metadata, final_prompt, final_negative
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
    else:
        local_error = LOCAL_LOAD_ERROR or "local pipeline is not initialized"

    # Local ZeroGPU is primary. Only reach this block if it failed to load or infer.
    fallback_started = time.perf_counter()
    image, actual_model, attempts = _generate_provider_fallback(
        route,
        provider_model,
        final_prompt,
        final_negative,
        width,
        height,
        steps,
        guidance,
        seed,
    )
    elapsed = round(time.perf_counter() - started, 2)
    fallback_elapsed = round(time.perf_counter() - fallback_started, 2)
    metadata = (
        f"Backend: `Inference Provider fallback` · Model: `{actual_model}` · Route: `{route}` · "
        f"Seed: `{seed}` · Total: `{elapsed}s` · Provider time: `{fallback_elapsed}s` · "
        f"Provider attempts: `{len(attempts)}` · ZeroGPU error: `{local_error}`"
    )
    return image, metadata, final_prompt, final_negative


def reset_form():
    return "", "", "Auto", True, None, "Ready.", "", ""


CSS = """
#hero { text-align: center; margin: 8px auto 20px; }
#hero h1 { font-size: 2.2rem; margin-bottom: 6px; }
#hero p { opacity: .75; }
#generate { min-height: 52px; font-size: 1.05rem; font-weight: 700; }
.preview-box textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
"""

with gr.Blocks(title="Athipan01 AI Image Generator") as demo:
    gr.Markdown(
        """
        <div id="hero">
          <h1>🎨 Athipan01 AI Image Generator</h1>
          <p>ZeroGPU-first image generation with intelligent provider fallback</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="เช่น ผู้หญิงไซเบอร์พังก์เดินในกรุงเทพตอนกลางคืน",
                lines=6,
            )
            with gr.Row():
                style = gr.Dropdown(
                    choices=list(STYLE_PRESETS.keys()),
                    value="Auto",
                    label="Style preset",
                )
                auto_enhance = gr.Checkbox(value=True, label="✨ Auto Prompt Enhance")

            negative_prompt = gr.Textbox(
                label="Extra negative prompt (provider fallback)",
                placeholder="สิ่งที่ไม่ต้องการเพิ่ม เช่น glasses, crowd",
                lines=2,
            )

            preview_btn = gr.Button("🧠 Preview AI Prompt")
            router_info = gr.Markdown("Primary backend: ZeroGPU")

            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    width = gr.Dropdown([512, 768, 1024], value=512, label="Width")
                    height = gr.Dropdown([512, 768, 1024], value=512, label="Height")
                with gr.Row():
                    steps = gr.Slider(1, 12, value=2, step=1, label="Steps (ZeroGPU uses max 4)")
                    guidance = gr.Slider(0, 15, value=0, step=0.5, label="Guidance (provider fallback)")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")

            with gr.Row():
                generate_btn = gr.Button("⚡ Generate with ZeroGPU", variant="primary", elem_id="generate")
                clear_btn = gr.Button("Clear")

        with gr.Column(scale=1):
            output = gr.Image(label="Generated Image", type="pil")
            status = gr.Markdown("Ready. ZeroGPU is the primary backend.")
            with gr.Accordion("🧠 What the AI sent to the model", open=False):
                enhanced_preview = gr.Textbox(
                    label="Enhanced prompt", lines=5, interactive=False, elem_classes=["preview-box"]
                )
                negative_preview = gr.Textbox(
                    label="Final negative prompt", lines=4, interactive=False, elem_classes=["preview-box"]
                )

    gr.Examples(
        examples=[
            ["A Thai floating market in the future at golden hour"],
            ["A cute astronaut exploring a lush alien jungle"],
            ["ผู้หญิงไซเบอร์พังก์เดินในกรุงเทพตอนกลางคืน"],
            ["minimal geometric rabbit logo for streetwear brand"],
            ["anime swordsman standing in a rainy neon city"],
        ],
        inputs=prompt,
        label="Examples",
    )

    preview_btn.click(
        preview_prompt,
        inputs=[prompt, negative_prompt, style, auto_enhance],
        outputs=[enhanced_preview, negative_preview, router_info],
    )

    generate_btn.click(
        generate_image,
        inputs=[
            prompt,
            negative_prompt,
            style,
            auto_enhance,
            width,
            height,
            steps,
            guidance,
            seed,
        ],
        outputs=[output, status, enhanced_preview, negative_preview],
    )

    clear_btn.click(
        reset_form,
        outputs=[
            prompt,
            negative_prompt,
            style,
            auto_enhance,
            output,
            status,
            enhanced_preview,
            negative_preview,
        ],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(css=CSS)
