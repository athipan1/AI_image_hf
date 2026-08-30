import os
import random
import re
import time

try:
    import spaces
except ImportError:
    class _SpacesFallback:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is not None:
                return func
            return lambda wrapped: wrapped
    spaces = _SpacesFallback()

import gradio as gr
from huggingface_hub import InferenceClient
from PIL import Image, ImageFilter

HF_TOKEN = os.getenv("HF_TOKEN")
IS_HF_SPACE = bool(os.getenv("SPACE_ID"))

FAST_MODEL = os.getenv("LOCAL_FAST_MODEL", "stabilityai/sdxl-turbo")
QUALITY_MODEL = os.getenv("LOCAL_QUALITY_MODEL", "ByteDance/SDXL-Lightning")
DEFAULT_PROVIDER_MODEL = os.getenv("IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
MODEL_ROUTES = {
    "general": os.getenv("IMAGE_MODEL_GENERAL", DEFAULT_PROVIDER_MODEL),
    "photo": os.getenv("IMAGE_MODEL_PHOTO", "black-forest-labs/FLUX.1-Krea-dev"),
    "anime": os.getenv("IMAGE_MODEL_ANIME", "falanaja/animefal"),
    "design": os.getenv("IMAGE_MODEL_DESIGN", "Qwen/Qwen-Image"),
}

STYLE_PRESETS = {
    "Auto": "",
    "Photorealistic": "photorealistic, natural skin texture, realistic materials, professional photography",
    "Cinematic": "cinematic composition, dramatic lighting, volumetric atmosphere, film still, high detail",
    "Anime": "anime illustration, clean line art, expressive character design, polished cel shading",
    "Concept Art": "professional concept art, strong silhouette, atmospheric perspective, production design",
    "Product / Logo": "clean commercial design, centered composition, crisp edges, minimal visual clutter",
}
BASE_NEGATIVE = [
    "low quality", "blurry", "pixelated", "distorted", "deformed", "bad anatomy",
    "extra fingers", "extra limbs", "duplicate subject", "watermark", "unwanted text",
]

provider_client = InferenceClient(provider="auto", api_key=HF_TOKEN) if HF_TOKEN else None
_torch = None
_PIPELINES = {}
_PIPELINE_ERRORS = {}


def _normalise_seed(seed):
    if seed is None or int(seed) < 0:
        return random.randint(0, 2**32 - 1)
    return int(seed)


def _normalise_space(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def _is_credit_error(exc):
    text = str(exc).lower()
    return "402" in text or "payment required" in text or ("depleted" in text and "credits" in text)


def route_model(prompt, style="Auto"):
    text = f"{prompt or ''} {style or ''}".lower()
    if any(x in text for x in ("anime", "manga", "waifu", "cel shading", "อนิเมะ", "มังงะ")):
        route = "anime"
    elif any(x in text for x in ("logo", "brand mark", "icon", "product shot", "packaging", "โลโก้")):
        route = "design"
    elif any(x in text for x in ("photo", "photoreal", "portrait", "camera", "lens", "photography", "ภาพถ่าย", "สมจริง")):
        route = "photo"
    else:
        route = "general"
    return route, MODEL_ROUTES[route]


def enhance_prompt(prompt, style="Auto"):
    prompt = _normalise_space(prompt)
    if not prompt:
        return ""
    additions = []
    preset = STYLE_PRESETS.get(style, "")
    if preset:
        additions.append(preset)
    lower = prompt.lower()
    if not any(x in lower for x in ("composition", "close-up", "wide shot", "portrait", "full body")):
        additions.append("balanced composition, clear focal subject")
    if not any(x in lower for x in ("lighting", "sunlight", "golden hour", "neon", "studio light")):
        additions.append("intentional lighting, dimensional shadows")
    if not any(x in lower for x in ("detailed", "detail", "high quality", "4k", "8k")):
        additions.append("highly detailed, high quality")
    return prompt if not additions else f"{prompt}, " + ", ".join(additions)


def build_negative_prompt(user_negative="", style="Auto"):
    values = list(BASE_NEGATIVE)
    if style == "Product / Logo":
        values += ["busy background", "illegible typography"]
    elif style == "Photorealistic":
        values += ["plastic skin", "oversmoothed skin", "uncanny face"]
    values += [x.strip() for x in (user_negative or "").split(",") if x.strip()]
    out, seen = [], set()
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            out.append(value)
    return ", ".join(out)


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


def _mode_profile(mode):
    if mode == "Quality":
        return {"model": QUALITY_MODEL, "width": 1024, "height": 1024, "steps": 4, "duration": 90}
    return {"model": FAST_MODEL, "width": 512, "height": 512, "steps": 2, "duration": 45}


def _upscale_image(image, upscale):
    if upscale != "2x Detail Upscale":
        return image
    result = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
    return result.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=2))


def preview_prompt(prompt, negative_prompt, style, auto_enhance, mode, upscale):
    final_prompt, final_negative, route, provider_model = prepare_generation(prompt, negative_prompt, style, auto_enhance)
    profile = _mode_profile(mode)
    info = (
        f"Mode: `{mode}` · ZeroGPU primary: `{profile['model']}` · "
        f"Native: `{profile['width']}x{profile['height']}` · Upscale: `{upscale}` · "
        f"Provider fallback: `{route} / {provider_model}`"
    )
    return final_prompt, final_negative, info


def _load_pipeline(mode):
    global _torch
    if mode in _PIPELINES:
        return _PIPELINES[mode]
    if not IS_HF_SPACE:
        raise RuntimeError("Local ZeroGPU pipeline is only loaded inside Hugging Face Space")
    try:
        import torch
        from diffusers import DiffusionPipeline
        _torch = torch
        profile = _mode_profile(mode)
        kwargs = {"torch_dtype": torch.float16}
        if mode == "Fast":
            kwargs["variant"] = "fp16"
        pipe = DiffusionPipeline.from_pretrained(profile["model"], **kwargs).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        _PIPELINES[mode] = pipe
        return pipe
    except Exception as exc:
        _PIPELINE_ERRORS[mode] = f"{type(exc).__name__}: {exc}"
        raise


def _provider_request(model, prompt, negative_prompt, width, height, steps, seed):
    if provider_client is None:
        raise RuntimeError("HF_TOKEN is not configured, so provider fallback is unavailable")
    return provider_client.text_to_image(
        prompt=prompt,
        model=model,
        negative_prompt=negative_prompt or None,
        width=int(width),
        height=int(height),
        num_inference_steps=int(steps),
        seed=seed,
    )


def _provider_fallback(route, selected_model, prompt, negative_prompt, width, height, steps, seed):
    candidates = [selected_model] + ([] if selected_model == DEFAULT_PROVIDER_MODEL else [DEFAULT_PROVIDER_MODEL])
    errors = []
    for model in candidates:
        try:
            return _provider_request(model, prompt, negative_prompt, width, height, steps, seed), model
        except Exception as exc:
            errors.append(f"{model}: {type(exc).__name__}: {exc}")
            if _is_credit_error(exc):
                raise gr.Error("ZeroGPU failed and Inference Provider credits are depleted.") from exc
    raise gr.Error("ZeroGPU and provider fallback failed. " + " | ".join(errors))


@spaces.GPU(duration=90)
def generate_image(prompt, negative_prompt, style, auto_enhance, mode, upscale, seed):
    final_prompt, final_negative, route, provider_model = prepare_generation(prompt, negative_prompt, style, auto_enhance)
    profile = _mode_profile(mode)
    seed = _normalise_seed(seed)
    started = time.perf_counter()
    local_error = None

    try:
        pipe = _load_pipeline(mode)
        generator = _torch.Generator(device="cuda").manual_seed(seed)
        image = pipe(
            prompt=final_prompt,
            width=profile["width"],
            height=profile["height"],
            num_inference_steps=profile["steps"],
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
        image = _upscale_image(image, upscale)
        elapsed = round(time.perf_counter() - started, 2)
        metadata = (
            f"Backend: `ZeroGPU` · Mode: `{mode}` · Model: `{profile['model']}` · "
            f"Output: `{image.width}x{image.height}` · Seed: `{seed}` · Time: `{elapsed}s`"
        )
        return image, metadata, final_prompt, final_negative
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    image, actual_model = _provider_fallback(
        route, provider_model, final_prompt, final_negative,
        profile["width"], profile["height"], profile["steps"], seed,
    )
    image = _upscale_image(image, upscale)
    elapsed = round(time.perf_counter() - started, 2)
    metadata = (
        f"Backend: `Inference Provider fallback` · Mode: `{mode}` · Model: `{actual_model}` · "
        f"Output: `{image.width}x{image.height}` · Seed: `{seed}` · Time: `{elapsed}s` · "
        f"ZeroGPU error: `{local_error}`"
    )
    return image, metadata, final_prompt, final_negative


def reset_form():
    return "", "", "Auto", True, "Fast", "None", -1, None, "Ready.", "", ""


CSS = """
#hero { text-align:center; margin:8px auto 20px; }
#hero h1 { font-size:2.2rem; margin-bottom:6px; }
#generate { min-height:52px; font-size:1.05rem; font-weight:700; }
"""

with gr.Blocks(title="Athipan01 AI Image Generator") as demo:
    gr.Markdown("""
    <div id="hero"><h1>🎨 Athipan01 AI Image Generator</h1>
    <p>Phase 1.7 · Fast / Quality ZeroGPU generation + optional 2x detail upscale</p></div>
    """)
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="เช่น cinematic Thai cyberpunk warrior", lines=5)
            with gr.Row():
                style = gr.Dropdown(list(STYLE_PRESETS.keys()), value="Auto", label="Style preset")
                auto_enhance = gr.Checkbox(value=True, label="✨ Auto Prompt Enhance")
            negative_prompt = gr.Textbox(label="Extra negative prompt", lines=2)
            with gr.Row():
                mode = gr.Radio(["Fast", "Quality"], value="Fast", label="Generation mode")
                upscale = gr.Radio(["None", "2x Detail Upscale"], value="None", label="Post-process")
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            preview_btn = gr.Button("🧠 Preview AI Prompt")
            router_info = gr.Markdown("Fast uses SDXL Turbo. Quality uses SDXL-Lightning.")
            with gr.Row():
                generate_btn = gr.Button("✨ Generate", variant="primary", elem_id="generate")
                clear_btn = gr.Button("Clear")
        with gr.Column():
            output = gr.Image(label="Generated Image", type="pil")
            status = gr.Markdown("Ready.")
            with gr.Accordion("🧠 What the AI sent to the model", open=False):
                enhanced_preview = gr.Textbox(label="Enhanced prompt", lines=5, interactive=False)
                negative_preview = gr.Textbox(label="Final negative prompt", lines=4, interactive=False)

    preview_btn.click(
        preview_prompt,
        inputs=[prompt, negative_prompt, style, auto_enhance, mode, upscale],
        outputs=[enhanced_preview, negative_preview, router_info],
    )
    generate_btn.click(
        generate_image,
        inputs=[prompt, negative_prompt, style, auto_enhance, mode, upscale, seed],
        outputs=[output, status, enhanced_preview, negative_preview],
    )
    clear_btn.click(
        reset_form,
        outputs=[prompt, negative_prompt, style, auto_enhance, mode, upscale, seed, output, status, enhanced_preview, negative_preview],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(css=CSS)
