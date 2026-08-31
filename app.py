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
    "product": os.getenv("IMAGE_MODEL_PRODUCT", "Qwen/Qwen-Image"),
    "character": os.getenv("IMAGE_MODEL_CHARACTER", DEFAULT_PROVIDER_MODEL),
    "logo": os.getenv("IMAGE_MODEL_LOGO", "Qwen/Qwen-Image"),
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

CATEGORY_RULES = {
    "Photo": (
        "photo", "photoreal", "photorealistic", "portrait", "camera", "lens", "photography",
        "dslr", "bokeh", "ภาพถ่าย", "สมจริง", "พอร์ตเทรต",
    ),
    "Anime": (
        "anime", "manga", "waifu", "cel shading", "cel-shaded", "อนิเมะ", "มังงะ",
    ),
    "Design": (
        "poster", "graphic design", "editorial design", "layout", "brochure", "flyer",
        "typography", "infographic", "โปสเตอร์", "กราฟิก",
    ),
    "Product": (
        "product shot", "product photography", "packaging", "bottle", "cosmetic", "sneaker",
        "commercial product", "studio product", "สินค้า", "แพ็กเกจ", "บรรจุภัณฑ์",
    ),
    "Character": (
        "character", "character sheet", "game character", "hero", "villain", "warrior",
        "full body", "concept character", "ตัวละคร", "คาแรกเตอร์", "นักรบ",
    ),
    "Logo": (
        "logo", "brand mark", "logomark", "emblem", "mascot logo", "icon mark", "โลโก้", "ตราสัญลักษณ์",
    ),
}

CATEGORY_NEGATIVES = {
    "Photo": ["plastic skin", "oversmoothed skin", "uncanny face", "cgi look", "cartoon"],
    "Anime": ["photorealistic skin", "messy line art", "muddy colors", "inconsistent eyes"],
    "Design": ["cluttered layout", "poor hierarchy", "crooked alignment", "illegible typography"],
    "Product": ["warped product", "incorrect label", "busy background", "cropped product", "distorted packaging"],
    "Character": ["bad hands", "asymmetrical eyes", "broken pose", "cropped feet", "duplicate character"],
    "Logo": ["photorealistic", "3d mockup", "busy background", "illegible typography", "complex tiny details"],
}

CATEGORY_CONFIG = {
    "Photo": {"route": "photo", "mode": "Quality", "width": 1024, "height": 1024, "steps": 4},
    "Anime": {"route": "anime", "mode": "Fast", "width": 768, "height": 768, "steps": 3},
    "Design": {"route": "design", "mode": "Fast", "width": 768, "height": 768, "steps": 3},
    "Product": {"route": "product", "mode": "Quality", "width": 1024, "height": 1024, "steps": 4},
    "Character": {"route": "character", "mode": "Quality", "width": 1024, "height": 1024, "steps": 4},
    "Logo": {"route": "logo", "mode": "Quality", "width": 1024, "height": 1024, "steps": 4},
    "General": {"route": "general", "mode": "Fast", "width": 512, "height": 512, "steps": 2},
}

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


def analyze_prompt(prompt, style="Auto"):
    text = f"{prompt or ''} {style or ''}".lower()
    scores = {category: 0 for category in CATEGORY_RULES}
    evidence = {category: [] for category in CATEGORY_RULES}

    for category, keywords in CATEGORY_RULES.items():
        for keyword in keywords:
            if keyword in text:
                scores[category] += 2 if " " in keyword else 1
                evidence[category].append(keyword)

    style_boosts = {
        "Photorealistic": "Photo",
        "Anime": "Anime",
        "Concept Art": "Character",
        "Product / Logo": "Logo",
    }
    boosted = style_boosts.get(style)
    if boosted:
        scores[boosted] += 3
        evidence[boosted].append(f"style:{style}")

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_category, top_score = ranked[0]
    second_score = ranked[1][1]
    if top_score <= 0:
        category = "General"
        confidence = 0.45
        reasons = ["no strong specialist signal"]
    else:
        category = top_category
        margin = top_score - second_score
        confidence = min(0.98, 0.58 + (top_score * 0.06) + (margin * 0.05))
        reasons = evidence[top_category][:4] or ["semantic keyword match"]

    config = CATEGORY_CONFIG[category]
    route = config["route"]
    return {
        "category": category,
        "confidence": round(confidence, 2),
        "reasons": reasons,
        "route": route,
        "provider_model": MODEL_ROUTES[route],
        "recommended_mode": config["mode"],
        "width": config["width"],
        "height": config["height"],
        "steps": config["steps"],
    }


def route_model(prompt, style="Auto"):
    plan = analyze_prompt(prompt, style)
    return plan["route"], plan["provider_model"]


def enhance_prompt(prompt, style="Auto", category=None):
    prompt = _normalise_space(prompt)
    if not prompt:
        return ""
    additions = []
    preset = STYLE_PRESETS.get(style, "")
    if preset:
        additions.append(preset)
    category_cues = {
        "Photo": "natural perspective, realistic texture, physically plausible lighting",
        "Anime": "clean silhouette, coherent line art, expressive pose",
        "Design": "clear visual hierarchy, deliberate spacing, professional layout",
        "Product": "commercial studio lighting, accurate product geometry, premium advertising composition",
        "Character": "consistent anatomy, readable silhouette, character-focused composition",
        "Logo": "simple memorable geometry, vector-like edges, scalable brand mark",
    }
    if category in category_cues:
        additions.append(category_cues[category])
    lower = prompt.lower()
    if not any(x in lower for x in ("composition", "close-up", "wide shot", "portrait", "full body")):
        additions.append("balanced composition, clear focal subject")
    if not any(x in lower for x in ("lighting", "sunlight", "golden hour", "neon", "studio light")):
        additions.append("intentional lighting, dimensional shadows")
    if not any(x in lower for x in ("detailed", "detail", "high quality", "4k", "8k")):
        additions.append("highly detailed, high quality")
    return prompt if not additions else f"{prompt}, " + ", ".join(additions)


def build_negative_prompt(user_negative="", style="Auto", category=None):
    values = list(BASE_NEGATIVE)
    if style == "Product / Logo":
        values += ["busy background", "illegible typography"]
    elif style == "Photorealistic":
        values += ["plastic skin", "oversmoothed skin", "uncanny face"]
    if category:
        values += CATEGORY_NEGATIVES.get(category, [])
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
    plan = analyze_prompt(prompt, style)
    final_prompt = enhance_prompt(prompt, style, plan["category"]) if auto_enhance else prompt
    final_negative = build_negative_prompt(user_negative, style, plan["category"])
    return final_prompt, final_negative, plan["route"], plan["provider_model"]


def prepare_intelligent_generation(prompt, user_negative, style, auto_enhance, requested_mode="Auto"):
    prompt = _normalise_space(prompt)
    if len(prompt) < 3:
        raise gr.Error("Prompt must contain at least 3 characters.")
    if len(prompt) > 2000:
        raise gr.Error("Prompt is too long. Maximum length is 2000 characters.")
    plan = analyze_prompt(prompt, style)
    final_prompt = enhance_prompt(prompt, style, plan["category"]) if auto_enhance else prompt
    final_negative = build_negative_prompt(user_negative, style, plan["category"])
    actual_mode = plan["recommended_mode"] if requested_mode == "Auto" else requested_mode
    profile = _mode_profile(actual_mode)
    plan = dict(plan)
    plan["actual_mode"] = actual_mode
    plan["local_model"] = profile["model"]
    plan["width"] = profile["width"] if requested_mode != "Auto" else plan["width"]
    plan["height"] = profile["height"] if requested_mode != "Auto" else plan["height"]
    plan["steps"] = profile["steps"] if requested_mode != "Auto" else plan["steps"]
    return final_prompt, final_negative, plan


def _mode_profile(mode):
    if mode == "Quality":
        return {"model": QUALITY_MODEL, "width": 1024, "height": 1024, "steps": 4, "duration": 90}
    return {"model": FAST_MODEL, "width": 512, "height": 512, "steps": 2, "duration": 45}


def _upscale_image(image, upscale):
    if upscale != "2x Detail Upscale":
        return image
    result = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
    return result.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=2))


def _plan_summary(plan, requested_mode, upscale):
    reasons = ", ".join(plan["reasons"])
    return (
        f"Category: `{plan['category']}` · Confidence: `{plan['confidence']:.0%}` · "
        f"Mode: `{plan['actual_mode']}` ({'AI-selected' if requested_mode == 'Auto' else 'manual override'}) · "
        f"ZeroGPU: `{plan['local_model']}` · Resolution: `{plan['width']}x{plan['height']}` · "
        f"Steps: `{plan['steps']}` · Provider fallback: `{plan['route']} / {plan['provider_model']}` · "
        f"Why: `{reasons}` · Upscale: `{upscale}`"
    )


def preview_prompt(prompt, negative_prompt, style, auto_enhance, mode, upscale):
    final_prompt, final_negative, plan = prepare_intelligent_generation(
        prompt, negative_prompt, style, auto_enhance, mode
    )
    return final_prompt, final_negative, _plan_summary(plan, mode, upscale)


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
    final_prompt, final_negative, plan = prepare_intelligent_generation(
        prompt, negative_prompt, style, auto_enhance, mode
    )
    seed = _normalise_seed(seed)
    started = time.perf_counter()
    local_error = None

    try:
        pipe = _load_pipeline(plan["actual_mode"])
        generator = _torch.Generator(device="cuda").manual_seed(seed)
        image = pipe(
            prompt=final_prompt,
            width=plan["width"],
            height=plan["height"],
            num_inference_steps=plan["steps"],
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
        image = _upscale_image(image, upscale)
        elapsed = round(time.perf_counter() - started, 2)
        metadata = (
            f"Backend: `ZeroGPU` · Category: `{plan['category']}` · Confidence: `{plan['confidence']:.0%}` · "
            f"Mode: `{plan['actual_mode']}` · Model: `{plan['local_model']}` · "
            f"Output: `{image.width}x{image.height}` · Steps: `{plan['steps']}` · Seed: `{seed}` · Time: `{elapsed}s`"
        )
        return image, metadata, final_prompt, final_negative
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    image, actual_model = _provider_fallback(
        plan["route"], plan["provider_model"], final_prompt, final_negative,
        plan["width"], plan["height"], plan["steps"], seed,
    )
    image = _upscale_image(image, upscale)
    elapsed = round(time.perf_counter() - started, 2)
    metadata = (
        f"Backend: `Inference Provider fallback` · Category: `{plan['category']}` · Confidence: `{plan['confidence']:.0%}` · "
        f"Mode: `{plan['actual_mode']}` · Model: `{actual_model}` · "
        f"Output: `{image.width}x{image.height}` · Steps: `{plan['steps']}` · Seed: `{seed}` · Time: `{elapsed}s` · "
        f"ZeroGPU error: `{local_error}`"
    )
    return image, metadata, final_prompt, final_negative


def reset_form():
    return "", "", "Auto", True, "Auto", "None", -1, None, "Ready.", "", ""


CSS = """
#hero { text-align:center; margin:8px auto 20px; }
#hero h1 { font-size:2.2rem; margin-bottom:6px; }
#generate { min-height:52px; font-size:1.05rem; font-weight:700; }
"""

with gr.Blocks(title="Athipan01 AI Image Generator") as demo:
    gr.Markdown("""
    <div id="hero"><h1>🎨 Athipan01 AI Image Generator</h1>
    <p>Prompt Intelligence · automatic category, model route, negative prompt, resolution and steps</p></div>
    """)
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="เช่น cinematic Thai cyberpunk warrior", lines=5)
            with gr.Row():
                style = gr.Dropdown(list(STYLE_PRESETS.keys()), value="Auto", label="Style preset")
                auto_enhance = gr.Checkbox(value=True, label="✨ Auto Prompt Enhance")
            negative_prompt = gr.Textbox(label="Extra negative prompt", lines=2)
            with gr.Row():
                mode = gr.Radio(["Auto", "Fast", "Quality"], value="Auto", label="Generation mode")
                upscale = gr.Radio(["None", "2x Detail Upscale"], value="None", label="Post-process")
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            preview_btn = gr.Button("🧠 Analyze Prompt")
            router_info = gr.Markdown("Auto mode lets Prompt Intelligence choose the generation plan.")
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
