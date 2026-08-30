import os
import random
import time

import gradio as gr
from huggingface_hub import InferenceClient

MODEL = os.getenv("IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is required. Add it as a Hugging Face Space secret.")

client = InferenceClient(provider="auto", api_key=HF_TOKEN)


def _normalise_seed(seed):
    if seed is None or int(seed) < 0:
        return random.randint(0, 2**32 - 1)
    return int(seed)


def generate_image(prompt, negative_prompt, width, height, steps, guidance, seed):
    prompt = (prompt or "").strip()
    negative_prompt = (negative_prompt or "").strip()

    if len(prompt) < 3:
        raise gr.Error("Prompt must contain at least 3 characters.")
    if len(prompt) > 2000:
        raise gr.Error("Prompt is too long. Maximum length is 2000 characters.")

    seed = _normalise_seed(seed)
    started = time.perf_counter()

    # Providers differ slightly in optional parameter support. Try the richer
    # contract first and fall back to the portable subset if necessary.
    try:
        image = client.text_to_image(
            prompt=prompt,
            model=MODEL,
            negative_prompt=negative_prompt or None,
            width=int(width),
            height=int(height),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            seed=seed,
        )
    except TypeError:
        image = client.text_to_image(
            prompt=prompt,
            model=MODEL,
            negative_prompt=negative_prompt or None,
            seed=seed,
        )
    except Exception as exc:
        raise gr.Error(f"Image generation failed: {exc}") from exc

    elapsed = round(time.perf_counter() - started, 2)
    metadata = f"Model: `{MODEL}` · Seed: `{seed}` · Time: `{elapsed}s`"
    return image, metadata


def reset_form():
    return (
        "",
        "low quality, blurry, distorted, deformed, bad anatomy",
        None,
        "Ready.",
    )


CSS = """
#hero { text-align: center; margin: 8px auto 20px; }
#hero h1 { font-size: 2.2rem; margin-bottom: 6px; }
#hero p { opacity: .75; }
#generate { min-height: 52px; font-size: 1.05rem; font-weight: 700; }
"""

with gr.Blocks(css=CSS, title="Athipan01 AI Image Generator") as demo:
    gr.Markdown(
        """
        <div id="hero">
          <h1>🎨 Athipan01 AI Image Generator</h1>
          <p>Serverless text-to-image generation powered by Hugging Face Inference Providers</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="A futuristic Thai city at sunset, cinematic lighting, highly detailed",
                lines=7,
            )
            negative_prompt = gr.Textbox(
                label="Negative prompt",
                value="low quality, blurry, distorted, deformed, bad anatomy",
                lines=3,
            )

            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    width = gr.Dropdown([512, 768, 1024], value=1024, label="Width")
                    height = gr.Dropdown([512, 768, 1024], value=1024, label="Height")
                with gr.Row():
                    steps = gr.Slider(1, 50, value=4, step=1, label="Steps")
                    guidance = gr.Slider(1, 15, value=3.5, step=0.5, label="Guidance")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")

            with gr.Row():
                generate_btn = gr.Button("✨ Generate Image", variant="primary", elem_id="generate")
                clear_btn = gr.Button("Clear")

        with gr.Column(scale=1):
            output = gr.Image(label="Generated Image", type="pil")
            status = gr.Markdown("Ready.")

    gr.Examples(
        examples=[
            ["A Thai floating market in the future, cinematic, photorealistic, golden hour"],
            ["A cute astronaut exploring a lush alien jungle, concept art, highly detailed"],
            ["A stylish cyberpunk woman walking through Bangkok at night, neon reflections"],
            ["A peaceful Japanese-inspired garden on Mars, cinematic environment art"],
        ],
        inputs=prompt,
        label="Examples",
    )

    generate_btn.click(
        generate_image,
        inputs=[prompt, negative_prompt, width, height, steps, guidance, seed],
        outputs=[output, status],
    )
    clear_btn.click(
        reset_form,
        outputs=[prompt, negative_prompt, output, status],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
