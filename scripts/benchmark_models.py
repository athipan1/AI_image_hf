import json
import math
import os
import statistics
import time
from pathlib import Path

from huggingface_hub import InferenceClient
from PIL import ImageFilter, ImageStat

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is required for the real inference benchmark")

OUT = Path("benchmark-results")
OUT.mkdir(parents=True, exist_ok=True)

BASELINE_MODEL = "black-forest-labs/FLUX.1-schnell"
CATEGORIES = {
    "photo": {
        "prompt": (
            "professional editorial portrait photo of a Thai man in Bangkok, natural skin texture, "
            "85mm lens, soft window light, realistic fabric, shallow depth of field, high detail"
        ),
        "candidates": [
            "black-forest-labs/FLUX.1-Krea-dev",
            BASELINE_MODEL,
        ],
    },
    "anime": {
        "prompt": (
            "anime swordswoman standing in rainy neon Bangkok at night, clean line art, expressive eyes, "
            "polished cel shading, coherent anatomy, detailed city background"
        ),
        "candidates": [
            "cagliostrolab/animagine-xl-4.0",
            BASELINE_MODEL,
        ],
    },
    "design": {
        "prompt": (
            "minimal geometric rabbit head logo for a modern streetwear brand, black and white, centered, "
            "clean vector-like silhouette, crisp edges, no mockup, no extra objects"
        ),
        "candidates": [
            "Qwen/Qwen-Image",
            BASELINE_MODEL,
        ],
    },
}

client = InferenceClient(provider="auto", api_key=HF_TOKEN)


def visual_signal_score(image):
    """Cheap non-semantic image sanity proxy in [0, 100].

    This is deliberately not called an aesthetic score. It detects blank/flat/low-detail
    outputs by combining luminance contrast and edge energy.
    """
    gray = image.convert("L").resize((256, 256))
    contrast = float(ImageStat.Stat(gray).stddev[0])
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_mean = float(ImageStat.Stat(edges).mean[0])

    contrast_component = min(1.0, contrast / 64.0)
    edge_component = min(1.0, edge_mean / 48.0)
    score = 100.0 * (0.55 * contrast_component + 0.45 * edge_component)
    return round(score, 2)


def safe_name(model):
    return model.replace("/", "__").replace(":", "_")


def generate(category, model, prompt):
    started = time.perf_counter()
    try:
        image = client.text_to_image(
            prompt=prompt,
            model=model,
            width=768,
            height=768,
            seed=20260830,
        )
        latency = round(time.perf_counter() - started, 3)
        path = OUT / f"{category}__{safe_name(model)}.png"
        image.save(path)
        return {
            "category": category,
            "model": model,
            "success": True,
            "latency_seconds": latency,
            "width": image.width,
            "height": image.height,
            "visual_signal_score": visual_signal_score(image),
            "image": str(path),
            "error": None,
        }
    except Exception as exc:  # benchmark must continue across candidates
        latency = round(time.perf_counter() - started, 3)
        return {
            "category": category,
            "model": model,
            "success": False,
            "latency_seconds": latency,
            "width": None,
            "height": None,
            "visual_signal_score": 0.0,
            "image": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def choose_model(results):
    """Promote a specialist only when it is healthy enough versus the baseline.

    Domain suitability comes from the model's intended use. This guard only verifies
    that provider inference works, the output is non-degenerate, and latency is sane.
    """
    successful = [item for item in results if item["success"]]
    if not successful:
        return None, "no candidate succeeded"

    baseline = next((x for x in successful if x["model"] == BASELINE_MODEL), None)
    specialist = next((x for x in successful if x["model"] != BASELINE_MODEL), None)

    if specialist and baseline:
        signal_ok = specialist["visual_signal_score"] >= baseline["visual_signal_score"] * 0.70
        latency_ok = specialist["latency_seconds"] <= max(60.0, baseline["latency_seconds"] * 4.0)
        if signal_ok and latency_ok:
            return specialist["model"], "specialist healthy versus FLUX.1-schnell baseline"
        return baseline["model"], "specialist failed health/latency guard; keep baseline"

    chosen = specialist or baseline or min(successful, key=lambda x: x["latency_seconds"])
    return chosen["model"], "only one healthy candidate available"


def main():
    all_results = []
    recommendations = {}

    for category, config in CATEGORIES.items():
        category_results = []
        for model in config["candidates"]:
            print(f"BENCHMARK_START category={category} model={model}", flush=True)
            result = generate(category, model, config["prompt"])
            category_results.append(result)
            all_results.append(result)
            print(
                "BENCHMARK_RESULT "
                + json.dumps(
                    {
                        "category": category,
                        "model": model,
                        "success": result["success"],
                        "latency_seconds": result["latency_seconds"],
                        "visual_signal_score": result["visual_signal_score"],
                        "error": result["error"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        model, reason = choose_model(category_results)
        recommendations[category] = {"model": model, "reason": reason}
        print(f"RECOMMEND category={category} model={model} reason={reason}", flush=True)

    success_latencies = [r["latency_seconds"] for r in all_results if r["success"]]
    summary = {
        "baseline_model": BASELINE_MODEL,
        "resolution": "768x768",
        "seed": 20260830,
        "quality_metric_note": (
            "visual_signal_score is a non-semantic sanity proxy for contrast/edge energy, not an aesthetic score"
        ),
        "results": all_results,
        "recommendations": recommendations,
        "successful_calls": sum(1 for r in all_results if r["success"]),
        "total_calls": len(all_results),
        "mean_latency_seconds": round(statistics.mean(success_latencies), 3) if success_latencies else None,
        "p95_latency_seconds": (
            round(sorted(success_latencies)[max(0, math.ceil(len(success_latencies) * 0.95) - 1)], 3)
            if success_latencies
            else None
        ),
    }

    (OUT / "benchmark.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Phase 1.5 Model Benchmark",
        "",
        "Real Hugging Face Inference Provider calls at 768x768 with a fixed seed.",
        "",
        "| Category | Model | Success | Latency (s) | Visual signal |",
        "|---|---|---:|---:|---:|",
    ]
    for r in all_results:
        lines.append(
            f"| {r['category']} | `{r['model']}` | {r['success']} | "
            f"{r['latency_seconds']} | {r['visual_signal_score']} |"
        )
    lines.extend(["", "## Recommendations", ""])
    for category, rec in recommendations.items():
        lines.append(f"- **{category}**: `{rec['model']}`. {rec['reason']}")
    lines.extend(
        [
            "",
            "> `visual_signal_score` is only a technical sanity proxy. Final visual quality should also be reviewed manually.",
        ]
    )
    (OUT / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("BENCHMARK_SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)

    if summary["successful_calls"] < 3:
        raise SystemExit("Fewer than 3 model calls succeeded; benchmark is not usable")


if __name__ == "__main__":
    main()
