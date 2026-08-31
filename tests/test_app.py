import importlib
import os

from PIL import Image

os.environ.setdefault("HF_TOKEN", "test-token")
os.environ.pop("SPACE_ID", None)

app = importlib.import_module("app")


def test_normalise_seed_keeps_explicit_seed():
    assert app._normalise_seed(123) == 123


def test_normalise_seed_generates_random_for_negative():
    value = app._normalise_seed(-1)
    assert isinstance(value, int)
    assert 0 <= value <= 2**32 - 1


def test_enhance_prompt_adds_quality_cues():
    result = app.enhance_prompt("cyberpunk Bangkok", "Cinematic", "Character")
    assert result.startswith("cyberpunk Bangkok")
    assert "cinematic composition" in result
    assert "consistent anatomy" in result
    assert "highly detailed" in result


def test_negative_prompt_merges_defaults_user_and_category_without_duplicates():
    result = app.build_negative_prompt("blurry, glasses", "Auto", "Photo")
    assert result.count("blurry") == 1
    assert "glasses" in result
    assert "watermark" in result
    assert "plastic skin" in result


def test_prompt_intelligence_detects_photo():
    plan = app.analyze_prompt("photorealistic portrait shot on 85mm lens", "Auto")
    assert plan["category"] == "Photo"
    assert plan["route"] == "photo"
    assert plan["recommended_mode"] == "Quality"
    assert plan["width"] == 1024
    assert plan["steps"] == 4
    assert plan["confidence"] >= 0.7


def test_prompt_intelligence_detects_anime():
    plan = app.analyze_prompt("anime swordsman with cel shading", "Auto")
    assert plan["category"] == "Anime"
    assert plan["route"] == "anime"
    assert plan["provider_model"] == app.MODEL_ROUTES["anime"]


def test_prompt_intelligence_detects_design():
    plan = app.analyze_prompt("editorial poster with strong typography and layout", "Auto")
    assert plan["category"] == "Design"
    assert plan["route"] == "design"


def test_prompt_intelligence_detects_product():
    plan = app.analyze_prompt("premium cosmetic bottle product shot on studio background", "Auto")
    assert plan["category"] == "Product"
    assert plan["route"] == "product"
    assert "warped product" in app.build_negative_prompt("", "Auto", plan["category"])


def test_prompt_intelligence_detects_character():
    plan = app.analyze_prompt("full body game character warrior character sheet", "Auto")
    assert plan["category"] == "Character"
    assert plan["route"] == "character"
    assert plan["recommended_mode"] == "Quality"


def test_prompt_intelligence_detects_logo():
    plan = app.analyze_prompt("minimal rabbit logo brand mark", "Auto")
    assert plan["category"] == "Logo"
    assert plan["route"] == "logo"
    assert plan["provider_model"] == app.MODEL_ROUTES["logo"]


def test_prompt_intelligence_uses_style_as_signal():
    plan = app.analyze_prompt("woman in Bangkok", "Photorealistic")
    assert plan["category"] == "Photo"
    assert any(reason.startswith("style:") for reason in plan["reasons"])


def test_prompt_intelligence_falls_back_to_general():
    plan = app.analyze_prompt("blue cloud above a hill", "Auto")
    assert plan["category"] == "General"
    assert plan["route"] == "general"


def test_route_model_remains_backward_compatible():
    route, model = app.route_model("minimal rabbit logo", "Auto")
    assert route == "logo"
    assert model == app.MODEL_ROUTES["logo"]


def test_fast_mode_uses_sdxl_turbo_512():
    profile = app._mode_profile("Fast")
    assert profile["model"] == "stabilityai/sdxl-turbo"
    assert profile["width"] == 512
    assert profile["height"] == 512
    assert profile["steps"] == 2


def test_quality_mode_uses_sdxl_lightning_1024():
    profile = app._mode_profile("Quality")
    assert profile["model"] == "ByteDance/SDXL-Lightning"
    assert profile["width"] == 1024
    assert profile["height"] == 1024
    assert profile["steps"] == 4


def test_auto_mode_uses_ai_recommendation():
    _, negative, plan = app.prepare_intelligent_generation(
        "photorealistic portrait with 85mm lens", "", "Auto", True, "Auto"
    )
    assert plan["actual_mode"] == "Quality"
    assert plan["local_model"] == app.QUALITY_MODEL
    assert plan["width"] == 1024
    assert plan["steps"] == 4
    assert "plastic skin" in negative


def test_manual_fast_overrides_ai_recommendation():
    _, _, plan = app.prepare_intelligent_generation(
        "photorealistic portrait with 85mm lens", "", "Auto", True, "Fast"
    )
    assert plan["category"] == "Photo"
    assert plan["actual_mode"] == "Fast"
    assert plan["local_model"] == app.FAST_MODEL
    assert plan["width"] == 512
    assert plan["height"] == 512
    assert plan["steps"] == 2


def test_two_x_upscale_doubles_resolution():
    source = Image.new("RGB", (32, 24), "white")
    result = app._upscale_image(source, "2x Detail Upscale")
    assert result.size == (64, 48)


def test_no_upscale_keeps_original_object():
    source = Image.new("RGB", (32, 24), "white")
    assert app._upscale_image(source, "None") is source


def test_credit_error_detection():
    assert app._is_credit_error(RuntimeError("402 Payment Required: depleted credits"))
    assert not app._is_credit_error(RuntimeError("temporary provider timeout"))


def test_prepare_generation_enhances_and_routes_fallback():
    prompt, negative, route, model = app.prepare_generation(
        "minimal coffee logo", "no cup", "Product / Logo", True
    )
    assert "clean commercial design" in prompt
    assert "simple memorable geometry" in prompt
    assert "no cup" in negative
    assert route == "logo"
    assert model == app.MODEL_ROUTES["logo"]


def test_prepare_generation_can_disable_enhancement():
    prompt, _, _, _ = app.prepare_generation(
        "simple forest scene", "", "Cinematic", False
    )
    assert prompt == "simple forest scene"


def test_plan_summary_explains_decision():
    _, _, plan = app.prepare_intelligent_generation(
        "anime hero character", "", "Anime", True, "Auto"
    )
    summary = app._plan_summary(plan, "Auto", "None")
    assert "Category:" in summary
    assert "Confidence:" in summary
    assert "AI-selected" in summary
    assert "Why:" in summary


def test_ci_does_not_load_large_local_pipeline():
    assert app.IS_HF_SPACE is False
    assert app._PIPELINES == {}
