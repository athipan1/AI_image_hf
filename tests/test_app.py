import importlib
import os

os.environ.setdefault("HF_TOKEN", "test-token")

app = importlib.import_module("app")


def test_normalise_seed_keeps_explicit_seed():
    assert app._normalise_seed(123) == 123


def test_normalise_seed_generates_random_for_negative():
    value = app._normalise_seed(-1)
    assert isinstance(value, int)
    assert 0 <= value <= 2**32 - 1


def test_enhance_prompt_adds_quality_cues():
    result = app.enhance_prompt("cyberpunk Bangkok", "Cinematic")
    assert result.startswith("cyberpunk Bangkok")
    assert "cinematic composition" in result
    assert "highly detailed" in result


def test_enhance_prompt_can_preserve_detailed_prompt():
    prompt = "portrait with studio lighting, balanced composition, highly detailed"
    result = app.enhance_prompt(prompt, "Auto")
    assert result == prompt


def test_negative_prompt_merges_defaults_and_user_values_without_duplicates():
    result = app.build_negative_prompt("blurry, glasses", "Auto")
    assert result.count("blurry") == 1
    assert "glasses" in result
    assert "watermark" in result


def test_negative_prompt_adds_photo_specific_quality_terms():
    result = app.build_negative_prompt("", "Photorealistic")
    assert "plastic skin" in result
    assert "uncanny face" in result


def test_router_detects_anime():
    route, model = app.route_model("anime swordsman in Tokyo", "Auto")
    assert route == "anime"
    assert model == app.MODEL_ROUTES["anime"]


def test_router_detects_design():
    route, model = app.route_model("minimal rabbit logo", "Auto")
    assert route == "design"
    assert model == app.MODEL_ROUTES["design"]


def test_router_detects_photo_from_style():
    route, model = app.route_model("woman in Bangkok", "Photorealistic")
    assert route == "photo"
    assert model == app.MODEL_ROUTES["photo"]


def test_prepare_generation_enhances_and_routes():
    prompt, negative, route, model = app.prepare_generation(
        "minimal coffee logo", "no cup", "Product / Logo", True
    )
    assert "clean commercial design" in prompt
    assert "no cup" in negative
    assert route == "design"
    assert model == app.MODEL_ROUTES["design"]


def test_prepare_generation_can_disable_enhancement():
    prompt, _, _, _ = app.prepare_generation(
        "simple forest scene", "", "Cinematic", False
    )
    assert prompt == "simple forest scene"
