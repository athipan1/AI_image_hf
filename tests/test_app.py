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
