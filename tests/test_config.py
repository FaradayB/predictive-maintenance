import pytest

from predictivecare import config


def test_model_defaults_present():
    # Defaults must be non-empty so an unset env var never becomes model=None.
    assert config.GOOGLE_MODEL
    assert config.GOOGLE_EMBEDDING


def test_require_key_raises_when_missing(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "")
    with pytest.raises(EnvironmentError):
        config.require_google_api_key()


def test_require_key_returns_when_present(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "dummy-key")
    assert config.require_google_api_key() == "dummy-key"


def test_model_paths_point_into_models_dir():
    assert config.TRACK1_MODEL_PATH.parent == config.MODELS_DIR
    assert config.TRACK2_MODEL_PATH.name == "track2_risk_classifier.pkl"
