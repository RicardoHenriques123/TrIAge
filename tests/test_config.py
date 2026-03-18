"""Tests for configuration loading."""

import pytest
import os
from pathlib import Path
from unittest.mock import patch

from triage.config import Config, _get_bool, load_config


class TestGetBool:
    """Tests for _get_bool helper function."""

    def test_true_values(self):
        """Test that various true representations are recognized."""
        assert _get_bool("true") is True
        assert _get_bool("True") is True
        assert _get_bool("TRUE") is True
        assert _get_bool("1") is True
        assert _get_bool("yes") is True
        assert _get_bool("Yes") is True

    def test_false_values(self):
        """Test that false values are recognized."""
        assert _get_bool("false") is False
        assert _get_bool("False") is False
        assert _get_bool("0") is False
        assert _get_bool("no") is False
        assert _get_bool("") is False

    def test_default_value(self):
        """Test that default value is used only for None input."""
        # Default is used when value is None
        assert _get_bool(None, default=True) is True
        assert _get_bool(None, default=False) is False
        
        # For unrecognized strings, returns False (not the default)
        assert _get_bool("invalid", default=True) is False
        assert _get_bool("invalid", default=False) is False

    def test_case_insensitive(self):
        """Test that parsing is case-insensitive."""
        assert _get_bool("yEs") is True
        assert _get_bool("nO") is False


class TestConfigDataclass:
    """Tests for Config dataclass."""

    def test_creation(self, config_dict):
        """Test creating a Config instance."""
        config = Config(**config_dict)
        
        assert config.github_token == "ghp_test_token"
        assert config.openrouter_api_key == "sk_test_key"
        assert config.confidence_threshold == 0.80
        assert config.dry_run is False

    def test_frozen_prevents_modification(self, config_dict):
        """Test that Config is immutable."""
        config = Config(**config_dict)
        
        with pytest.raises(AttributeError):
            config.github_token = "new_token"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_from_env(self, config_dict):
        """Test loading configuration from environment variables."""
        with patch.dict(os.environ, {
            "GITHUB_TOKEN": "ghp_env_token",
            "OPENROUTER_API_KEY": "sk_env_key",
            "OPENROUTER_MODEL": "openai/gpt-4o-mini",
            "OPENROUTER_TIMEOUT": "55",
            "CONFIDENCE_THRESHOLD": "0.75",
        }):
            config = load_config()
            
            assert config.github_token == "ghp_env_token"
            assert config.openrouter_api_key == "sk_env_key"
            assert config.openrouter_model == "openai/gpt-4o-mini"
            assert config.openrouter_timeout == 55
            assert config.confidence_threshold == 0.75

    def test_load_config_defaults(self):
        """Test that defaults are used when env vars are not set."""
        with patch.dict(os.environ, {
            "GITHUB_TOKEN": "ghp_test",
            "OPENROUTER_API_KEY": "sk_test",
        }, clear=True):
            config = load_config()
            
            # Check default values
            assert config.openrouter_model == "openai/gpt-4o-mini"
            assert config.openrouter_timeout == 45
            assert config.confidence_threshold == 0.80
            assert config.fallback_label == "needs-triage"

    def test_load_config_dry_run(self):
        """Test DRY_RUN configuration."""
        with patch.dict(os.environ, {
            "GITHUB_TOKEN": "ghp_test",
            "OPENROUTER_API_KEY": "sk_test",
            "DRY_RUN": "true",
        }):
            config = load_config()
            assert config.dry_run is True

        with patch.dict(os.environ, {
            "GITHUB_TOKEN": "ghp_test",
            "OPENROUTER_API_KEY": "sk_test",
            "DRY_RUN": "false",
        }):
            config = load_config()
            assert config.dry_run is False

    def test_load_config_custom_paths(self, temp_dir):
        """Test custom model and log paths."""
        model_path = temp_dir / "custom_model.joblib"
        log_path = temp_dir / "custom_logs.jsonl"
        
        with patch.dict(os.environ, {
            "GITHUB_TOKEN": "ghp_test",
            "OPENROUTER_API_KEY": "sk_test",
            "MODEL_PATH": str(model_path),
            "LOG_PATH": str(log_path),
        }):
            config = load_config()
            
            assert config.model_path == model_path
            assert config.log_path == log_path

    def test_load_config_missing_required_token(self):
        """Test that missing token loads with empty strings."""
        # Config itself doesn't validate - adapters do when instantiated
        with patch.dict(os.environ, {}, clear=True):
            config = load_config()
            # Config loads successfully but with empty strings
            assert config.github_token == ""
            assert config.openrouter_api_key == ""

    def test_load_config_github_api_url(self):
        """Test custom GitHub API URL."""
        with patch.dict(os.environ, {
            "GITHUB_TOKEN": "ghp_test",
            "OPENROUTER_API_KEY": "sk_test",
            "GITHUB_API_URL": "https://github.enterprise.com/api/v3",
        }):
            config = load_config()
            assert config.github_api_url == "https://github.enterprise.com/api/v3"

    def test_load_config_integer_parsing(self):
        """Test that integer environment variables are parsed correctly."""
        with patch.dict(os.environ, {
            "GITHUB_TOKEN": "ghp_test",
            "OPENROUTER_API_KEY": "sk_test",
            "OPENROUTER_TIMEOUT": "120",
        }):
            config = load_config()
            assert config.openrouter_timeout == 120
            assert isinstance(config.openrouter_timeout, int)

    def test_load_config_float_parsing(self):
        """Test that float environment variables are parsed correctly."""
        with patch.dict(os.environ, {
            "GITHUB_TOKEN": "ghp_test",
            "OPENROUTER_API_KEY": "sk_test",
            "CONFIDENCE_THRESHOLD": "0.95",
        }):
            config = load_config()
            assert config.confidence_threshold == 0.95
            assert isinstance(config.confidence_threshold, float)

    def test_load_config_invalid_float(self):
        """Test that invalid float raises error."""
        with patch.dict(os.environ, {
            "GITHUB_TOKEN": "ghp_test",
            "OPENROUTER_API_KEY": "sk_test",
            "CONFIDENCE_THRESHOLD": "not_a_float",
        }):
            with pytest.raises((ValueError, RuntimeError)):
                load_config()

    def test_load_config_invalid_integer(self):
        """Test that invalid integer raises error."""
        with patch.dict(os.environ, {
            "GITHUB_TOKEN": "ghp_test",
            "OPENROUTER_API_KEY": "sk_test",
            "OPENROUTER_TIMEOUT": "not_an_int",
        }):
            with pytest.raises((ValueError, RuntimeError)):
                load_config()
