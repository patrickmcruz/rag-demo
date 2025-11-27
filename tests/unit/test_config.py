import os
from pathlib import Path

from src.config import AppConfig, ConfigManager, resolve_path


def test_app_config_defaults():
    cfg = AppConfig.load({})
    assert cfg.data_dir == Path("./data")
    assert cfg.vectorstore_dir == Path("./vectorstore")
    assert cfg.model == "llama3"
    assert cfg.embedding_model == "all-MiniLM-L6-v2"


def test_app_config_env_overrides(monkeypatch):
    env = {
        "DATA_DIR": "/tmp/data",
        "VECTORSTORE_DIR": "/tmp/vs",
        "OLLAMA_MODEL": "mistral",
        "EMBEDDING_MODEL": "bge-small",
        "CHUNK_SIZE": "123",
        "CHUNK_OVERLAP": "7",
        "TOP_K_DOCUMENTS": "9",
        "TEMPERATURE": "0.5",
        "LOG_LEVEL": "DEBUG",
    }
    cfg = AppConfig.load(env)
    assert cfg.data_dir == Path("/tmp/data")
    assert cfg.vectorstore_dir == Path("/tmp/vs")
    assert cfg.model == "mistral"
    assert cfg.embedding_model == "bge-small"
    assert cfg.chunk_size == 123
    assert cfg.chunk_overlap == 7
    assert cfg.top_k_documents == 9
    assert cfg.temperature == 0.5
    assert cfg.log_level == "DEBUG"


def test_config_manager_caches():
    manager = ConfigManager({"DATA_DIR": "/x"})
    first = manager.load()
    second = manager.load()
    assert first is second
    assert first.data_dir == Path("/x")


def test_resolve_path_expands(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    p = resolve_path(Path("~/example"))
    assert p == tmp_path / "example"
