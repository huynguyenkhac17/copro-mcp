from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_shared_key_state(tmp_path, monkeypatch):
    """Providers built from the real config must not share key state with the MCP
    servers running on this machine (~/.copro/key_state.json)."""
    monkeypatch.setattr(
        "copro.providers.registry.default_shared_state_path",
        lambda: tmp_path / "key_state.json",
    )
