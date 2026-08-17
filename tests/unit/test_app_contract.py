"""Guard the Streamlit app's contract with the library.

The app is not type-checked (streamlit ships no stubs we rely on) and is not exercised by
the rest of the suite, so the realistic way for it to break is a rename in the library
leaving a dangling import. This parses ``app/streamlit_app.py`` and verifies every symbol
it imports from ``gatekeeper`` actually exists -- cheap, and it runs in CI where booting a
Streamlit server would not.

Verified separately by hand: the app boots headless and serves HTTP 200.
"""

from __future__ import annotations

import ast
import importlib

from gatekeeper.data.ingest import project_root

APP_PATH = project_root() / "app" / "streamlit_app.py"


def test_app_file_exists():
    assert APP_PATH.exists(), f"expected the app at {APP_PATH}"


def test_app_parses():
    ast.parse(APP_PATH.read_text(encoding="utf-8"))


def test_every_imported_gatekeeper_symbol_exists():
    """A library rename must not leave the app importing something gone."""
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    missing: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if not node.module.startswith("gatekeeper"):
            continue
        module = importlib.import_module(node.module)
        for alias in node.names:
            if not hasattr(module, alias.name):
                missing.append(f"{node.module}.{alias.name}")

    assert not missing, f"app imports symbols that no longer exist: {missing}"


def test_app_reads_the_committed_spec():
    """The app must not invent its own thresholds (R1.2)."""
    source = APP_PATH.read_text(encoding="utf-8")
    assert "load_spec" in source
    assert "cookie_cats_gate.yaml" in source
    # Anything the spec owns must come from `spec.`, never a literal in the app.
    for owned in ("practical_threshold", "srm_threshold", "alpha"):
        assert f"spec.{owned}" in source, f"app should read {owned} from the spec"


def test_app_falls_back_to_synthetic_data():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "make_cookie_cats_like" in source
    assert "SYNTHETIC DATA" in source


def test_app_shows_the_gate_before_the_metrics():
    """Design.md §4.1 ordering, checked positionally in the source."""
    source = APP_PATH.read_text(encoding="utf-8")
    assert source.index("run_sanity_checks(") < source.index("build_readout(")
