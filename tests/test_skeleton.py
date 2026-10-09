"""Phase 1 gate: every module in the plan's repository layout (§0.4) exists and imports."""

import importlib
import pkgutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Explicit list from §0.4, so a missing file fails here rather than being silently skipped.
MODULES = [
    "data", "data.schema", "data.queries", "data.overlay", "data.dataset",
    "data.augment", "data.build",
    "data.loaders", "data.loaders.geometry3k", "data.loaders.pgdp5k", "data.loaders.pgps9k",
    "data.synthetic", "data.synthetic.generator", "data.synthetic.render",
    "verifier", "verifier.primitives", "verifier.checks", "verifier.tolerance",
    "verifier.audit", "verifier.consistency",
    "models", "models.backbone", "models.head", "models.probe", "models.finetune",
    "experiments", "experiments.run", "experiments.evaluate", "experiments.scaling",
    "experiments.selftrain",
    "analysis", "analysis.metrics", "analysis.headline_table", "analysis.attention_maps",
    "analysis.error_analysis",
]
PACKAGES = ["data", "verifier", "models", "experiments", "analysis"]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_no_unlisted_modules():
    """Any module added to a package must also be added to MODULES above."""
    found = set()
    for pkg in PACKAGES:
        mod = importlib.import_module(pkg)
        found.add(pkg)
        for info in pkgutil.walk_packages(mod.__path__, prefix=f"{pkg}."):
            found.add(info.name)
    assert found == set(MODULES), f"unlisted: {found - set(MODULES)}, missing: {set(MODULES) - found}"


def test_dispatch_targets_exist():
    from experiments.run import DISPATCH, resolve_target
    for mode in DISPATCH:
        assert callable(resolve_target(mode))


@pytest.mark.parametrize("path", ["configs", "reports/figures", "reports/tables", "verifier/gold", "notebooks"])
def test_layout_dirs(path):
    assert (REPO_ROOT / path).is_dir()
