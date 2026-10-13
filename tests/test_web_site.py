"""The static site's own tests (vitest: number formatting, tracing, and the page rendered from the
exported JSON with every number traced). Skipped when the site's node_modules are not installed."""

import shutil
import subprocess

import pytest

from experiments.run import REPO_ROOT

SITE = REPO_ROOT / "web" / "site"


@pytest.mark.skipif(not (SITE / "node_modules").exists() or shutil.which("npx") is None,
                    reason="run `npm ci` in web/site first")
def test_site_vitest():
    r = subprocess.run(["npx", "vitest", "run"], cwd=SITE, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
