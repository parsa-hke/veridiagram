"""web/deploy.py: the Space's files (no network, no credentials)."""

import re

import yaml

from experiments.run import REPO_ROOT
from web import deploy
from web.export import SITE_YAML


def _copies(text: str) -> set[str]:
    """Destinations of COPY lines into the app directory (the code the image carries)."""
    return {m.group(1).rstrip("/") for m in re.finditer(r"^COPY .*? \./(\S+)$", text, re.M)}


def test_space_files(tmp_path):
    site = yaml.safe_load(SITE_YAML.read_text())
    readme, dockerfile = deploy.space_files(tmp_path, "someone/repo", "0123456789abcdef", site)
    front = yaml.safe_load(readme.read_text().split("---")[1])
    assert front["sdk"] == "docker" and front["app_port"] == 7860 and front["title"] == site["title"]
    assert site["author"] in readme.read_text() and "0123456789ab" in readme.read_text()
    d = dockerfile.read_text()
    assert "ARG REPO=someone/repo" in d and "ARG REF=0123456789abcdef" in d
    assert "--mount=type=secret,id=GITHUB_TOKEN,mode=0444,required=true" in d
    assert "rm -rf /src/.git" in d                                  # the token never stays in the image
    assert "GITHUB_TOKEN" not in d.split("FROM python")[1]           # nor reaches the runtime stage
    tested = (REPO_ROOT / "web" / "server" / "Dockerfile").read_text()
    assert _copies(d) == _copies(tested)                              # the same code as the image tested locally
    assert d.split("\n")[-2] == tested.strip().split("\n")[-1]        # same CMD
