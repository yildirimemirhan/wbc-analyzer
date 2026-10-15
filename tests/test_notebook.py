"""Gate tests for the Kaggle quickstart notebook (no model, no dataset, no network)."""

import ast
import base64
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "wbc_analyzer_kaggle_quickstart.ipynb"
CLASS_NAMES = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location(
        "build_kaggle_notebook", ROOT / "notebooks" / "build_kaggle_notebook.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def code_cells(notebook):
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_notebook_is_valid_nbformat4(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["language"] == "python"
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))


def test_every_code_cell_parses(notebook):
    for index, cell in enumerate(code_cells(notebook)):
        try:
            ast.parse(cell["source"])
        except SyntaxError as exc:
            pytest.fail(f"code cell {index} does not parse: {exc}")


def test_notebook_is_committed_without_outputs(notebook):
    for cell in code_cells(notebook):
        assert cell["outputs"] == []
        assert cell["execution_count"] is None


def test_notebook_matches_current_sources(notebook, builder):
    """Fails when src/ or docs/samples changed and the notebook was not rebuilt."""
    rebuilt = json.dumps(builder.build(), indent=1, ensure_ascii=False) + "\n"
    assert NOTEBOOK.read_text(encoding="utf-8") == rebuilt, (
        "Run: python notebooks/build_kaggle_notebook.py"
    )


def test_embedded_project_sources_are_verbatim(notebook):
    sources = [cell["source"] for cell in code_cells(notebook)]
    for filename in ("custom_layers.py", "custom_losses.py", "preprocessing.py"):
        expected = (ROOT / "src" / filename).read_text(encoding="utf-8").rstrip("\n")
        assert any(expected in source for source in sources), f"{filename} not embedded verbatim"


def test_embedded_samples_are_the_repo_samples(notebook):
    namespace = {}
    cell = next(c for c in code_cells(notebook) if "SAMPLE_IMAGES_B64" in c["source"].split("=")[0])
    exec(cell["source"], namespace)
    embedded = namespace["SAMPLE_IMAGES_B64"]
    assert list(embedded) == CLASS_NAMES
    for name in CLASS_NAMES:
        raw = base64.b64decode(embedded[name])
        assert raw == (ROOT / "docs" / "samples" / f"{name.lower()}.jpg").read_bytes()
        assert raw[:2] == b"\xff\xd8"


def test_class_order_matches_the_app(notebook):
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    app_order = re.search(r"class_names = (\[[^\]]+\])", app_source).group(1)
    assert ast.literal_eval(app_order) == CLASS_NAMES
    setup = next(c for c in code_cells(notebook) if "CLASS_NAMES =" in c["source"])
    assert f"CLASS_NAMES = {json.dumps(CLASS_NAMES)}" in setup["source"]


def test_notebook_contains_no_secrets(notebook):
    blob = NOTEBOOK.read_text(encoding="utf-8")
    for pattern in (r"ghp_[A-Za-z0-9]{20,}", r"github_pat_", r"AIza[0-9A-Za-z_-]{20,}", r"sk-[A-Za-z0-9]{20,}"):
        assert not re.search(pattern, blob)
    assert "GITHUB_TOKEN" not in blob and "GEMINI_API_KEY" not in blob


def test_notebook_has_no_llm_calls(notebook):
    """The quickstart must stay offline: no API clients, no network."""
    joined = "\n".join(cell["source"] for cell in code_cells(notebook))
    for needle in ("openai", "genai", "requests.", "urllib", "http://", "https://"):
        assert needle not in joined
