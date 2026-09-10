"""Resolves a *model source* string to a local ``.keras`` file.

Two forms are accepted:

* a filesystem path (``models/alexnet_model.keras``), returned unchanged;
* a Hugging Face Hub reference ``hf://namespace/repo[/filename.keras]``,
  downloaded (and cached by ``huggingface_hub``) on first use.

``keras.saving.load_model("hf://...")`` only understands repos published in
Keras' unzipped directory layout (``config.json`` + weights). Repos that hold
a single zipped ``.keras`` archive — like the pretrained door-entry
classifier — need the file fetched explicitly, which is what this does.
"""
from __future__ import annotations

from pathlib import Path

HF_PREFIX = "hf://"


def is_hub_source(source: str | Path) -> bool:
    return str(source).startswith(HF_PREFIX)


def resolve_model_source(source: str | Path, default_filename: str = "alexnet.keras") -> Path:
    """Return a local path for ``source``, downloading from the Hub if needed."""
    text = str(source)
    if not is_hub_source(text):
        return Path(text)

    from huggingface_hub import hf_hub_download

    parts = text[len(HF_PREFIX):].strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(
            f"invalid Hugging Face source {text!r}: expected hf://namespace/repo[/filename]"
        )
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:]) or default_filename
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))
