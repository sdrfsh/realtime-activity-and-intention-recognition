"""Model construction, training and inference.

Keras 3 picks its numerical backend once, when ``keras`` is first imported,
so the default is pinned here (before any submodule imports it). Set the
``KERAS_BACKEND`` environment variable yourself to override.
"""
import os

os.environ.setdefault("KERAS_BACKEND", "jax")
