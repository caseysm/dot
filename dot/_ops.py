"""
Low-level operator access for dot.

This module provides direct access to the underlying C++/CUDA operators
registered with PyTorch's dispatcher.

Usage:
    from dot import _ops

    # Direct operator access
    transport = _ops.sinkhorn_forward(cost, reg, max_iter, lengths)
"""

import os
import glob
import torch

# Extension loading state
_extension_loaded = False


def _load_extension():
    """Load the C++/CUDA extension library."""
    global _extension_loaded

    if _extension_loaded:
        return

    lib_dir = os.path.dirname(__file__)
    lib_pattern = os.path.join(lib_dir, '_C*.so')
    libs = glob.glob(lib_pattern)

    # Also check meson-python editable build location
    if not libs:
        project_root = os.path.dirname(lib_dir)
        editable_pattern = os.path.join(project_root, 'build', '*', '_C*.so')
        libs = glob.glob(editable_pattern)

    if libs:
        torch.ops.load_library(libs[0])
        _extension_loaded = True
    else:
        raise ImportError(
            f"Could not find _C extension library in {lib_dir}. "
            "Run: pip install -e . or meson setup builddir && meson compile -C builddir"
        )


def _ensure_loaded():
    """Ensure extension is loaded before accessing ops."""
    if not _extension_loaded:
        _load_extension()


# Load on module import
_load_extension()

# Expose the full dot namespace for advanced users
dot = torch.ops.dot

# Wrapped operator access
sinkhorn_forward = torch.ops.dot.sinkhorn_forward
sinkhorn_backward = torch.ops.dot.sinkhorn_backward
