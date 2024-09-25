# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

"""Astroid hooks for numpy.core.numeric module."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from astroid.brain.brain_numpy_utils import (
    attribute_name_looks_like_numpy_member,
    infer_numpy_attribute,
)
from astroid.brain.helpers import register_module_extender
from astroid.builder import parse
from astroid.inference_tip import inference_tip
from astroid.nodes.node_classes import Attribute

if TYPE_CHECKING:
    from astroid import nodes
    from astroid.manager import AstroidManager


def numpy_core_numeric_transform() -> nodes.Module:
    return parse(
        """
    # different functions defined in numeric.py
    import numpy
    def zeros_like(a, dtype=None, order='K', subok=True, shape=None): return numpy.ndarray((0, 0))
    def ones_like(a, dtype=None, order='K', subok=True, shape=None): return numpy.ndarray((0, 0))
    def full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None): return numpy.ndarray((0, 0))
        """
    )


METHODS_TO_BE_INFERRED = {
    "ones": """def ones(shape, dtype=None, order='C'):
            return numpy.ndarray([0, 0])"""
}


def register(manager: AstroidManager) -> None:
    register_module_extender(
        manager, "numpy.core.numeric", numpy_core_numeric_transform
    )

    manager.register_transform(
        Attribute,
        inference_tip(functools.partial(infer_numpy_attribute, METHODS_TO_BE_INFERRED)),
        functools.partial(
            attribute_name_looks_like_numpy_member,
            frozenset(METHODS_TO_BE_INFERRED.keys()),
        ),
    )
