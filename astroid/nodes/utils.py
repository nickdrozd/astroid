# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable

    from astroid.context import InferenceContext
    from astroid.exceptions import AstroidImportError
    from astroid.nodes import NodeNG
    from astroid.transforms import TransformVisitor


class Position(NamedTuple):
    """Position with line and column information."""

    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int


class InferenceErrorInfo(TypedDict):
    """Store additional Inference error information
    raised with StopIteration exception.
    """

    node: NodeNG
    context: InferenceContext | None
