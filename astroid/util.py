# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt


from __future__ import annotations

import contextlib
import sys
from typing import TYPE_CHECKING

from astroid.exceptions import InferenceError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Final

    from astroid import bases, nodes
    from astroid.context import InferenceContext
    from astroid.typing import InferenceResult


class UninferableBase:
    """Special inference object, which is returned when inference fails.

    This is meant to be used as a singleton. Use astroid.util.Uninferable to access it.
    """

    def __repr__(self) -> str:
        return "Uninferable"

    __str__ = __repr__

    def __getattribute__(self, name: str):
        if name == "next":
            raise AttributeError("next method should not be called")
        if name.startswith("__") and name.endswith("__"):
            return object.__getattribute__(self, name)
        if name == "accept":
            return object.__getattribute__(self, name)
        return self

    def __call__(self, *args, **kwargs) -> UninferableBase:
        return self

    def __bool__(self) -> bool:
        return False

    __nonzero__ = __bool__

    def accept(self, visitor):
        return visitor.visit_uninferable(self)


Uninferable: Final = UninferableBase()


class BadOperationMessage:
    """Object which describes a TypeError occurred somewhere in the inference chain.

    This is not an exception, but a container object which holds the types and
    the error which occurred.
    """


class BadUnaryOperationMessage(BadOperationMessage):
    """Object which describes operational failures on UnaryOps."""

    def __init__(self, operand, op, error):
        self.operand = operand
        self.op = op
        self.error = error

    @property
    def _object_type_helper(self):
        from astroid import helpers  # pylint: disable=import-outside-toplevel

        return helpers.object_type

    def _object_type(self, obj):
        objtype = self._object_type_helper(obj)
        if isinstance(objtype, UninferableBase):
            return None

        return objtype

    def __str__(self) -> str:
        return "bad operand type for unary {}: {}".format(
            self.op,
            (
                self.operand.name
                if hasattr(self.operand, "name")
                else (
                    object_type.name
                    if hasattr(object_type := self._object_type(self.operand), "name")
                    else object_type.as_string()
                )
            ),
        )


class BadBinaryOperationMessage(BadOperationMessage):
    """Object which describes type errors for BinOps."""

    def __init__(self, left_type, op, right_type):
        self.left_type = left_type
        self.right_type = right_type
        self.op = op

    def __str__(self) -> str:
        msg = "unsupported operand type(s) for {}: {!r} and {!r}"
        return msg.format(self.op, self.left_type.name, self.right_type.name)


def safe_infer(
    node: nodes.NodeNG | bases.Proxy | UninferableBase,
    context: InferenceContext | None = None,
) -> InferenceResult | None:
    """Return the inferred value for the given node.

    Return None if inference failed or if there is some ambiguity (more than
    one node has been inferred).
    """
    if isinstance(node, UninferableBase):
        return node
    try:
        inferit = node.infer(context=context)
        value = next(inferit)
    except (InferenceError, StopIteration):
        return None
    try:
        next(inferit)
        return None  # None if there is ambiguity on the inferred node
    except InferenceError:
        return None  # there is some kind of ambiguity
    except StopIteration:
        return value


def _augment_sys_path(additional_paths: Sequence[str]) -> list[str]:
    original = list(sys.path)
    changes = []
    seen = set()
    for additional_path in additional_paths:
        if additional_path not in seen:
            changes.append(additional_path)
            seen.add(additional_path)

    sys.path[:] = changes + sys.path
    return original


@contextlib.contextmanager
def augmented_sys_path(additional_paths: Sequence[str]) -> Iterator[None]:
    """Augment 'sys.path' by adding entries from additional_paths."""
    original = _augment_sys_path(additional_paths)
    try:
        yield
    finally:
        sys.path[:] = original
