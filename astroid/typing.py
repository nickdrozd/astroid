# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import (
    TYPE_CHECKING,
    Generic,
    Protocol,
    TypeVar,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from astroid import bases, exceptions, nodes, transforms, util
    from astroid.context import InferenceContext
    from astroid.interpreter._import import spec
    from astroid.nodes import (
        Const,
        Dict,
        EmptyNode,
        List,
        Set,
        Tuple,
    )

    ConstFactoryResult = Const | Dict | EmptyNode | List | Set | Tuple


InferenceResult = (
    type["nodes.NodeNG"] | type["util.UninferableBase"] | type["bases.Proxy"]
)

SuccessfulInferenceResult = type["nodes.NodeNG"] | type["bases.Proxy"]

_SuccessfulInferenceResultT = TypeVar(
    "_SuccessfulInferenceResultT",
    bound=SuccessfulInferenceResult,
)
_SuccessfulInferenceResultT_contra = TypeVar(
    "_SuccessfulInferenceResultT_contra",
    bound=SuccessfulInferenceResult,
    contravariant=True,
)

InferBinaryOp = Callable[
    [
        _SuccessfulInferenceResultT,
        type["nodes.AugAssign"] | type["nodes.BinOp"],
        str,
        InferenceResult,
        type["InferenceContext"],
        SuccessfulInferenceResult,
    ],
    Generator[InferenceResult],
]


class InferFn(Protocol, Generic[_SuccessfulInferenceResultT_contra]):
    def __call__(
        self,
        node: _SuccessfulInferenceResultT_contra,
        context: InferenceContext | None = None,
        **kwargs,
    ) -> Iterator[InferenceResult]: ...  # pragma: no cover


class TransformFn(Protocol, Generic[_SuccessfulInferenceResultT]):
    def __call__(
        self,
        node: _SuccessfulInferenceResultT,
        infer_function: InferFn[_SuccessfulInferenceResultT] = ...,
    ) -> _SuccessfulInferenceResultT | None: ...  # pragma: no cover
