# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

"""A few useful function/method decorators."""

from __future__ import annotations

import functools
import sys
from typing import TYPE_CHECKING, TypeVar

from astroid.context import InferenceContext
from astroid.exceptions import InferenceError
from astroid.util import Uninferable

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from astroid.typing import InferenceResult

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

_R = TypeVar("_R")
_P = ParamSpec("_P")


def path_wrapper(func):
    """Return the given infer function wrapped to handle the path.

    Used to stop inference if the node has already been looked
    at for a given `InferenceContext` to prevent infinite recursion
    """

    @functools.wraps(func)
    def wrapped(
        node, context: InferenceContext | None = None, _func=func, **kwargs
    ) -> Generator:
        """Wrapper function handling context."""
        if context is None:
            context = InferenceContext()
        if context.push(node):
            return

        yielded = set()

        for res in _func(node, context, **kwargs):
            # unproxy only true instance, not const, tuple, dict...
            ares = res._proxied if res.__class__.__name__ == "Instance" else res
            if ares not in yielded:
                yield res
                yielded.add(ares)

    return wrapped


def yes_if_nothing_inferred(
    func: Callable[_P, Generator[InferenceResult]]
) -> Callable[_P, Generator[InferenceResult]]:
    def inner(*args: _P.args, **kwargs: _P.kwargs) -> Generator[InferenceResult]:
        generator = func(*args, **kwargs)

        try:
            yield next(generator)
        except StopIteration:
            # generator is empty
            yield Uninferable
            return

        yield from generator

    return inner


def raise_if_nothing_inferred(
    func: Callable[_P, Generator[InferenceResult]],
) -> Callable[_P, Generator[InferenceResult]]:
    def inner(*args: _P.args, **kwargs: _P.kwargs) -> Generator[InferenceResult]:
        generator = func(*args, **kwargs)
        try:
            yield next(generator)
        except StopIteration as error:
            # generator is empty
            if error.args:
                raise InferenceError(**error.args[0]) from error
            raise InferenceError(
                "StopIteration raised without any error information."
            ) from error
        except RecursionError as error:
            raise InferenceError(
                f"RecursionError raised with limit {sys.getrecursionlimit()}."
            ) from error

        yield from generator

    return inner
