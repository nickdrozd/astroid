# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

"""This module contains a set of functions to handle python protocols for nodes
where it makes sense.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from astroid import nodes
from astroid.bases import Instance
from astroid.context import InferenceContext
from astroid.decorators import (
    raise_if_nothing_inferred,
    yes_if_nothing_inferred,
)
from astroid.exceptions import (
    AstroidIndexError,
    AstroidTypeError,
    InferenceError,
)
from astroid.util import Uninferable, UninferableBase, safe_infer

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator, Sequence
    from typing import Any, TypeVar

    from astroid.nodes.node_classes import AssignedStmtsPossibleNode
    from astroid.typing import (
        ConstFactoryResult,
        InferenceResult,
        SuccessfulInferenceResult,
    )

    TupleOrList = nodes.Tuple | nodes.List


@yes_if_nothing_inferred
def tl_infer_binary_op(
    self: nodes.Tuple | nodes.List,
    opnode: nodes.AugAssign | nodes.BinOp,
    operator: str,
    other: InferenceResult,
    context: InferenceContext,
    method: SuccessfulInferenceResult,
) -> Generator[nodes.Tuple | nodes.List | nodes.Const | UninferableBase]:
    """Infer a binary operation on a tuple or list.

    The instance on which the binary operation is performed is a tuple
    or list. This refers to the left-hand side of the operation, so:
    'tuple() + 1' or '[] + A()'
    """
    # pylint: disable=import-outside-toplevel
    from astroid import helpers

    def _multiply_seq_by_int(self, value: int) -> nodes.Tuple | nodes.List:
        nonlocal opnode, context

        node = self.__class__(parent=opnode)

        node.elts = (
            []
            if value <= 0 or not self.elts
            else (
                [Uninferable]
                if len(self.elts) * value > 1e8
                else value
                * list(
                    (
                        safe_infer(elt, context) or Uninferable
                        for elt in self.elts
                        if not isinstance(elt, UninferableBase)
                    )
                )
            )
        )

        return node

    def _filter_uninferable_nodes(elts) -> Iterator[SuccessfulInferenceResult]:
        nonlocal context

        for elt in elts:
            if isinstance(elt, UninferableBase):
                yield nodes.Unknown()
                continue

            for inferred in elt.infer(context):
                yield (
                    nodes.Unknown()
                    if isinstance(inferred, UninferableBase)
                    else inferred
                )

    # For tuples and list the boundnode is no longer the tuple or list instance
    context.boundnode = None
    not_implemented = nodes.Const(NotImplemented)
    if isinstance(other, self.__class__) and operator == "+":
        node = self.__class__(parent=opnode)
        node.elts = list(
            _filter_uninferable_nodes(
                itertools.chain(
                    self.elts,
                    other.elts,
                )
            )
        )
        yield node

    elif isinstance(other, nodes.Const) and operator == "*":
        if not isinstance(other.value, int):
            yield not_implemented
            return

        yield _multiply_seq_by_int(self, other.value)

    elif isinstance(other, Instance) and operator == "*":
        # Verify if the instance supports __index__.
        if not (as_index := helpers.class_instance_as_index(other)):
            yield Uninferable

        elif not isinstance(as_index.value, int):  # pragma: no cover
            # already checked by class_instance_as_index()
            # but faster than casting
            raise AssertionError("Please open a bug report.")

        else:
            yield _multiply_seq_by_int(self, as_index.value)
    else:
        yield not_implemented


# assignment ##################################################################
# pylint: disable-next=pointless-string-statement
"""The assigned_stmts method is responsible to return the assigned statement
(e.g. not inferred) according to the assignment type.

The `assign_path` argument is used to record the lhs path of the original node.
For instance if we want assigned statements for 'c' in 'a, (b,c)', assign_path
will be [1, 1] once arrived to the Assign node.

The `context` argument is the current inference context which should be given
to any intermediary inference necessary.
"""


@raise_if_nothing_inferred
def for_assigned_stmts(
    self: nodes.For | nodes.Comprehension,
    node: AssignedStmtsPossibleNode = None,
    context: InferenceContext | None = None,
    assign_path: list[int] | None = None,
) -> Any:
    if assign_path is None:
        for lst in self.iter.infer(context):
            if isinstance(lst, (nodes.Tuple, nodes.List)):
                yield from lst.elts

        return {
            "node": self,
            "unknown": node,
            "assign_path": assign_path,
            "context": context,
        }

    def _resolve_looppart(parts, assign_path, context):
        """Recursive function to resolve multiple assignments on loops."""
        assign_path = assign_path[:]
        index = assign_path.pop(0)

        for part in parts:
            if isinstance(part, UninferableBase):
                continue

            if not hasattr(part, "itered"):
                continue

            try:
                itered = part.itered()
            except TypeError:
                continue

            try:
                if isinstance(itered[index], (nodes.Const, nodes.Name)):
                    itered = [part]
            except IndexError:
                pass

            for stmt in itered:
                try:
                    assigned = stmt.getitem(nodes.Const(index), context)
                except (AttributeError, AstroidTypeError, AstroidIndexError):
                    continue

                if not assign_path:
                    # we achieved to resolved the assignment path,
                    # don't infer the last part
                    yield assigned
                elif isinstance(assigned, UninferableBase):
                    break
                else:
                    # we are not yet on the last part of the path
                    # search on each possibly inferred value
                    try:
                        yield from _resolve_looppart(
                            assigned.infer(context),
                            assign_path,
                            context,
                        )
                    except InferenceError:
                        break

    yield from _resolve_looppart(
        self.iter.infer(context),
        assign_path,
        context,
    )

    return {
        "node": self,
        "unknown": node,
        "assign_path": assign_path,
        "context": context,
    }


@raise_if_nothing_inferred
def assign_assigned_stmts(
    self: nodes.AugAssign | nodes.Assign | nodes.AnnAssign | nodes.TypeAlias,
    node: AssignedStmtsPossibleNode = None,
    context: InferenceContext | None = None,
    assign_path: list[int] | None = None,
) -> Any:
    if not assign_path:
        yield self.value
        return None

    def _resolve_assignment_parts(parts):
        """Recursive function to resolve multiple assignments."""
        nonlocal assign_path, context

        try:
            parts = parts.infer(context)
        except InferenceError:
            return

        assign_path = assign_path[:]
        index = assign_path.pop(0)

        for part in parts:
            assigned = None

            if isinstance(part, nodes.Dict):
                # A dictionary in an iterating context
                try:
                    assigned, _ = part.items[index]
                except IndexError:
                    return

            elif hasattr(part, "getitem"):
                try:
                    assigned = part.getitem(nodes.Const(index), context)
                except (AstroidTypeError, AstroidIndexError):
                    return

            if not assigned or isinstance(assigned, UninferableBase):
                return

            if not assign_path:
                # last part
                yield assigned
            else:
                yield from _resolve_assignment_parts(assigned)

    yield from _resolve_assignment_parts(self.value)

    return {
        "node": self,
        "unknown": node,
        "assign_path": assign_path,
        "context": context,
    }
