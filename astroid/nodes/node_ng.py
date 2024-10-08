# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

from __future__ import annotations

import pprint
from functools import cached_property
from functools import singledispatch as _singledispatch
from typing import TYPE_CHECKING

from astroid.context import InferenceContext
from astroid.exceptions import (
    AstroidError,
    InferenceError,
    ParentMissingError,
    StatementMissing,
    UseInferenceDefault,
)
from astroid.manager import AstroidManager
from astroid.nodes.const import OP_PRECEDENCE
from astroid.util import Uninferable

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator
    from sys import version_info

    from astroid.nodes import (
        Assign,
        ClassDef,
        FunctionDef,
        Lambda,
        LocalsDictNodeNG,
        Module,
    )
    from astroid.nodes._base_nodes import Statement
    from astroid.nodes.utils import Position
    from astroid.typing import InferenceErrorInfo, InferenceResult, InferFn

    if version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self


class NodeNG:
    """A node of the new Abstract Syntax Tree (AST).

    This is the base class for all Astroid node classes.
    """

    is_statement: bool = False
    """Whether this node indicates a statement."""

    # True for For (and for Comprehension if py <3.0)
    optional_assign: bool = False
    """Whether this node optionally assigns a variable.

    This is for loop assignments because loop won't necessarily perform an
    assignment if the loop has no iterations.
    This is also the case from comprehensions in Python 2.
    """
    is_function: bool = False  # True for FunctionDef nodes
    """Whether this node indicates a function."""
    is_lambda: bool = False

    # Attributes below are set by the builder module or by raw factories
    _astroid_fields: tuple[str, ...] = ()
    """Node attributes that contain child nodes.

    This is redefined in most concrete classes.
    """
    _other_fields: tuple[str, ...] = ()
    """Node attributes that do not contain child nodes."""
    _other_other_fields: tuple[str, ...] = ()
    """Attributes that contain AST-dependent fields."""
    # instance specific inference function infer(node, context)
    _explicit_inference: InferFn[Self] | None = None

    def __init__(
        self,
        lineno: int | None,
        col_offset: int | None,
        parent: NodeNG | None,
        *,
        end_lineno: int | None,
        end_col_offset: int | None,
    ) -> None:
        self.lineno = lineno
        """The line that this node appears on in the source code."""

        self.col_offset = col_offset
        """The column that this node appears on in the source code."""

        self.parent = parent
        """The parent node in the syntax tree."""

        self.end_lineno = end_lineno
        """The last line this node appears on in the source code."""

        self.end_col_offset = end_col_offset
        """The end column this node appears on in the source code.

        Note: This is after the last symbol.
        """

        self.position: Position | None = None
        """Position of keyword(s) and name.

        Used as fallback for block nodes which might not provide good
        enough positional information. E.g. ClassDef, FunctionDef.
        """

    def infer(
        self, context: InferenceContext | None = None, **kwargs
    ) -> Generator[InferenceResult]:
        """Get a generator of the inferred values.

        This is the main entry point to the inference system.

        .. seealso:: :ref:`inference`

        If the instance has some explicit inference function set, it will be
        called instead of the default interface.

        :returns: The inferred values.
        :rtype: iterable
        """
        context = (
            InferenceContext()
            if context is None
            else context.extra_context.get(self, context)
        )
        if self._explicit_inference is not None:
            # explicit_inference is not bound, give it self explicitly
            try:
                for result in self._explicit_inference(
                    self,  # type: ignore[arg-type]
                    context,
                    **kwargs,
                ):
                    context.nodes_inferred += 1
                    yield result
                return
            except UseInferenceDefault:
                pass

        key = (self, context.lookupname, context.callcontext, context.boundnode)
        if key in context.inferred:
            yield from context.inferred[key]
            return

        results = []

        # Limit inference amount to help with performance issues with
        # exponentially exploding possible results.
        limit = AstroidManager().max_inferable_values
        for i, result in enumerate(self._infer(context=context, **kwargs)):
            if i >= limit or (context.nodes_inferred > context.max_inferred):
                results.append(Uninferable)
                yield Uninferable
                break
            results.append(result)
            yield result
            context.nodes_inferred += 1

        # Cache generated results for subsequent inferences of the
        # same node using the same context
        context.inferred[key] = tuple(results)
        return

    def repr_name(self) -> str:
        """Get a name for nice representation.

        This is either :attr:`name`, :attr:`attrname`, or the empty string.
        """
        if all(name not in self._astroid_fields for name in ("name", "attrname")):
            return getattr(self, "name", "") or getattr(self, "attrname", "")
        return ""

    def __str__(self) -> str:
        rname = self.repr_name()
        cname = type(self).__name__
        if rname:
            string = "%(cname)s.%(rname)s(%(fields)s)"
            alignment = len(cname) + len(rname) + 2
        else:
            string = "%(cname)s(%(fields)s)"
            alignment = len(cname) + 1
        result = []
        for field in self._other_fields + self._astroid_fields:
            value = getattr(self, field, "Unknown")
            width = 80 - len(field) - alignment
            lines = pprint.pformat(value, indent=2, width=width).splitlines(True)

            inner = [lines[0]]
            for line in lines[1:]:
                inner.append(" " * alignment + line)
            result.append(f"{field}={''.join(inner)}")

        return string % {
            "cname": cname,
            "rname": rname,
            "fields": (",\n" + " " * alignment).join(result),
        }

    def __repr__(self) -> str:
        rname = self.repr_name()
        # The dependencies used to calculate fromlineno (if not cached) may not exist at the time
        try:
            lineno = self.fromlineno
        except AttributeError:
            lineno = 0

        return (
            "<%(cname)s.%(rname)s l.%(lineno)s at 0x%(id)x>"
            if rname
            else "<%(cname)s l.%(lineno)s at 0x%(id)x>"
        ) % {
            "cname": type(self).__name__,
            "rname": rname,
            "lineno": lineno,
            "id": id(self),
        }

    def accept(self, visitor) -> str:
        """Visit this node using the given visitor."""
        return getattr(visitor, f"visit_{self.__class__.__name__.lower()}")(self)

    def get_children(self) -> Iterator[NodeNG]:
        """Get the child nodes below this node."""
        for field in self._astroid_fields:
            if (attr := getattr(self, field)) is None:
                continue
            if isinstance(attr, (list, tuple)):
                yield from attr
            else:
                yield attr
        yield from ()

    def last_child(self) -> NodeNG | None:
        """An optimized version of list(get_children())[-1]."""
        for field in self._astroid_fields[::-1]:
            if not (attr := getattr(self, field)):
                continue
            if isinstance(attr, (list, tuple)):
                return attr[-1]
            return attr
        return None

    def node_ancestors(self) -> Iterator[NodeNG]:
        """Yield parent, grandparent, etc until there are no more."""
        parent = self.parent
        while parent is not None:
            yield parent
            parent = parent.parent

    def parent_of(self, node) -> bool:
        """Check if this node is the parent of the given node.

        :param node: The node to check if it is the child.
        :type node: NodeNG

        :returns: Whether this node is the parent of the given node.
        """
        return any(self is parent for parent in node.node_ancestors())

    def statement(self) -> Statement:
        """The first parent node, including self, marked as statement node.

        :raises StatementMissing: If self has no parent attribute.
        """
        if self.is_statement:
            return self
        if not self.parent:
            raise StatementMissing(target=self)
        return self.parent.statement()

    def frame(self) -> FunctionDef | Module | ClassDef | Lambda:
        """The first parent frame node.

        A frame node is a :class:`Module`, :class:`FunctionDef`,
        :class:`ClassDef` or :class:`Lambda`.

        :returns: The first parent frame node.
        :raises ParentMissingError: If self has no parent attribute.
        """
        if self.parent is None:
            raise ParentMissingError(target=self)
        return self.parent.frame()

    def scope(self) -> LocalsDictNodeNG:
        """The first parent node defining a new scope.

        These can be Module, FunctionDef, ClassDef, Lambda, or GeneratorExp nodes.

        :returns: The first parent scope node.
        """
        if not self.parent:
            raise ParentMissingError(target=self)
        return self.parent.scope()

    def root(self) -> Module:
        """Return the root node of the syntax tree.

        :returns: The root node.
        """
        if not (parent := self.parent):
            # assert isinstance(self, nodes.Module)
            return self

        while parent.parent:
            parent = parent.parent
        # assert isinstance(parent, nodes.Module)
        return parent

    def child_sequence(self, child):
        """Search for the sequence that contains this child.

        :param child: The child node to search sequences for.
        :type child: NodeNG

        :returns: The sequence containing the given child node.
        :rtype: iterable(NodeNG)

        :raises AstroidError: If no sequence could be found that contains
            the given child.
        """
        for field in self._astroid_fields:
            if (node_or_sequence := getattr(self, field)) is child:
                return [node_or_sequence]
            # /!\ compiler.ast Nodes have an __iter__ walking over child nodes
            if (
                isinstance(node_or_sequence, (tuple, list))
                and child in node_or_sequence
            ):
                return node_or_sequence

        msg = "Could not find %s in %s's children"
        raise AstroidError(msg % (repr(child), repr(self)))

    def locate_child(self, child):
        """Find the field of this node that contains the given child.

        :param child: The child node to search fields for.
        :type child: NodeNG

        :returns: A tuple of the name of the field that contains the child,
            and the sequence or node that contains the child node.
        :rtype: tuple(str, iterable(NodeNG) or NodeNG)

        :raises AstroidError: If no field could be found that contains
            the given child.
        """
        for field in self._astroid_fields:
            node_or_sequence = getattr(self, field)
            # /!\ compiler.ast Nodes have an __iter__ walking over child nodes
            if child is node_or_sequence:
                return field, child
            if (
                isinstance(node_or_sequence, (tuple, list))
                and child in node_or_sequence
            ):
                return field, node_or_sequence
        msg = "Could not find %s in %s's children"
        raise AstroidError(msg % (repr(child), repr(self)))

    # FIXME : should we merge child_sequence and locate_child ? locate_child
    # is only used in are_exclusive, child_sequence one time in pylint.

    def next_sibling(self) -> NodeNG | None:
        """The next sibling statement node."""
        return self.parent.next_sibling()

    def previous_sibling(self) -> NodeNG | None:
        """The previous sibling statement."""
        return self.parent.previous_sibling()

    # these are lazy because they're relatively expensive to compute for every
    # single node, and they rarely get looked at

    @cached_property
    def fromlineno(self) -> int:
        """The first line that this node appears on in the source code.

        Can also return 0 if the line can not be determined.
        """
        if self.lineno is None:
            return self._fixed_source_line()
        return self.lineno

    @cached_property
    def tolineno(self) -> int:
        """The last line that this node appears on in the source code.

        Can also return 0 if the line can not be determined.
        """
        return (
            self.end_lineno
            if self.end_lineno is not None
            else (
                self.fromlineno
                if not self._astroid_fields or (last_child := self.last_child()) is None
                else last_child.tolineno
            )
        )

    def _fixed_source_line(self) -> int:
        """Attempt to find the line that this node appears on.

        We need this method since not all nodes have :attr:`lineno` set.
        Will return 0 if the line number can not be determined.
        """
        line = self.lineno
        _node = self
        try:
            while line is None:
                _node = next(_node.get_children())
                line = _node.lineno
        except StopIteration:
            parent = self.parent
            while parent and line is None:
                line = parent.lineno
                parent = parent.parent
        return line or 0

    def block_range(self, lineno: int) -> tuple[int, int]:
        """Get a range from the given line number to where this node ends.

        :param lineno: The line number to start the range at.

        :returns: The range of line numbers that this node belongs to,
            starting at the given line number.
        """
        return lineno, self.tolineno

    def set_local(self, name: str, stmt: NodeNG) -> None:
        """Define that the given name is declared in the given statement node.

        This definition is stored on the parent scope node.

        .. seealso:: :meth:`scope`

        :param name: The name that is being defined.

        :param stmt: The statement that defines the given name.
        """
        assert self.parent
        self.parent.set_local(name, stmt)

    def nodes_of_class(self, klass, skip_klass=None) -> Iterator:
        """Get the nodes (including this one or below) of the given types."""
        if isinstance(self, klass):
            yield self

        if skip_klass is None:
            for child_node in self.get_children():
                yield from child_node.nodes_of_class(klass, skip_klass)

            return

        for child_node in self.get_children():
            if isinstance(child_node, skip_klass):
                continue
            yield from child_node.nodes_of_class(klass, skip_klass)

    @cached_property
    def _assign_nodes_in_scope(self) -> list[Assign]:
        return []

    def _get_name_nodes(self):
        for child_node in self.get_children():
            yield from child_node._get_name_nodes()

    def _get_return_nodes_skip_functions(self):
        yield from ()

    def _get_yield_nodes_skip_functions(self):
        yield from ()

    def _get_yield_nodes_skip_lambdas(self):
        yield from ()

    def _infer_name(self, frame, name):
        # overridden for ImportFrom, Import, Global, Try, TryStar and Arguments
        pass

    def _infer(
        self, context: InferenceContext | None = None, **kwargs
    ) -> Generator[InferenceResult, None, InferenceErrorInfo | None]:
        """We don't know how to resolve a statement by default."""
        # this method is overridden by most concrete classes
        raise InferenceError(
            "No inference function for {node!r}.", node=self, context=context
        )

    def inferred(self):
        """Get a list of the inferred values.

        .. seealso:: :ref:`inference`

        :returns: The inferred values.
        :rtype: list
        """
        return list(self.infer())

    def instantiate_class(self):
        """Instantiate an instance of the defined class.

        .. note::

            On anything other than a :class:`ClassDef` this will return self.

        :returns: An instance of the defined class.
        :rtype: object
        """
        return self

    def has_base(self, node) -> bool:
        """Check if this node inherits from the given type.

        :param node: The node defining the base to look for.
            Usually this is a :class:`Name` node.
        :type node: NodeNG
        """
        return False

    def callable(self) -> bool:
        """Whether this node defines something that is callable.

        :returns: Whether this defines something that is callable.
        """
        return False

    def eq(self, value) -> bool:
        return False

    def as_string(self) -> str:
        """Get the source code that this node represents."""

    def repr_tree(
        self,
        ids=False,
        include_linenos=False,
        ast_state=False,
        indent="   ",
        max_depth=0,
        max_width=80,
    ) -> str:
        """Get a string representation of the AST from this node.

        :param ids: If true, includes the ids with the node type names.
        :type ids: bool

        :param include_linenos: If true, includes the line numbers and
            column offsets.
        :type include_linenos: bool

        :param ast_state: If true, includes information derived from
            the whole AST like local and global variables.
        :type ast_state: bool

        :param indent: A string to use to indent the output string.
        :type indent: str

        :param max_depth: If set to a positive integer, won't return
            nodes deeper than max_depth in the string.
        :type max_depth: int

        :param max_width: Attempt to format the output string to stay
            within this number of characters, but can exceed it under some
            circumstances. Only positive integer values are valid, the default is 80.
        :type max_width: int

        :returns: The string representation of the AST.
        :rtype: str
        """

        # pylint: disable = too-many-statements

        @_singledispatch
        def _repr_tree(node, result, done, cur_indent="", depth=1):
            """Outputs a representation of a non-tuple/list, non-node that's
            contained within an AST, including strings.
            """
            lines = pprint.pformat(
                node, width=max(max_width - len(cur_indent), 1)
            ).splitlines(True)
            result.append(lines[0])
            result.extend([cur_indent + line for line in lines[1:]])
            return len(lines) != 1

        # pylint: disable=unused-variable,useless-suppression; doesn't understand singledispatch
        @_repr_tree.register(tuple)
        @_repr_tree.register(list)
        def _repr_seq(node, result, done, cur_indent="", depth=1):
            """Outputs a representation of a sequence that's contained within an
            AST.
            """
            cur_indent += indent
            result.append("[")
            if not node:
                broken = False
            elif len(node) == 1:
                broken = _repr_tree(node[0], result, done, cur_indent, depth)
            elif len(node) == 2:

                if not (broken := _repr_tree(node[0], result, done, cur_indent, depth)):
                    result.append(", ")
                else:
                    result.append(",\n")
                    result.append(cur_indent)
                broken = _repr_tree(node[1], result, done, cur_indent, depth) or broken
            else:
                result.append("\n")
                result.append(cur_indent)
                for child in node[:-1]:
                    _repr_tree(child, result, done, cur_indent, depth)
                    result.append(",\n")
                    result.append(cur_indent)
                _repr_tree(node[-1], result, done, cur_indent, depth)
                broken = True
            result.append("]")
            return broken

        # pylint: disable=unused-variable,useless-suppression; doesn't understand singledispatch
        @_repr_tree.register(NodeNG)
        def _repr_node(node, result, done, cur_indent="", depth=1):
            """Outputs a strings representation of an astroid node."""
            if node in done:
                result.append(
                    indent + f"<Recursion on {type(node).__name__} with id={id(node)}"
                )
                return False
            done.add(node)

            if max_depth and depth > max_depth:
                result.append("...")
                return False
            depth += 1
            cur_indent += indent
            if ids:
                result.append(f"{type(node).__name__}<0x{id(node):x}>(\n")
            else:
                result.append(f"{type(node).__name__}(")
            fields = []
            if include_linenos:
                fields.extend(("lineno", "col_offset"))
            fields.extend(node._other_fields)
            fields.extend(node._astroid_fields)
            if ast_state:
                fields.extend(node._other_other_fields)
            if not fields:
                broken = False
            elif len(fields) == 1:
                result.append(f"{fields[0]}=")
                broken = _repr_tree(
                    getattr(node, fields[0]), result, done, cur_indent, depth
                )
            else:
                result.append("\n")
                result.append(cur_indent)
                for field in fields[:-1]:
                    # TODO: Remove this after removal of the 'doc' attribute
                    if field == "doc":
                        continue
                    result.append(f"{field}=")
                    _repr_tree(getattr(node, field), result, done, cur_indent, depth)
                    result.append(",\n")
                    result.append(cur_indent)
                result.append(f"{fields[-1]}=")
                _repr_tree(getattr(node, fields[-1]), result, done, cur_indent, depth)
                broken = True
            result.append(")")
            return broken

        result: list[str] = []
        _repr_tree(self, result, set())
        return "".join(result)

    def bool_value(self, context: InferenceContext | None = None):
        """Determine the boolean value of this node.

        The boolean value of a node can have three
        possible values:

            * False: For instance, empty data structures,
              False, empty strings, instances which return
              explicitly False from the __nonzero__ / __bool__
              method.
            * True: Most of constructs are True by default:
              classes, functions, modules etc
            * Uninferable: The inference engine is uncertain of the
              node's value.

        :returns: The boolean value of this node.
        :rtype: bool or Uninferable
        """
        return Uninferable

    def op_precedence(self) -> int:
        # Look up by class name or default to highest precedence
        return OP_PRECEDENCE.get(self.__class__.__name__, len(OP_PRECEDENCE))

    def op_left_associative(self) -> bool:
        # Everything is left associative except `**` and IfExp
        return True

    def precedence_parens(self, child: NodeNG, is_left: bool = True) -> str:
        child_str = child.as_string()

        node_precedence = self.op_precedence()
        child_precedence = child.op_precedence()

        # Wrap child if:
        #  - it has lower precedence
        #  - same precedence with position opposite to
        #    associativity direction
        should_wrap = (
            # 3 * (4 + 5)
            node_precedence > child_precedence
            # 3 - (4 - 5)
            # (2**3)**4
            or (
                node_precedence == child_precedence
                and is_left != self.op_left_associative()
            )
        )

        return f"({child_str})" if should_wrap else child_str
