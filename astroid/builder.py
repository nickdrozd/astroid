# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

"""The AstroidBuilder makes astroid from living object and / or from _ast.

The builder is not thread safe and can't be used to parse different sources
at the same time.
"""

from __future__ import annotations

import ast
import io
import os
import re
import sys
import textwrap
import token
import warnings
from tokenize import detect_encoding, generate_tokens
from typing import TYPE_CHECKING

from astroid import bases, modutils, nodes, raw_building, util
from astroid._ast import Context, get_parser_module, parse_function_type_comment
from astroid.const import PY312_PLUS
from astroid.exceptions import AstroidBuildingError, AstroidSyntaxError, InferenceError
from astroid.manager import AstroidManager
from astroid.nodes.node_classes import AssignName
from astroid.nodes.utils import Position

if TYPE_CHECKING:
    import types
    from collections.abc import Callable, Generator, Iterator, Sequence
    from io import TextIOWrapper
    from tokenize import TokenInfo
    from typing import Final

    from astroid._ast import ParserModule
    from astroid.nodes import NodeNG


# The name of the transient function that is used to
# wrap expressions to be extracted when calling
# extract_node.
_TRANSIENT_FUNCTION = "__"

# The comment used to select a statement to be extracted
# when calling extract_node.
_STATEMENT_SELECTOR = "#@"

if PY312_PLUS:
    warnings.filterwarnings("ignore", "invalid escape sequence", SyntaxWarning)


def open_source_file(filename: str) -> tuple[TextIOWrapper, str, str]:
    # pylint: disable=consider-using-with
    with open(filename, "rb") as byte_stream:
        encoding = detect_encoding(byte_stream.readline)[0]
    stream = open(filename, newline=None, encoding=encoding)
    data = stream.read()
    return stream, encoding, data


def _can_assign_attr(node: nodes.ClassDef, attrname: str | None) -> bool:
    try:
        slots = node.slots()
    except NotImplementedError:
        pass
    else:
        if slots and attrname not in {slot.value for slot in slots}:
            return False
    return node.qname() != "builtins.object"


class AstroidBuilder(raw_building.InspectBuilder):
    """Class for building an astroid tree from source code or from a live module.

    The param *manager* specifies the manager class which should be used.
    If no manager is given, then the default one will be used. The
    param *apply_transforms* determines if the transforms should be
    applied after the tree was built from source or from a live object,
    by default being True.
    """

    def __init__(
        self, manager: AstroidManager | None = None, apply_transforms: bool = True
    ) -> None:
        super().__init__(manager)
        self._apply_transforms = apply_transforms
        if not raw_building.InspectBuilder.bootstrapped:
            raw_building._astroid_bootstrapping()

    def module_build(
        self, module: types.ModuleType, modname: str | None = None
    ) -> nodes.Module:
        """Build an astroid from a living module instance."""
        node = None
        path = getattr(module, "__file__", None)
        # Prefer the loader to get the source rather than assuming we have a
        # filesystem to read the source file from ourselves.
        if loader := getattr(module, "__loader__", None):
            modname = modname or module.__name__
            if source := loader.get_source(modname):
                node = self.string_build(source, modname, path=path)
        if node is None and path is not None:
            path_, ext = os.path.splitext(modutils._path_from_filename(path))
            if ext in {".py", ".pyc", ".pyo"} and os.path.exists(path_ + ".py"):
                node = self.file_build(path_ + ".py", modname)
        if node is None:
            # this is a built-in module
            # get a partial representation by introspection
            node = self.inspect_build(module, modname=modname, path=path)
            if self._apply_transforms:
                # We have to handle transformation by ourselves since the
                # rebuilder isn't called for builtin nodes
                node = self._manager.visit_transforms(node)
        assert isinstance(node, nodes.Module)
        return node

    def file_build(self, path: str, modname: str | None = None) -> nodes.Module:
        """Build astroid from a source code file (i.e. from an ast).

        *path* is expected to be a python source file
        """
        try:
            stream, encoding, data = open_source_file(path)
        except OSError as exc:
            raise AstroidBuildingError(
                "Unable to load file {path}:\n{error}",
                modname=modname,
                path=path,
                error=exc,
            ) from exc
        except (SyntaxError, LookupError) as exc:
            raise AstroidSyntaxError(
                "Python 3 encoding specification error or unknown encoding:\n"
                "{error}",
                modname=modname,
                path=path,
                error=exc,
            ) from exc
        except UnicodeError as exc:  # wrong encoding
            # detect_encoding returns utf-8 if no encoding specified
            raise AstroidBuildingError(
                "Wrong or no encoding specified for {filename}.", filename=path
            ) from exc
        with stream:
            # get module name if necessary
            if modname is None:
                try:
                    modname = ".".join(modutils.modpath_from_file(path))
                except ImportError:
                    modname = os.path.splitext(os.path.basename(path))[0]
            # build astroid representation
            module, builder = self._data_build(data, modname, path)
            return self._post_build(module, builder, encoding)

    def string_build(
        self, data: str, modname: str = "", path: str | None = None
    ) -> nodes.Module:
        """Build astroid from source code string."""
        module, builder = self._data_build(data, modname, path)
        module.file_bytes = data.encode("utf-8")
        return self._post_build(module, builder, "utf-8")

    def _post_build(
        self, module: nodes.Module, builder: TreeRebuilder, encoding: str
    ) -> nodes.Module:
        """Handles encoding and delayed nodes after a module has been built."""
        module.file_encoding = encoding
        self._manager.cache_module(module)
        # post tree building steps after we stored the module in the cache:
        for from_node in builder._import_from_nodes:
            if from_node.modname == "__future__":
                for symbol, _ in from_node.names:
                    module.future_imports.add(symbol)
            self.add_from_names_to_locals(from_node)
        # handle delayed assattr nodes
        for delayed in builder._delayed_assattr:
            self.delayed_assattr(delayed)

        # Visit the transforms
        if self._apply_transforms:
            module = self._manager.visit_transforms(module)
        return module

    def _data_build(
        self, data: str, modname: str, path: str | None
    ) -> tuple[nodes.Module, TreeRebuilder]:
        """Build tree node from data and add some informations."""
        try:
            node, parser_module = _parse_string(
                data, type_comments=True, modname=modname
            )
        except (TypeError, ValueError, SyntaxError) as exc:
            raise AstroidSyntaxError(
                "Parsing Python code failed:\n{error}",
                source=data,
                modname=modname,
                path=path,
                error=exc,
            ) from exc

        node_file = os.path.abspath(path) if path is not None else "<?>"

        if modname.endswith(".__init__"):
            modname = modname[:-9]
            package = True
        else:
            package = (
                path is not None
                and os.path.splitext(os.path.basename(path))[0] == "__init__"
            )
        builder = TreeRebuilder(self._manager, parser_module, data)
        module = builder.visit_module(node, modname, node_file, package)
        return module, builder

    def add_from_names_to_locals(self, node: nodes.ImportFrom) -> None:
        """Store imported names to the locals.

        Resort the locals if coming from a delayed node
        """

        def _key_func(node: nodes.NodeNG) -> int:
            return node.fromlineno or 0

        def sort_locals(my_list: list[nodes.NodeNG]) -> None:
            my_list.sort(key=_key_func)

        assert node.parent  # It should always default to the module
        for name, asname in node.names:
            if name == "*":
                try:
                    imported = node.do_import_module()
                except AstroidBuildingError:
                    continue
                for name in imported.public_names():
                    node.parent.set_local(name, node)
                    sort_locals(node.parent.scope().locals[name])  # type: ignore[arg-type]
            else:
                node.parent.set_local(asname or name, node)
                sort_locals(node.parent.scope().locals[asname or name])  # type: ignore[arg-type]

    def delayed_assattr(self, node: nodes.AssignAttr) -> None:
        """Visit an AssignAttr node.

        This adds name to locals and handle members definition.
        """
        from astroid import objects  # pylint: disable=import-outside-toplevel

        try:
            for inferred in node.expr.infer():
                if isinstance(inferred, util.UninferableBase):
                    continue
                try:
                    # We want a narrow check on the parent type, not all of its subclasses
                    if type(inferred) in {bases.Instance, objects.ExceptionInstance}:
                        inferred = inferred._proxied
                        iattrs = inferred.instance_attrs
                        if not _can_assign_attr(inferred, node.attrname):
                            continue
                    elif isinstance(inferred, bases.Instance):
                        # Const, Tuple or other containers that inherit from
                        # `Instance`
                        continue
                    elif isinstance(inferred, (bases.Proxy, util.UninferableBase)):
                        continue
                    elif inferred.is_function:
                        iattrs = inferred.instance_attrs
                    else:
                        iattrs = inferred.locals
                except AttributeError:
                    # XXX log error
                    continue
                values = iattrs.setdefault(node.attrname, [])
                if node in values:
                    continue
                values.append(node)
        except InferenceError:
            pass


def build_namespace_package_module(name: str, path: Sequence[str]) -> nodes.Module:
    module = nodes.Module(name, path=path, package=True)
    module.postinit(body=[], doc_node=None)
    return module


def parse(
    code: str,
    module_name: str = "",
    path: str | None = None,
    apply_transforms: bool = True,
) -> nodes.Module:
    """Parses a source string in order to obtain an astroid AST from it.

    :param str code: The code for the module.
    :param str module_name: The name for the module, if any
    :param str path: The path for the module
    :param bool apply_transforms:
        Apply the transforms for the give code. Use it if you
        don't want the default transforms to be applied.
    """
    code = textwrap.dedent(code)
    builder = AstroidBuilder(
        manager=AstroidManager(), apply_transforms=apply_transforms
    )
    return builder.string_build(code, modname=module_name, path=path)


def _extract_expressions(node: nodes.NodeNG) -> Iterator[nodes.NodeNG]:
    """Find expressions in a call to _TRANSIENT_FUNCTION and extract them.

    The function walks the AST recursively to search for expressions that
    are wrapped into a call to _TRANSIENT_FUNCTION. If it finds such an
    expression, it completely removes the function call node from the tree,
    replacing it by the wrapped expression inside the parent.

    :param node: An astroid node.
    :type node:  astroid.bases.NodeNG
    :yields: The sequence of wrapped expressions on the modified tree
    expression can be found.
    """
    if not (
        isinstance(node, nodes.Call)
        and isinstance(node.func, nodes.Name)
        and node.func.name == _TRANSIENT_FUNCTION
    ):
        for child in node.get_children():
            yield from _extract_expressions(child)

        return

    real_expr = node.args[0]
    assert node.parent
    real_expr.parent = node.parent
    # Search for node in all _astng_fields (the fields checked when
    # get_children is called) of its parent. Some of those fields may
    # be lists or tuples, in which case the elements need to be checked.
    # When we find it, replace it by real_expr, so that the AST looks
    # like no call to _TRANSIENT_FUNCTION ever took place.
    for name in node.parent._astroid_fields:
        child = getattr(node.parent, name)
        if isinstance(child, list):
            for idx, compound_child in enumerate(child):
                if compound_child is node:
                    child[idx] = real_expr
        elif child is node:
            setattr(node.parent, name, real_expr)
    yield real_expr


def _find_statement_by_line(node: nodes.NodeNG, line: int) -> nodes.NodeNG | None:
    """Extracts the statement on a specific line from an AST.

    If the line number of node matches line, it will be returned;
    otherwise its children are iterated and the function is called
    recursively.

    :param node: An astroid node.
    :type node: astroid.bases.NodeNG
    :param line: The line number of the statement to extract.
    :type line: int
    :returns: The statement on the line, or None if no statement for the line
      can be found.
    :rtype:  astroid.bases.NodeNG or None
    """
    # This is an inaccuracy in the AST: the nodes that can be
    # decorated do not carry explicit information on which line
    # the actual definition (class/def), but .fromline seems to
    # be close enough.
    node_line = (
        node.fromlineno
        if isinstance(node, (nodes.ClassDef, nodes.FunctionDef, nodes.MatchCase))
        else node.lineno
    )

    if node_line == line:
        return node

    for child in node.get_children():
        if result := _find_statement_by_line(child, line):
            return result

    return None


def extract_node(code: str, module_name: str = "") -> nodes.NodeNG | list[nodes.NodeNG]:
    """Parses some Python code as a module and extracts a designated AST node.

    Statements:
     To extract one or more statement nodes, append #@ to the end of the line

     Examples:
       >>> def x():
       >>>   def y():
       >>>     return 1 #@

       The return statement will be extracted.

       >>> class X(object):
       >>>   def meth(self): #@
       >>>     pass

      The function object 'meth' will be extracted.

    Expressions:
     To extract arbitrary expressions, surround them with the fake
     function call __(...). After parsing, the surrounded expression
     will be returned and the whole AST (accessible via the returned
     node's parent attribute) will look like the function call was
     never there in the first place.

     Examples:
       >>> a = __(1)

       The const node will be extracted.

       >>> def x(d=__(foo.bar)): pass

       The node containing the default argument will be extracted.

       >>> def foo(a, b):
       >>>   return 0 < __(len(a)) < b

       The node containing the function call 'len' will be extracted.

    If no statements or expressions are selected, the last toplevel
    statement will be returned.

    If the selected statement is a discard statement, (i.e. an expression
    turned into a statement), the wrapped expression is returned instead.

    For convenience, singleton lists are unpacked.

    :param str code: A piece of Python code that is parsed as
    a module. Will be passed through textwrap.dedent first.
    :param str module_name: The name of the module.
    :returns: The designated node from the parse tree, or a list of nodes.
    """

    def _extract(node: nodes.NodeNG | None) -> nodes.NodeNG | None:
        if isinstance(node, nodes.Expr):
            return node.value

        return node

    requested_lines: list[int] = []
    for idx, line in enumerate(code.splitlines()):
        if line.strip().endswith(_STATEMENT_SELECTOR):
            requested_lines.append(idx + 1)

    tree = parse(code, module_name=module_name)
    if not tree.body:
        raise ValueError("Empty tree, cannot extract from it")

    extracted: list[nodes.NodeNG | None] = []
    if requested_lines:
        extracted = [_find_statement_by_line(tree, line) for line in requested_lines]

    # Modifies the tree.
    extracted.extend(_extract_expressions(tree))

    if not extracted:
        extracted.append(tree.body[-1])

    extracted = [_extract(node) for node in extracted]
    extracted_without_none = [node for node in extracted if node is not None]
    if len(extracted_without_none) == 1:
        return extracted_without_none[0]
    return extracted_without_none


def _extract_single_node(code: str, module_name: str = "") -> nodes.NodeNG:
    """Call extract_node while making sure that only one value is returned."""
    ret = extract_node(code, module_name)
    if isinstance(ret, list):
        return ret[0]
    return ret


def _parse_string(
    data: str, type_comments: bool = True, modname: str | None = None
) -> tuple[ast.Module, ParserModule]:
    parser_module = get_parser_module(type_comments=type_comments)
    try:
        parsed = parser_module.parse(
            data + "\n", type_comments=type_comments, filename=modname
        )
    except SyntaxError as exc:
        # If the type annotations are misplaced for some reason, we do not want
        # to fail the entire parsing of the file, so we need to retry the
        # parsing without type comment support. We use a heuristic for
        # determining if the error is due to type annotations.
        type_annot_related = re.search(r"#\s+type:", exc.text or "")
        if not (type_annot_related and type_comments):
            raise

        parser_module = get_parser_module(type_comments=False)
        parsed = parser_module.parse(data + "\n", type_comments=False)
    return parsed, parser_module


########################################

REDIRECT: Final[dict[str, str]] = {
    "arguments": "Arguments",
    "comprehension": "Comprehension",
    "ListCompFor": "Comprehension",
    "GenExprFor": "Comprehension",
    "excepthandler": "ExceptHandler",
    "keyword": "Keyword",
    "match_case": "MatchCase",
}


class TreeRebuilder:
    """Rebuilds the _ast tree to become an Astroid tree."""

    def __init__(
        self,
        manager: AstroidManager,
        parser_module: ParserModule | None = None,
        data: str | None = None,
    ) -> None:
        self._manager = manager
        self._data = data.split("\n") if data else None
        self._global_names: list[dict[str, list[nodes.Global]]] = []
        self._import_from_nodes: list[nodes.ImportFrom] = []
        self._delayed_assattr: list[nodes.AssignAttr] = []
        self._visit_meths: dict[type[ast.AST], Callable[[ast.AST, NodeNG], NodeNG]] = {}

        self._parser_module = (
            get_parser_module() if parser_module is None else parser_module
        )

    def _get_doc(self, node):
        """Return the doc ast node."""
        try:
            if node.body and isinstance(node.body[0], ast.Expr):
                first_value = node.body[0].value
                if isinstance(first_value, ast.Constant) and isinstance(
                    first_value.value, str
                ):
                    doc_ast_node = first_value
                    node.body = node.body[1:]
                    return node, doc_ast_node
        except IndexError:
            pass  # ast built from scratch
        return node, None

    def _get_context(
        self,
        node: (
            ast.Attribute
            | ast.List
            | ast.Name
            | ast.Subscript
            | ast.Starred
            | ast.Tuple
        ),
    ) -> Context:
        return self._parser_module.context_classes.get(type(node.ctx), Context.Load)

    def _get_position_info(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        parent: nodes.ClassDef | nodes.FunctionDef | nodes.AsyncFunctionDef,
    ) -> Position | None:
        """Return position information for ClassDef and FunctionDef nodes.

        In contrast to AST positions, these only include the actual keyword(s)
        and the class / function name.

        >>> @decorator
        >>> async def some_func(var: int) -> None:
        >>> ^^^^^^^^^^^^^^^^^^^
        """
        if not self._data:
            return None
        end_lineno = node.end_lineno
        if node.body:
            end_lineno = node.body[0].lineno
        # pylint: disable-next=unsubscriptable-object
        data = "\n".join(self._data[node.lineno - 1 : end_lineno])

        start_token: TokenInfo | None = None
        keyword_tokens: tuple[int, ...] = (token.NAME,)
        if isinstance(parent, nodes.AsyncFunctionDef):
            search_token = "async"
        elif isinstance(parent, nodes.FunctionDef):
            search_token = "def"
        else:
            search_token = "class"

        for t in generate_tokens(io.StringIO(data).readline):
            if (
                start_token is not None
                and t.type == token.NAME
                and t.string == node.name
            ):
                break
            if t.type in keyword_tokens:
                if t.string == search_token:
                    start_token = t
                    continue
                if t.string in {"def"}:
                    continue
            start_token = None
        else:
            return None

        return Position(
            lineno=node.lineno + start_token.start[0] - 1,
            col_offset=start_token.start[1],
            end_lineno=node.lineno + t.end[0] - 1,
            end_col_offset=t.end[1],
        )

    def visit_module(
        self, node: ast.Module, modname: str, modpath: str, package: bool
    ) -> nodes.Module:
        """Visit a Module node by returning a fresh instance of it.

        Note: Method not called by 'visit'
        """
        node, doc_ast_node = self._get_doc(node)
        newnode = nodes.Module(
            name=modname,
            file=modpath,
            path=[modpath],
            package=package,
        )
        newnode.postinit(
            [self.visit(child, newnode) for child in node.body],
            doc_node=self.visit(doc_ast_node, newnode),
        )
        return newnode

    def visit(self, node: ast.AST | None, parent: NodeNG) -> NodeNG | None:
        if node is None:
            return None
        if (cls := node.__class__) in self._visit_meths:
            visit_method = self._visit_meths[cls]
        else:
            cls_name = cls.__name__
            visit_name = "visit_" + REDIRECT.get(cls_name, cls_name).lower()
            visit_method = getattr(self, visit_name)
            self._visit_meths[cls] = visit_method
        return visit_method(node, parent)

    def _save_assignment(self, node: nodes.AssignName | nodes.DelName) -> None:
        """Save assignment situation since node.parent is not available yet."""
        if self._global_names and node.name in self._global_names[-1]:
            node.root().set_local(node.name, node)
        else:
            assert node.parent
            assert node.name
            node.parent.set_local(node.name, node)

    def visit_arg(self, node: ast.arg, parent: NodeNG) -> nodes.AssignName:
        """Visit an arg node by returning a fresh AssignName instance."""
        return self.visit_assignname(node, parent, node.arg)

    def visit_arguments(self, node: ast.arguments, parent: NodeNG) -> nodes.Arguments:
        """Visit an Arguments node by returning a fresh instance of it."""
        vararg: str | None = None
        kwarg: str | None = None
        vararg_node = node.vararg
        kwarg_node = node.kwarg

        newnode = nodes.Arguments(
            node.vararg.arg if node.vararg else None,
            node.kwarg.arg if node.kwarg else None,
            parent,
            (
                AssignName(
                    vararg_node.arg,
                    vararg_node.lineno,
                    vararg_node.col_offset,
                    parent,
                    end_lineno=vararg_node.end_lineno,
                    end_col_offset=vararg_node.end_col_offset,
                )
                if vararg_node
                else None
            ),
            (
                AssignName(
                    kwarg_node.arg,
                    kwarg_node.lineno,
                    kwarg_node.col_offset,
                    parent,
                    end_lineno=kwarg_node.end_lineno,
                    end_col_offset=kwarg_node.end_col_offset,
                )
                if kwarg_node
                else None
            ),
        )
        args = [self.visit(child, newnode) for child in node.args]
        defaults = [self.visit(child, newnode) for child in node.defaults]
        varargannotation: NodeNG | None = None
        kwargannotation: NodeNG | None = None
        if node.vararg:
            vararg = node.vararg.arg
            varargannotation = self.visit(node.vararg.annotation, newnode)
        if node.kwarg:
            kwarg = node.kwarg.arg
            kwargannotation = self.visit(node.kwarg.annotation, newnode)

        kwonlyargs = [self.visit(child, newnode) for child in node.kwonlyargs]
        kw_defaults = [self.visit(child, newnode) for child in node.kw_defaults]
        annotations = [self.visit(arg.annotation, newnode) for arg in node.args]
        kwonlyargs_annotations = [
            self.visit(arg.annotation, newnode) for arg in node.kwonlyargs
        ]

        posonlyargs = [self.visit(child, newnode) for child in node.posonlyargs]
        posonlyargs_annotations = [
            self.visit(arg.annotation, newnode) for arg in node.posonlyargs
        ]
        type_comment_args = [
            self.check_type_comment(child, parent=newnode) for child in node.args
        ]
        type_comment_kwonlyargs = [
            self.check_type_comment(child, parent=newnode) for child in node.kwonlyargs
        ]
        type_comment_posonlyargs = [
            self.check_type_comment(child, parent=newnode) for child in node.posonlyargs
        ]

        newnode.postinit(
            args=args,
            defaults=defaults,
            kwonlyargs=kwonlyargs,
            posonlyargs=posonlyargs,
            kw_defaults=kw_defaults,
            annotations=annotations,
            kwonlyargs_annotations=kwonlyargs_annotations,
            posonlyargs_annotations=posonlyargs_annotations,
            varargannotation=varargannotation,
            kwargannotation=kwargannotation,
            type_comment_args=type_comment_args,
            type_comment_kwonlyargs=type_comment_kwonlyargs,
            type_comment_posonlyargs=type_comment_posonlyargs,
        )
        # save argument names in locals:
        assert newnode.parent
        if vararg:
            newnode.parent.set_local(vararg, newnode)
        if kwarg:
            newnode.parent.set_local(kwarg, newnode)
        return newnode

    def visit_assert(self, node: ast.Assert, parent: NodeNG) -> nodes.Assert:
        """Visit a Assert node by returning a fresh instance of it."""
        newnode = nodes.Assert(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        msg: NodeNG | None = None
        if node.msg:
            msg = self.visit(node.msg, newnode)
        newnode.postinit(self.visit(node.test, newnode), msg)
        return newnode

    def check_type_comment(
        self,
        node: ast.Assign | ast.arg | ast.For | ast.AsyncFor | ast.With | ast.AsyncWith,
        parent: (
            nodes.Assign
            | nodes.Arguments
            | nodes.For
            | nodes.AsyncFor
            | nodes.With
            | nodes.AsyncWith
        ),
    ) -> NodeNG | None:
        if not node.type_comment:
            return None

        try:
            type_comment_ast = self._parser_module.parse(node.type_comment)
        except SyntaxError:
            # Invalid type comment, just skip it.
            return None

        # For '# type: # any comment' ast.parse returns a Module node,
        # without any nodes in the body.
        if not type_comment_ast.body:
            return None

        type_object = self.visit(type_comment_ast.body[0], parent=parent)
        if not isinstance(type_object, nodes.Expr):
            return None

        return type_object.value

    def check_function_type_comment(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, parent: NodeNG
    ) -> tuple[NodeNG | None, list[NodeNG]] | None:
        if not node.type_comment:
            return None

        try:
            type_comment_ast = parse_function_type_comment(node.type_comment)
        except SyntaxError:
            # Invalid type comment, just skip it.
            return None

        if not type_comment_ast:
            return None

        returns: NodeNG | None = None
        argtypes: list[NodeNG] = [
            self.visit(elem, parent) for elem in (type_comment_ast.argtypes or [])
        ]
        if type_comment_ast.returns:
            returns = self.visit(type_comment_ast.returns, parent)

        return returns, argtypes

    def visit_asyncfunctiondef(
        self, node: ast.AsyncFunctionDef, parent: NodeNG
    ) -> nodes.AsyncFunctionDef:
        return self._visit_functiondef(nodes.AsyncFunctionDef, node, parent)

    def visit_asyncfor(self, node: ast.AsyncFor, parent: NodeNG) -> nodes.AsyncFor:
        return self._visit_for(nodes.AsyncFor, node, parent)

    def visit_await(self, node: ast.Await, parent: NodeNG) -> nodes.Await:
        newnode = nodes.Await(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(value=self.visit(node.value, newnode))
        return newnode

    def visit_asyncwith(self, node: ast.AsyncWith, parent: NodeNG) -> nodes.AsyncWith:
        return self._visit_with(nodes.AsyncWith, node, parent)

    def visit_assign(self, node: ast.Assign, parent: NodeNG) -> nodes.Assign:
        """Visit a Assign node by returning a fresh instance of it."""
        newnode = nodes.Assign(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        type_annotation = self.check_type_comment(node, parent=newnode)
        newnode.postinit(
            targets=[self.visit(child, newnode) for child in node.targets],
            value=self.visit(node.value, newnode),
            type_annotation=type_annotation,
        )
        return newnode

    def visit_annassign(self, node: ast.AnnAssign, parent: NodeNG) -> nodes.AnnAssign:
        """Visit an AnnAssign node by returning a fresh instance of it."""
        newnode = nodes.AnnAssign(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(
            target=self.visit(node.target, newnode),
            annotation=self.visit(node.annotation, newnode),
            simple=node.simple,
            value=self.visit(node.value, newnode),
        )
        return newnode

    def visit_assignname(
        self, node: ast.AST, parent: NodeNG, node_name: str | None
    ) -> nodes.AssignName | None:
        """Visit a node and return a AssignName node.

        Note: Method not called by 'visit'
        """
        if node_name is None:
            return None
        newnode = nodes.AssignName(
            name=node_name,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        self._save_assignment(newnode)
        return newnode

    def visit_augassign(self, node: ast.AugAssign, parent: NodeNG) -> nodes.AugAssign:
        """Visit a AugAssign node by returning a fresh instance of it."""
        newnode = nodes.AugAssign(
            op=self._parser_module.bin_op_classes[type(node.op)] + "=",
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(
            self.visit(node.target, newnode), self.visit(node.value, newnode)
        )
        return newnode

    def visit_binop(self, node: ast.BinOp, parent: NodeNG) -> nodes.BinOp:
        """Visit a BinOp node by returning a fresh instance of it."""
        newnode = nodes.BinOp(
            op=self._parser_module.bin_op_classes[type(node.op)],
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(
            self.visit(node.left, newnode), self.visit(node.right, newnode)
        )
        return newnode

    def visit_boolop(self, node: ast.BoolOp, parent: NodeNG) -> nodes.BoolOp:
        """Visit a BoolOp node by returning a fresh instance of it."""
        newnode = nodes.BoolOp(
            op=self._parser_module.bool_op_classes[type(node.op)],
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit([self.visit(child, newnode) for child in node.values])
        return newnode

    def visit_break(self, node: ast.Break, parent: NodeNG) -> nodes.Break:
        """Visit a Break node by returning a fresh instance of it."""
        return nodes.Break(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )

    def visit_call(self, node: ast.Call, parent: NodeNG) -> nodes.Call:
        """Visit a CallFunc node by returning a fresh instance of it."""
        newnode = nodes.Call(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(
            func=self.visit(node.func, newnode),
            args=[self.visit(child, newnode) for child in node.args],
            keywords=[self.visit(child, newnode) for child in node.keywords],
        )
        return newnode

    def visit_classdef(
        self, node: ast.ClassDef, parent: NodeNG, newstyle: bool = True
    ) -> nodes.ClassDef:
        """Visit a ClassDef node to become astroid."""
        node, doc_ast_node = self._get_doc(node)
        newnode = nodes.ClassDef(
            name=node.name,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        metaclass = None
        for keyword in node.keywords:
            if keyword.arg == "metaclass":
                metaclass = self.visit(keyword, newnode).value
                break
        decorators = self.visit_decorators(node, newnode)
        newnode.postinit(
            [self.visit(child, newnode) for child in node.bases],
            [self.visit(child, newnode) for child in node.body],
            decorators,
            newstyle,
            metaclass,
            [
                self.visit(kwd, newnode)
                for kwd in node.keywords
                if kwd.arg != "metaclass"
            ],
            position=self._get_position_info(node, newnode),
            doc_node=self.visit(doc_ast_node, newnode),
            type_params=(
                [self.visit(param, newnode) for param in node.type_params]
                if PY312_PLUS
                else []
            ),
        )
        parent.set_local(newnode.name, newnode)
        return newnode

    def visit_continue(self, node: ast.Continue, parent: NodeNG) -> nodes.Continue:
        """Visit a Continue node by returning a fresh instance of it."""
        return nodes.Continue(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )

    def visit_compare(self, node: ast.Compare, parent: NodeNG) -> nodes.Compare:
        """Visit a Compare node by returning a fresh instance of it."""
        newnode = nodes.Compare(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(
            self.visit(node.left, newnode),
            [
                (
                    self._parser_module.cmp_op_classes[op.__class__],
                    self.visit(expr, newnode),
                )
                for (op, expr) in zip(node.ops, node.comparators)
            ],
        )
        return newnode

    def visit_comprehension(
        self, node: ast.comprehension, parent: NodeNG
    ) -> nodes.Comprehension:
        """Visit a Comprehension node by returning a fresh instance of it."""
        newnode = nodes.Comprehension(
            parent=parent,
            # Comprehension nodes don't have these attributes
            # see https://docs.python.org/3/library/ast.html#abstract-grammar
            lineno=None,
            col_offset=None,
            end_lineno=None,
            end_col_offset=None,
        )
        newnode.postinit(
            self.visit(node.target, newnode),
            self.visit(node.iter, newnode),
            [self.visit(child, newnode) for child in node.ifs],
            bool(node.is_async),
        )
        return newnode

    def visit_decorators(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        parent: NodeNG,
    ) -> nodes.Decorators | None:
        """Visit a Decorators node by returning a fresh instance of it.

        Note: Method not called by 'visit'
        """
        if not node.decorator_list:
            return None
        # /!\ node is actually an _ast.FunctionDef node while
        # parent is an astroid.nodes.FunctionDef node

        # Set the line number of the first decorator for Python 3.8+.
        lineno = node.decorator_list[0].lineno
        end_lineno = node.decorator_list[-1].end_lineno
        end_col_offset = node.decorator_list[-1].end_col_offset

        newnode = nodes.Decorators(
            lineno=lineno,
            col_offset=node.col_offset,
            end_lineno=end_lineno,
            end_col_offset=end_col_offset,
            parent=parent,
        )
        newnode.postinit([self.visit(child, newnode) for child in node.decorator_list])
        return newnode

    def visit_delete(self, node: ast.Delete, parent: NodeNG) -> nodes.Delete:
        """Visit a Delete node by returning a fresh instance of it."""
        newnode = nodes.Delete(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit([self.visit(child, newnode) for child in node.targets])
        return newnode

    def _visit_dict_items(
        self, node: ast.Dict, parent: NodeNG, newnode: nodes.Dict
    ) -> Generator[tuple[NodeNG, NodeNG]]:
        for key, value in zip(node.keys, node.values):
            rebuilt_key: NodeNG
            rebuilt_value = self.visit(value, newnode)
            rebuilt_key = (
                self.visit(key, newnode)
                if key
                else
                # Extended unpacking
                nodes.DictUnpack(
                    lineno=rebuilt_value.lineno,
                    col_offset=rebuilt_value.col_offset,
                    end_lineno=rebuilt_value.end_lineno,
                    end_col_offset=rebuilt_value.end_col_offset,
                    parent=parent,
                )
            )
            yield rebuilt_key, rebuilt_value

    def visit_dict(self, node: ast.Dict, parent: NodeNG) -> nodes.Dict:
        """Visit a Dict node by returning a fresh instance of it."""
        newnode = nodes.Dict(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        items = list(self._visit_dict_items(node, parent, newnode))
        newnode.postinit(items)
        return newnode

    def visit_dictcomp(self, node: ast.DictComp, parent: NodeNG) -> nodes.DictComp:
        """Visit a DictComp node by returning a fresh instance of it."""
        newnode = nodes.DictComp(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(
            self.visit(node.key, newnode),
            self.visit(node.value, newnode),
            [self.visit(child, newnode) for child in node.generators],
        )
        return newnode

    def visit_expr(self, node: ast.Expr, parent: NodeNG) -> nodes.Expr:
        """Visit a Expr node by returning a fresh instance of it."""
        newnode = nodes.Expr(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(self.visit(node.value, newnode))
        return newnode

    def visit_excepthandler(
        self, node: ast.ExceptHandler, parent: NodeNG
    ) -> nodes.ExceptHandler:
        """Visit an ExceptHandler node by returning a fresh instance of it."""
        newnode = nodes.ExceptHandler(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(
            self.visit(node.type, newnode),
            self.visit_assignname(node, newnode, node.name),
            [self.visit(child, newnode) for child in node.body],
        )
        return newnode

    def _visit_for(
        self,
        cls,
        node: ast.For | ast.AsyncFor,
        parent: NodeNG,
    ):
        """Visit a For node by returning a fresh instance of it."""
        newnode = cls(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        type_annotation = self.check_type_comment(node, parent=newnode)
        newnode.postinit(
            target=self.visit(node.target, newnode),
            iter=self.visit(node.iter, newnode),
            body=[self.visit(child, newnode) for child in node.body],
            orelse=[self.visit(child, newnode) for child in node.orelse],
            type_annotation=type_annotation,
        )
        return newnode

    def visit_for(self, node: ast.For, parent: NodeNG) -> nodes.For:
        return self._visit_for(nodes.For, node, parent)

    def visit_importfrom(
        self, node: ast.ImportFrom, parent: NodeNG
    ) -> nodes.ImportFrom:
        """Visit an ImportFrom node by returning a fresh instance of it."""
        names = [(alias.name, alias.asname) for alias in node.names]
        newnode = nodes.ImportFrom(
            fromname=node.module or "",
            names=names,
            level=node.level or None,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        # store From names to add them to locals after building
        self._import_from_nodes.append(newnode)
        return newnode

    def _visit_functiondef(
        self,
        cls,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent: NodeNG,
    ):
        """Visit an FunctionDef node to become astroid."""
        self._global_names.append({})
        node, doc_ast_node = self._get_doc(node)

        lineno = node.lineno
        if node.decorator_list:
            # Python 3.8 sets the line number of a decorated function
            # to be the actual line number of the function, but the
            # previous versions expected the decorator's line number instead.
            # We reset the function's line number to that of the
            # first decorator to maintain backward compatibility.
            # It's not ideal but this discrepancy was baked into
            # the framework for *years*.
            lineno = node.decorator_list[0].lineno

        newnode = cls(
            name=node.name,
            lineno=lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        decorators = self.visit_decorators(node, newnode)

        returns: NodeNG | None = (
            self.visit(node.returns, newnode) if node.returns else None
        )

        type_comment_args = type_comment_returns = None
        if type_comment_annotation := self.check_function_type_comment(node, newnode):
            type_comment_returns, type_comment_args = type_comment_annotation
        newnode.postinit(
            args=self.visit(node.args, newnode),
            body=[self.visit(child, newnode) for child in node.body],
            decorators=decorators,
            returns=returns,
            type_comment_returns=type_comment_returns,
            type_comment_args=type_comment_args,
            position=self._get_position_info(node, newnode),
            doc_node=self.visit(doc_ast_node, newnode),
            type_params=(
                [self.visit(param, newnode) for param in node.type_params]
                if PY312_PLUS
                else []
            ),
        )
        self._global_names.pop()
        parent.set_local(newnode.name, newnode)
        return newnode

    def visit_functiondef(
        self, node: ast.FunctionDef, parent: NodeNG
    ) -> nodes.FunctionDef:
        return self._visit_functiondef(nodes.FunctionDef, node, parent)

    def visit_generatorexp(
        self, node: ast.GeneratorExp, parent: NodeNG
    ) -> nodes.GeneratorExp:
        """Visit a GeneratorExp node by returning a fresh instance of it."""
        newnode = nodes.GeneratorExp(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(
            self.visit(node.elt, newnode),
            [self.visit(child, newnode) for child in node.generators],
        )
        return newnode

    def visit_attribute(
        self, node: ast.Attribute, parent: NodeNG
    ) -> nodes.Attribute | nodes.AssignAttr | nodes.DelAttr:
        """Visit an Attribute node by returning a fresh instance of it."""
        context = self._get_context(node)
        newnode: nodes.Attribute | nodes.AssignAttr | nodes.DelAttr
        if context == Context.Del:
            # FIXME : maybe we should reintroduce and visit_delattr ?
            # for instance, deactivating assign_ctx
            newnode = nodes.DelAttr(
                attrname=node.attr,
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
                parent=parent,
            )
        elif context == Context.Store:
            newnode = nodes.AssignAttr(
                attrname=node.attr,
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
                parent=parent,
            )
            # Prohibit a local save if we are in an ExceptHandler.
            if not isinstance(parent, nodes.ExceptHandler):
                # mypy doesn't recognize that newnode has to be AssignAttr because it
                # doesn't support ParamSpec
                # See https://github.com/python/mypy/issues/8645
                self._delayed_assattr.append(newnode)
        else:
            newnode = nodes.Attribute(
                attrname=node.attr,
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
                parent=parent,
            )
        newnode.postinit(self.visit(node.value, newnode))
        return newnode

    def visit_global(self, node: ast.Global, parent: NodeNG) -> nodes.Global:
        """Visit a Global node to become astroid."""
        newnode = nodes.Global(
            names=node.names,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        if self._global_names:  # global at the module level, no effect
            for name in node.names:
                self._global_names[-1].setdefault(name, []).append(newnode)
        return newnode

    def visit_if(self, node: ast.If, parent: NodeNG) -> nodes.If:
        """Visit an If node by returning a fresh instance of it."""
        newnode = nodes.If(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(
            self.visit(node.test, newnode),
            [self.visit(child, newnode) for child in node.body],
            [self.visit(child, newnode) for child in node.orelse],
        )
        return newnode

    def visit_ifexp(self, node: ast.IfExp, parent: NodeNG) -> nodes.IfExp:
        """Visit a IfExp node by returning a fresh instance of it."""
        newnode = nodes.IfExp(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(
            self.visit(node.test, newnode),
            self.visit(node.body, newnode),
            self.visit(node.orelse, newnode),
        )
        return newnode

    def visit_import(self, node: ast.Import, parent: NodeNG) -> nodes.Import:
        """Visit a Import node by returning a fresh instance of it."""
        names = [(alias.name, alias.asname) for alias in node.names]
        newnode = nodes.Import(
            names=names,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        # save import names in parent's locals:
        for name, asname in newnode.names:
            name = asname or name
            parent.set_local(name.split(".")[0], newnode)
        return newnode

    def visit_joinedstr(self, node: ast.JoinedStr, parent: NodeNG) -> nodes.JoinedStr:
        newnode = nodes.JoinedStr(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit([self.visit(child, newnode) for child in node.values])
        return newnode

    def visit_formattedvalue(
        self, node: ast.FormattedValue, parent: NodeNG
    ) -> nodes.FormattedValue:
        newnode = nodes.FormattedValue(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(
            value=self.visit(node.value, newnode),
            conversion=node.conversion,
            format_spec=self.visit(node.format_spec, newnode),
        )
        return newnode

    def visit_namedexpr(self, node: ast.NamedExpr, parent: NodeNG) -> nodes.NamedExpr:
        newnode = nodes.NamedExpr(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(
            self.visit(node.target, newnode), self.visit(node.value, newnode)
        )
        return newnode

    def visit_keyword(self, node: ast.keyword, parent: NodeNG) -> nodes.Keyword:
        """Visit a Keyword node by returning a fresh instance of it."""
        newnode = nodes.Keyword(
            arg=node.arg,
            # position attributes added in 3.9
            lineno=getattr(node, "lineno", None),
            col_offset=getattr(node, "col_offset", None),
            end_lineno=getattr(node, "end_lineno", None),
            end_col_offset=getattr(node, "end_col_offset", None),
            parent=parent,
        )
        newnode.postinit(self.visit(node.value, newnode))
        return newnode

    def visit_lambda(self, node: ast.Lambda, parent: NodeNG) -> nodes.Lambda:
        """Visit a Lambda node by returning a fresh instance of it."""
        newnode = nodes.Lambda(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(self.visit(node.args, newnode), self.visit(node.body, newnode))
        return newnode

    def visit_list(self, node: ast.List, parent: NodeNG) -> nodes.List:
        """Visit a List node by returning a fresh instance of it."""
        context = self._get_context(node)
        newnode = nodes.List(
            ctx=context,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit([self.visit(child, newnode) for child in node.elts])
        return newnode

    def visit_listcomp(self, node: ast.ListComp, parent: NodeNG) -> nodes.ListComp:
        """Visit a ListComp node by returning a fresh instance of it."""
        newnode = nodes.ListComp(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(
            self.visit(node.elt, newnode),
            [self.visit(child, newnode) for child in node.generators],
        )
        return newnode

    def visit_name(
        self, node: ast.Name, parent: NodeNG
    ) -> nodes.Name | nodes.AssignName | nodes.DelName:
        """Visit a Name node by returning a fresh instance of it."""
        context = self._get_context(node)
        newnode: nodes.Name | nodes.AssignName | nodes.DelName
        if context == Context.Del:
            newnode = nodes.DelName(
                name=node.id,
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
                parent=parent,
            )
        elif context == Context.Store:
            newnode = nodes.AssignName(
                name=node.id,
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
                parent=parent,
            )
        else:
            newnode = nodes.Name(
                name=node.id,
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
                parent=parent,
            )
        # XXX REMOVE me :
        if context in (Context.Del, Context.Store):  # 'Aug' ??
            self._save_assignment(newnode)
        return newnode

    def visit_nonlocal(self, node: ast.Nonlocal, parent: NodeNG) -> nodes.Nonlocal:
        """Visit a Nonlocal node and return a new instance of it."""
        return nodes.Nonlocal(
            names=node.names,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )

    def visit_constant(self, node: ast.Constant, parent: NodeNG) -> nodes.Const:
        """Visit a Constant node by returning a fresh instance of Const."""
        return nodes.Const(
            value=node.value,
            kind=node.kind,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )

    def visit_paramspec(self, node: ast.ParamSpec, parent: NodeNG) -> nodes.ParamSpec:
        """Visit a ParamSpec node by returning a fresh instance of it."""
        newnode = nodes.ParamSpec(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        # Add AssignName node for 'node.name'
        # https://bugs.python.org/issue43994
        newnode.postinit(name=self.visit_assignname(node, newnode, node.name))
        return newnode

    def visit_pass(self, node: ast.Pass, parent: NodeNG) -> nodes.Pass:
        """Visit a Pass node by returning a fresh instance of it."""
        return nodes.Pass(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )

    def visit_raise(self, node: ast.Raise, parent: NodeNG) -> nodes.Raise:
        """Visit a Raise node by returning a fresh instance of it."""
        newnode = nodes.Raise(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        # no traceback; anyway it is not used in Pylint
        newnode.postinit(
            exc=self.visit(node.exc, newnode),
            cause=self.visit(node.cause, newnode),
        )
        return newnode

    def visit_return(self, node: ast.Return, parent: NodeNG) -> nodes.Return:
        """Visit a Return node by returning a fresh instance of it."""
        newnode = nodes.Return(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(self.visit(node.value, newnode))
        return newnode

    def visit_set(self, node: ast.Set, parent: NodeNG) -> nodes.Set:
        """Visit a Set node by returning a fresh instance of it."""
        newnode = nodes.Set(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit([self.visit(child, newnode) for child in node.elts])
        return newnode

    def visit_setcomp(self, node: ast.SetComp, parent: NodeNG) -> nodes.SetComp:
        """Visit a SetComp node by returning a fresh instance of it."""
        newnode = nodes.SetComp(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(
            self.visit(node.elt, newnode),
            [self.visit(child, newnode) for child in node.generators],
        )
        return newnode

    def visit_slice(self, node: ast.Slice, parent: nodes.Subscript) -> nodes.Slice:
        """Visit a Slice node by returning a fresh instance of it."""
        newnode = nodes.Slice(
            # position attributes added in 3.9
            lineno=getattr(node, "lineno", None),
            col_offset=getattr(node, "col_offset", None),
            end_lineno=getattr(node, "end_lineno", None),
            end_col_offset=getattr(node, "end_col_offset", None),
            parent=parent,
        )
        newnode.postinit(
            lower=self.visit(node.lower, newnode),
            upper=self.visit(node.upper, newnode),
            step=self.visit(node.step, newnode),
        )
        return newnode

    def visit_subscript(self, node: ast.Subscript, parent: NodeNG) -> nodes.Subscript:
        """Visit a Subscript node by returning a fresh instance of it."""
        context = self._get_context(node)
        newnode = nodes.Subscript(
            ctx=context,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(
            self.visit(node.value, newnode), self.visit(node.slice, newnode)
        )
        return newnode

    def visit_starred(self, node: ast.Starred, parent: NodeNG) -> nodes.Starred:
        """Visit a Starred node and return a new instance of it."""
        context = self._get_context(node)
        newnode = nodes.Starred(
            ctx=context,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(self.visit(node.value, newnode))
        return newnode

    def visit_try(self, node: ast.Try, parent: NodeNG) -> nodes.Try:
        """Visit a Try node by returning a fresh instance of it"""
        newnode = nodes.Try(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(
            body=[self.visit(child, newnode) for child in node.body],
            handlers=[self.visit(child, newnode) for child in node.handlers],
            orelse=[self.visit(child, newnode) for child in node.orelse],
            finalbody=[self.visit(child, newnode) for child in node.finalbody],
        )
        return newnode

    def visit_trystar(self, node: ast.TryStar, parent: NodeNG) -> nodes.TryStar:
        newnode = nodes.TryStar(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(
            body=[self.visit(n, newnode) for n in node.body],
            handlers=[self.visit(n, newnode) for n in node.handlers],
            orelse=[self.visit(n, newnode) for n in node.orelse],
            finalbody=[self.visit(n, newnode) for n in node.finalbody],
        )
        return newnode

    def visit_tuple(self, node: ast.Tuple, parent: NodeNG) -> nodes.Tuple:
        """Visit a Tuple node by returning a fresh instance of it."""
        context = self._get_context(node)
        newnode = nodes.Tuple(
            ctx=context,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit([self.visit(child, newnode) for child in node.elts])
        return newnode

    def visit_typealias(self, node: ast.TypeAlias, parent: NodeNG) -> nodes.TypeAlias:
        """Visit a TypeAlias node by returning a fresh instance of it."""
        newnode = nodes.TypeAlias(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(
            name=self.visit(node.name, newnode),
            type_params=[self.visit(p, newnode) for p in node.type_params],
            value=self.visit(node.value, newnode),
        )
        return newnode

    def visit_typevar(self, node: ast.TypeVar, parent: NodeNG) -> nodes.TypeVar:
        """Visit a TypeVar node by returning a fresh instance of it."""
        newnode = nodes.TypeVar(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        # Add AssignName node for 'node.name'
        # https://bugs.python.org/issue43994
        newnode.postinit(
            name=self.visit_assignname(node, newnode, node.name),
            bound=self.visit(node.bound, newnode),
        )
        return newnode

    def visit_typevartuple(
        self, node: ast.TypeVarTuple, parent: NodeNG
    ) -> nodes.TypeVarTuple:
        """Visit a TypeVarTuple node by returning a fresh instance of it."""
        newnode = nodes.TypeVarTuple(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        # Add AssignName node for 'node.name'
        # https://bugs.python.org/issue43994
        newnode.postinit(name=self.visit_assignname(node, newnode, node.name))
        return newnode

    def visit_unaryop(self, node: ast.UnaryOp, parent: NodeNG) -> nodes.UnaryOp:
        """Visit a UnaryOp node by returning a fresh instance of it."""
        newnode = nodes.UnaryOp(
            op=self._parser_module.unary_op_classes[node.op.__class__],
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(self.visit(node.operand, newnode))
        return newnode

    def visit_while(self, node: ast.While, parent: NodeNG) -> nodes.While:
        """Visit a While node by returning a fresh instance of it."""
        newnode = nodes.While(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(
            self.visit(node.test, newnode),
            [self.visit(child, newnode) for child in node.body],
            [self.visit(child, newnode) for child in node.orelse],
        )
        return newnode

    def _visit_with(
        self,
        cls,
        node: ast.With | ast.AsyncWith,
        parent: NodeNG,
    ):
        newnode = cls(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )

        def visit_child(child: ast.withitem) -> tuple[NodeNG, NodeNG | None]:
            expr = self.visit(child.context_expr, newnode)
            var = self.visit(child.optional_vars, newnode)
            return expr, var

        type_annotation = self.check_type_comment(node, parent=newnode)
        newnode.postinit(
            items=[visit_child(child) for child in node.items],
            body=[self.visit(child, newnode) for child in node.body],
            type_annotation=type_annotation,
        )
        return newnode

    def visit_with(self, node: ast.With, parent: NodeNG) -> NodeNG:
        return self._visit_with(nodes.With, node, parent)

    def visit_yield(self, node: ast.Yield, parent: NodeNG) -> NodeNG:
        """Visit a Yield node by returning a fresh instance of it."""
        newnode = nodes.Yield(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(self.visit(node.value, newnode))
        return newnode

    def visit_yieldfrom(self, node: ast.YieldFrom, parent: NodeNG) -> NodeNG:
        newnode = nodes.YieldFrom(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            parent=parent,
        )
        newnode.postinit(self.visit(node.value, newnode))
        return newnode

    if sys.version_info >= (3, 10):

        def visit_match(self, node: ast.Match, parent: NodeNG) -> nodes.Match:
            newnode = nodes.Match(
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
                parent=parent,
            )
            newnode.postinit(
                subject=self.visit(node.subject, newnode),
                cases=[self.visit(case, newnode) for case in node.cases],
            )
            return newnode

        def visit_matchcase(
            self, node: ast.match_case, parent: NodeNG
        ) -> nodes.MatchCase:
            newnode = nodes.MatchCase(parent=parent)
            newnode.postinit(
                pattern=self.visit(node.pattern, newnode),
                guard=self.visit(node.guard, newnode),
                body=[self.visit(child, newnode) for child in node.body],
            )
            return newnode

        def visit_matchvalue(
            self, node: ast.MatchValue, parent: NodeNG
        ) -> nodes.MatchValue:
            newnode = nodes.MatchValue(
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
                parent=parent,
            )
            newnode.postinit(value=self.visit(node.value, newnode))
            return newnode

        def visit_matchsingleton(
            self, node: ast.MatchSingleton, parent: NodeNG
        ) -> nodes.MatchSingleton:
            return nodes.MatchSingleton(
                value=node.value,
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
                parent=parent,
            )

        def visit_matchsequence(
            self, node: ast.MatchSequence, parent: NodeNG
        ) -> nodes.MatchSequence:
            newnode = nodes.MatchSequence(
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
                parent=parent,
            )
            newnode.postinit(
                patterns=[self.visit(pattern, newnode) for pattern in node.patterns]
            )
            return newnode

        def visit_matchmapping(
            self, node: ast.MatchMapping, parent: NodeNG
        ) -> nodes.MatchMapping:
            newnode = nodes.MatchMapping(
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
                parent=parent,
            )
            # Add AssignName node for 'node.name'
            # https://bugs.python.org/issue43994
            newnode.postinit(
                keys=[self.visit(child, newnode) for child in node.keys],
                patterns=[self.visit(pattern, newnode) for pattern in node.patterns],
                rest=self.visit_assignname(node, newnode, node.rest),
            )
            return newnode

        def visit_matchclass(
            self, node: ast.MatchClass, parent: NodeNG
        ) -> nodes.MatchClass:
            newnode = nodes.MatchClass(
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
                parent=parent,
            )
            newnode.postinit(
                cls=self.visit(node.cls, newnode),
                patterns=[self.visit(pattern, newnode) for pattern in node.patterns],
                kwd_attrs=node.kwd_attrs,
                kwd_patterns=[
                    self.visit(pattern, newnode) for pattern in node.kwd_patterns
                ],
            )
            return newnode

        def visit_matchstar(
            self, node: ast.MatchStar, parent: NodeNG
        ) -> nodes.MatchStar:
            newnode = nodes.MatchStar(
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
                parent=parent,
            )
            # Add AssignName node for 'node.name'
            # https://bugs.python.org/issue43994
            newnode.postinit(name=self.visit_assignname(node, newnode, node.name))
            return newnode

        def visit_matchas(self, node: ast.MatchAs, parent: NodeNG) -> nodes.MatchAs:
            newnode = nodes.MatchAs(
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
                parent=parent,
            )
            # Add AssignName node for 'node.name'
            # https://bugs.python.org/issue43994
            newnode.postinit(
                pattern=self.visit(node.pattern, newnode),
                name=self.visit_assignname(node, newnode, node.name),
            )
            return newnode

        def visit_matchor(self, node: ast.MatchOr, parent: NodeNG) -> nodes.MatchOr:
            newnode = nodes.MatchOr(
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
                parent=parent,
            )
            newnode.postinit(
                patterns=[self.visit(pattern, newnode) for pattern in node.patterns]
            )
            return newnode
