# Copyright (c) 2015-2016 Claudiu Popa <pcmanticore@gmail.com>

# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/master/COPYING.LESSER


import collections
import warnings

from astroid.node_classes import (
    Arguments, AssignAttr, Assert, Assign, AnnAssign,
    AssignName, AugAssign, Repr, BinOp, BoolOp, Break, Call, Compare,
    Comprehension, Const, Continue, Decorators, DelAttr, DelName, Delete,
    Dict, Expr, Ellipsis, EmptyNode, ExceptHandler, Exec, ExtSlice, For,
    ImportFrom, Attribute, Global, If, IfExp, Import, Index, Keyword,
    List, Name, Nonlocal, Pass, Print, Raise, Return, Set, Slice, Starred, Subscript,
    TryExcept, TryFinally, Tuple, UnaryOp, While, With, Yield, YieldFrom,
    const_factory,
    AsyncFor, Await, AsyncWith,
    FormattedValue, JoinedStr,
)

class TransformVisitor(object):
    """A visitor for handling transforms.

    The standard approach of using it is to call
    :meth:`~visit` with an *astroid* module and the class
    will take care of the rest, walking the tree and running the
    transforms for each encountered node.
    """

    def __init__(self):
        self.transforms = collections.defaultdict(list)

    def _transform(self, node):
        """Call matching transforms for the given node if any and return the
        transformed node.
        """
        cls = node.__class__
        if cls not in self.transforms:
            # no transform registered for this class of node
            return node

        transforms = self.transforms[cls]
        orig_node = node  # copy the reference
        for transform_func, predicate in transforms:
            if predicate is None or predicate(node):
                ret = transform_func(node)
                # if the transformation function returns something, it's
                # expected to be a replacement for the node
                if ret is not None:
                    if node is not orig_node:
                        # node has already be modified by some previous
                        # transformation, warn about it
                        warnings.warn('node %s substituted multiple times' % node)
                    node = ret
        return node

    def _visit(self, node):
        if node is None or isinstance(node, str):
            return node

        if isinstance(node, (AssignName, Const, Import, Name, Pass)):
            pass
        elif isinstance(node, (Index, Keyword, Return)):
            node.value = self._visit(node.value)
        elif isinstance(node, Assign):
            node.value = self._visit(node.value)
            node.targets = [self._visit(elt) for elt in node.targets]
        elif isinstance(node, Call):
            node.func = self._visit(node.func)
            node.args = [self._visit(elt) for elt in node.args]
            if node.keywords is not None:
                node.keywords = [self._visit(elt) for elt in node.keywords]
        elif isinstance(node, Compare):
            node.left = self._visit(node.left)
            node.ops = [
                (string, self._visit(child))
                for (string, child) in node.ops
            ]
        elif isinstance(node, BinOp):
            node.left = self._visit(node.left)
            node.right = self._visit(node.right)
        else:
            # print(type(node))
            for field in node._astroid_fields:
                value = getattr(node, field)
                visited = self._visit_generic(value)
                setattr(node, field, visited)

        return self._transform(node)

    def _visit_generic(self, node):
        if isinstance(node, list):
            return [self._visit_generic(child) for child in node]
        elif isinstance(node, tuple):
            return tuple(self._visit_generic(child) for child in node)

        return self._visit(node)

    def register_transform(self, node_class, transform, predicate=None):
        """Register `transform(node)` function to be applied on the given
        astroid's `node_class` if `predicate` is None or returns true
        when called with the node as argument.

        The transform function may return a value which is then used to
        substitute the original node in the tree.
        """
        self.transforms[node_class].append((transform, predicate))

    def unregister_transform(self, node_class, transform, predicate=None):
        """Unregister the given transform."""
        self.transforms[node_class].remove((transform, predicate))

    def visit(self, module):
        """Walk the given astroid *tree* and transform each encountered node

        Only the nodes which have transforms registered will actually
        be replaced or changed.
        """
        module.body = [self._visit(child) for child in module.body]
        return self._transform(module)
