# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

from __future__ import annotations

import operator as operator_mod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


OP_PRECEDENCE = {
    op: precedence
    for precedence, ops in enumerate(
        [
            ["Lambda"],  # lambda x: x + 1
            ["IfExp"],  # 1 if True else 2
            ["or"],
            ["and"],
            ["not"],
            ["Compare"],  # in, not in, is, is not, <, <=, >, >=, !=, ==
            ["|"],
            ["^"],
            ["&"],
            ["<<", ">>"],
            ["+", "-"],
            ["*", "@", "/", "//", "%"],
            ["UnaryOp"],  # +, -, ~
            ["**"],
            ["Await"],
        ]
    )
    for op in ops
}

UNARY_OPERATORS: dict[str, Callable[[Any], Any]] = {
    "+": operator_mod.pos,
    "-": operator_mod.neg,
    "~": operator_mod.invert,
    "not": operator_mod.not_,
}

BIN_OP_IMPL = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "/": lambda a, b: a / b,
    "//": lambda a, b: a // b,
    "*": lambda a, b: a * b,
    "**": lambda a, b: a**b,
    "%": lambda a, b: a % b,
    "&": lambda a, b: a & b,
    "|": lambda a, b: a | b,
    "^": lambda a, b: a ^ b,
    "<<": lambda a, b: a << b,
    ">>": lambda a, b: a >> b,
    "@": operator_mod.matmul,
}

for _KEY, _IMPL in list(BIN_OP_IMPL.items()):
    BIN_OP_IMPL[_KEY + "="] = _IMPL
