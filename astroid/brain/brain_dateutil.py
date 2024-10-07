# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

"""Astroid hooks for dateutil."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from astroid.brain.util import register_module_extender
from astroid.builder import AstroidBuilder
from astroid.manager import AstroidManager

if TYPE_CHECKING:
    from astroid import nodes


def dateutil_transform() -> nodes.Module:
    return AstroidBuilder(AstroidManager()).string_build(
        textwrap.dedent(
            """
    import datetime
    def parse(timestr, parserinfo=None, **kwargs):
        return datetime.datetime()
    """
        )
    )


def register(manager: AstroidManager) -> None:
    register_module_extender(manager, "dateutil.parser", dateutil_transform)
