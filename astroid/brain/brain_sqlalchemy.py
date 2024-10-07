# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

from __future__ import annotations

from typing import TYPE_CHECKING

from astroid.brain.util import register_module_extender
from astroid.builder import parse

if TYPE_CHECKING:
    from astroid import nodes
    from astroid.manager import AstroidManager


def _session_transform() -> nodes.Module:
    return parse(
        """
    from sqlalchemy.orm.session import Session

    class sessionmaker:
        def __init__(
            self,
            bind=None,
            class_=Session,
            autoflush=True,
            autocommit=False,
            expire_on_commit=True,
            info=None,
            **kw
        ):
            return

        def __call__(self, **local_kw):
            return Session()

        def configure(self, **new_kw):
            return

        return Session()
    """
    )


def register(manager: AstroidManager) -> None:
    register_module_extender(manager, "sqlalchemy.orm.session", _session_transform)
