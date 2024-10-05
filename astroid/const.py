# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

import sys

PY310_PLUS = sys.version_info >= (3, 10)
PY311_PLUS = sys.version_info >= (3, 11)
PY312_PLUS = sys.version_info >= (3, 12)
PY313_PLUS = sys.version_info >= (3, 13)

IS_PYPY = sys.implementation.name == "pypy"
IS_JYTHON = sys.implementation.name == "jython"
