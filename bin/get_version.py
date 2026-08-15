#!/usr/bin/env python3
"""Print the project version for meson's ``project(version: ...)``.

meson runs this at configure time, before build dependencies are
necessarily importable, so setuptools_scm is installed on demand.
"""

import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from pathlib import Path


try:
    from setuptools_scm import get_version
except ModuleNotFoundError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "setuptools_scm"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    from setuptools_scm import get_version


def main() -> None:
    """Print the version derived from the git history."""
    root = Path(__file__).parent.parent
    try:
        version = get_version(root=root)
    except LookupError:
        # No git metadata, e.g. building from an unpacked sdist.
        version = None
    print(version or "0.0.0")


if __name__ == "__main__":
    main()
