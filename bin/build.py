#!/usr/bin/env python3
"""soerp build helper.

Wraps the meson-python build so the common operations are one command:

    python bin/build.py install   # build the extension and install it
    python bin/build.py wheel     # build a wheel into dist/
    python bin/build.py clean     # remove build artefacts
"""

import argparse
import logging
import os
import shlex
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from pathlib import Path


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", "%d-%m-%Y %H:%M:%S"
)

file_handler = logging.FileHandler("build.log", "w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.addFilter(lambda record: record.levelno != logging.ERROR)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.ERROR)
stderr_handler.setFormatter(formatter)
logger.addHandler(stderr_handler)

ARTEFACT_DIRS = frozenset({
    "dist",
    "build",
    "lib",
    ".pytest_cache",
    ".ruff_cache",
})


def run_command(command: str, cwd: str | None = None) -> None:
    """Run ``command``, streaming its output to the log.

    Exits the interpreter with the command's status if it fails.
    """
    if cwd is None:
        logger.warning("No working directory specified. Using current one.")
        cwd = Path.cwd()
    else:
        cwd = Path(cwd)

    log_file_path = cwd / "build.log"
    logger.info(f"Executing command: '{command}' in '{cwd}'")

    with subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
        shlex.split(command),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env=dict(**os.environ, PYTHONUNBUFFERED="1"),
        text=True,
    ) as proc:
        with open(log_file_path, "a", encoding="utf-8") as _log_file:
            for line in proc.stdout:
                logger.debug(line.rstrip())
        rv = proc.wait()

    if rv != 0:
        logger.error(f"Command exited with status {rv}")
        sys.exit(rv)

    logger.info("Command executed successfully.")


def install() -> None:
    """Build the extension module and install the package."""
    run_command("uv pip install . -v")


def wheel() -> None:
    """Build a binary wheel."""
    run_command("uv build --wheel -v")


def clean() -> None:
    """Remove build artefacts left behind by meson and the wheel build."""
    logger.debug("Starting cleanup ...")

    run_command("uv pip uninstall soerp")

    for entry in Path().iterdir():
        if entry.name in ARTEFACT_DIRS:
            logger.info(f"Removing '{entry}'")
            shutil.rmtree(entry)
        if entry.name.startswith(".mesonpy"):
            logger.info(f"Removing '{entry}'")
            shutil.rmtree(entry)
        if entry.name.endswith("egg-info"):
            logger.info(f"Removing '{entry}'")
            shutil.rmtree(entry)
        if entry.suffix == ".log":
            logger.info(f"Removing '{entry}'")
            entry.unlink()

    logger.info("Finished cleanup.")


def main() -> None:
    """Parse arguments and dispatch to the requested build mode."""
    parser = argparse.ArgumentParser(description="soerp Build Script")
    parser.add_argument(
        "mode",
        help="""Build mode:
        'install' -- Build and install the package
        'wheel' -- Build a binary wheel
        'clean' -- Remove build artefacts""",
        type=str,
        choices=["install", "wheel", "clean"],
    )

    args = parser.parse_args()

    if args.mode == "install":
        install()
    if args.mode == "wheel":
        wheel()
    if args.mode == "clean":
        clean()


if __name__ == "__main__":
    main()
