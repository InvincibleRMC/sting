#! /usr/bin/env python3

"""useful wrapper around shell commands to generate and deploy docs to github pages
"""
import os
from pathlib import Path
from typing import List

import typer

app = typer.Typer()


def run_cmd(cmd: str) -> int:
    """wrapper to call os.system with the specified command

    Args:
        cmd (str): the command to run in a bash shell

    Returns:
        int: the exit code of the command
    """
    typer.secho(f"> {cmd}", fg="blue")
    return os.system(cmd)


@app.command()
def build(
    out_dir: Path = Path("docs"),
    modules: List[Path] = [Path("sting")],
    force: bool = True,
):
    """wrapper around pdoc3 documentation generation"""
    run_cmd(
        f"python3 -m pdoc --html -o {out_dir} {' '.join(map(str, modules))} {'--force' if force else ''}"
    )


@app.command()
def deploy(
    out_dir: Path = Path("docs"),
    module: str = "sting",
    branch: str = "gh-pages",
    remote: str = "origin",
):
    """wrapper around git-subtree"""
    run_cmd(f"git subtree push --prefix {out_dir / module} {remote} {branch}")


if __name__ == "__main__":
    app()
