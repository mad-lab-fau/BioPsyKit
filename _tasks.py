import platform
import re
import subprocess
import sys
from typing import Sequence
from pathlib import Path

HERE = Path(__file__).parent


def task_docs():
    """Build the html docs using Sphinx."""
    if platform.system() == "Windows":
        return subprocess.run([HERE / "docs/make.bat", "html"], check=False)
    return subprocess.run(["make", "-C", HERE / "docs", "html"], check=False)


def update_version_strings(file_path, new_version):
    # taken from:
    # https://stackoverflow.com/questions/57108712/replace-updated-version-strings-in-files-via-python
    version_regex = re.compile(r"(^_*?version_*?\s*=\s*\")(\d+\.\d+\.\d+-?\S*)\"", re.M)
    with file_path.open("r+") as f:
        content = f.read()
        f.seek(0)
        f.write(
            re.sub(
                version_regex,
                lambda match: f'{match.group(1)}{new_version}"',
                content,
            )
        )
        f.truncate()


def update_version(version: Sequence[str]):
    if len(version) == 0:
        # no argument passed => return the current version
        subprocess.run(["uv", "version"], shell=False, check=True, capture_output=False)
    else:
        # update the version
        subprocess.run(["uv", "version", *version], shell=False, check=True)
        new_version = (
            subprocess.run(["uv", "version"], shell=False, check=True, capture_output=True)
            .stdout.decode()
            .strip()
            .split(" ", 1)[1:][0]
        )

        update_version_strings(HERE.joinpath("src/biopsykit/__init__.py"), new_version)


def task_update_version():
    version_arr = sys.argv[1:] if len(sys.argv) > 1 else []
    update_version(version_arr)
