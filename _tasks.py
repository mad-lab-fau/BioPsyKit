import platform
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def task_docs():
    """Build the html docs using Sphinx."""
    if platform.system() == "Windows":
        return subprocess.run([HERE / "docs/make.bat", "html"])
    else:
        return subprocess.run(["make", "-C", HERE / "docs", "html"])


def update_version_strings(file_path, new_version):
    # taken from:
    # https://stackoverflow.com/questions/57108712/replace-updated-version-strings-in-files-via-python
    version_regex = re.compile(r"(^_*?version_*?\s*=\s*\")(\d+\.\d+\.\d+-?\S*)\"", re.M)
    with open(file_path, "r+") as f:
        content = f.read()
        f.seek(0)
        f.write(
            re.sub(
                version_regex,
                lambda match: '{}{}"'.format(match.group(1), new_version),
                content,
            )
        )
        f.truncate()


def update_version(version):
    subprocess.run(["poetry", "version", version], shell=False, check=True)
    new_version = (
        subprocess.run(["poetry", "version"], shell=False, check=True, capture_output=True)
        .stdout.decode()
        .strip()
        .split(" ", 1)[1]
    )
    update_version_strings(HERE.joinpath("src/biopsykit/__init__.py"), new_version)


def task_update_version():
    version = sys.argv[1]
    return update_version(version)
