from pathlib import Path
from typing import Sequence, Optional, Union


def mkdirs(dir_list: Union['path_t', Sequence['path_t']]) -> None:
    if isinstance(dir_list, (str, Path)):
        dir_list = [dir_list]
    for directory in dir_list:
        directory.mkdir(exist_ok=True, parents=True)


def get_subject_dirs(base_path: 'path_t', pattern: str):
    return [p for p in sorted(base_path.glob(pattern)) if p.is_dir()]
