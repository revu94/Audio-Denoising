import sys
from pathlib import Path


def iter_parents(path):
    if path.is_dir():
        yield path
    while path != path.root:
        path = path.parent
        yield path


def get_path():
    path = Path.cwd()
    for path in iter_parents(Path.cwd()):
        if (path / "models").is_dir():
            str_path = str(path)
            if str_path not in sys.path:
                sys.path.insert(0, str_path)
            break
    else:
        print(
            "Could not find a parent directory which contained the clutter_detector module."
        )
    return path
