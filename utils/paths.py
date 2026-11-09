from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(path):
    """Resolve relative configuration paths against the project, not the shell."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def image_basename(path):
    """Accept both Windows and POSIX separators in stored image names."""
    return Path(str(path).replace('\\', '/')).name
