from os import path, environ

# Package root directory (exozippy/)
PACKAGE_DIR = path.dirname(path.abspath(__file__))

# Legacy: MODULE_PATH used to point 3 levels above __file__ (repo root).
# Now it points to the package directory for pip-installed compatibility.
MODULE_PATH = PACKAGE_DIR

# data/ lives at the repo root (one level above the package directory);
# fall back to exozippy/data/ for pip-installed environments.
_repo_data = path.join(path.dirname(PACKAGE_DIR), 'data')
DATA_PATH = _repo_data if path.isdir(_repo_data) else path.join(PACKAGE_DIR, 'data')

MULENS_DATA_PATH = path.join(DATA_PATH, 'mulens')

# NextGen stellar atmosphere models path
# Set via NEXTGENFIN_PATH environment variable, or fallback to default location
NEXTGENFIN_PATH = environ.get(
    'NEXTGENFIN_PATH',
    path.join(path.dirname(__file__), 'sed', 'nextgenfin')
)

from exozippy.mmexofast import gridsearches, mmexofast, ulens, estimate_params, fitters, com_trans
