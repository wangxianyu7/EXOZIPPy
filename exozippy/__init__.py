from os import path, environ

# Package root directory (exozippy/)
PACKAGE_DIR = path.dirname(path.abspath(__file__))

# Legacy: MODULE_PATH used to point 3 levels above __file__ (repo root).
# Now it points to the package directory for pip-installed compatibility.
MODULE_PATH = PACKAGE_DIR

path_1 = path.join(MODULE_PATH, 'data')
if path.isdir(path_1):
    DATA_PATH = path_1
else:
    DATA_PATH = path.join(path.dirname(__file__), 'data')

MULENS_DATA_PATH = path.join(DATA_PATH, 'mulens')

# NextGen stellar atmosphere models path
# Set via NEXTGENFIN_PATH environment variable, or fallback to default location
NEXTGENFIN_PATH = environ.get(
    'NEXTGENFIN_PATH',
    path.join(path.dirname(__file__), 'sed', 'nextgenfin')
)

from exozippy.mmexofast import gridsearches, mmexofast, ulens, estimate_params, fitters, com_trans
