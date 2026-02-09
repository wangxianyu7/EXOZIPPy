from os import path, environ

MODULE_PATH = path.abspath(__file__)
for i in range(3):
    MODULE_PATH = path.dirname(MODULE_PATH)

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
