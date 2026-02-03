import re
import json
# def read_parfile(filename):
#     with open(filename) as f:
#         contents = f.read()
#     return json.loads(contents)
def read_par(filepath):
    text = open(filepath, 'r').read()
    params = {}
    lines = text.strip().split("\n")
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue  # skip comments and blank lines

        parts = re.split(r'\s+', line.split("#")[0].strip())
        label = parts[0]
        values = list(map(float, parts[1:]))

        # Assign values based on number of columns
        mu = values[0] if len(values) > 0 else None
        sigma = values[1] if len(values) > 1 else None
        lower = values[2] if len(values) > 2 else None
        upper = values[3] if len(values) > 3 else None
        initval = values[4] if len(values) > 4 else None

        params[label] = {
            "mu": mu,
            "sigma": sigma,
            "lower": lower,
            "upper": upper,
            "initval": initval
        }

    return params
