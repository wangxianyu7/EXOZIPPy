import re

def printandlog(msg, logname):
    print(msg)
    if logname:
        with open(logname, 'a') as logf:
            logf.write(msg + '\n')

# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
def parse_param_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    params = {}

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue  # skip comments and blank lines

        # Remove inline comments and split on whitespace
        parts = re.split(r'\s+', line.split("#")[0].strip())
        if not parts:
            continue

        label = parts[0]
        try:
            values = list(map(float, parts[1:]))
        except ValueError:
            # Handle lines with malformed numbers
            continue

        # Assign values safely
        mu      = values[0] if len(values) > 0 else None
        sigma   = values[1] if len(values) > 1 else None
        lower   = values[2] if len(values) > 2 else None
        upper   = values[3] if len(values) > 3 else None
        initval = values[4] if len(values) > 4 else None

        params[label] = {
            "mu": mu,
            "sigma": sigma,
            "lower": lower,
            "upper": upper,
            "initval": initval
        }

    return params

