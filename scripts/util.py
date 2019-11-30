import json
from pathlib import Path

def hdf_attr_check(attr, hdf_attr_dict, default):
    out = default if attr not in hdf_attr_dict else hdf_attr_dict[attr]
    return out

def filter_files(files, include=[], exclude=[]):
    for incl in include:
        files = [f for f in files if incl in f.name]
    for excl in exclude:
        files = [f for f in files if excl not in f.name]
    return sorted(files)

def dict2json(data:dict, file):
    with open(file, 'w') as f:
        f.write(json.dumps(data))

def ls(x, recursive=False, include=[], exclude=[]):
    if not recursive:
        out = list(x.iterdir())
    else:
        out = [o for o in x.glob('**/*')]
    out = filter_files(out, include=include, exclude=exclude)
    return out

Path.ls = ls