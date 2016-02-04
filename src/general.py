import os
import hashlib
import json

def check_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def get_parameter_hash(params):
    md5 = hashlib.md5()
    md5.update(str(json.dumps(params, sort_keys=True)))
    return md5.hexdigest()

