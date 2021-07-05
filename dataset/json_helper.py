import json
from tqdm import tqdm
import os
import codecs


def load_json(f_name):
    with open(f_name, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(f_name, obj, indent=None):
    with open(f_name, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_json_line(f_name):
    obj_arr = []
    with open(f_name, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            obj = json.loads(line.rstrip('\n'))
            obj_arr.append(obj)
    return obj_arr


def load_json_line_part(f_name):
    with open(f_name, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            obj = json.loads(line.rstrip('\n'))
            yield obj


def save_json_line(f_name, obj, indent=None):
    with codecs.open(f_name, 'w', encoding='utf-8') as f:
        for ele in obj:
            f.write(json.dumps(ele, indent=indent, ensure_ascii=False) + os.linesep)
