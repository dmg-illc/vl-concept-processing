from pathlib import Path
import pickle
import json


def open_pickle(path):
    opened_file = pickle.load(open(path, 'rb'))
    return opened_file

def open_json(path):
    opened_file = json.load(open(path, 'rb'))
    return opened_file

def dict_to_json(dict_to_save, path):
    with open(path, "w") as outfile: 
        json.dump(dict_to_save, outfile)