# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import json
import os


def splitData(data: str, maxsplit: int, index: int) -> dict:
    dict_str = data.split(' ', maxsplit)[index].strip()
    return ast.literal_eval(dict_str)


def load_settings(JsonDirectory):

    file_path = os.path.join(JsonDirectory, 'settings.json')

    try:
        with open(file_path, 'r') as json_file:
            settings = json.load(json_file)
            return settings
    except FileNotFoundError:
        return "FileError"
    except json.JSONDecodeError:
        return "JsonFormatError"
