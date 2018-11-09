""" configuration from JSON file.
    data priorities:
    1) JSON
    2) defaults
"""

import json

class Config(object):
    def __init__(self, dvals, req_keys=None):
        self.__init_done = False
        if not isinstance(dvals, dict):
            raise ValueError("expecting 'dvals' as 'dict'. Got '{}'".format(type(dvals)))
        self._dict = dvals
        if req_keys:
            dvkeys = set(self._dict)
            rqkeys = set(req_keys)
            if dvkeys != rqkeys:
                raise ValueError("Invalid configuration key(s). Allowed keys: '{}'".format(req_keys))
        self.__init_done = True

    def __setattr__(self, key, val):
        if key == '_Config__init_done':
            super().__setattr__(key, val)
            return
        if self.__init_done:
            if key not in self._dict:
                raise AttributeError("no such attribute '{}'".format(key))
            raise RuntimeError("Could not set value for readonly attribute '{}'".format(key))
        super().__setattr__(key, val)

    def __getattr__(self, item):
        if item in self._dict:
            return self._dict[item]
        raise AttributeError("no such attribute '{}'".format(item))

    @staticmethod
    def from_json(jfname, req_keys=None):
        with open(jfname, 'r') as jfile:
            dvals = json.load(jfile)
        conf = Config(dvals, req_keys)
        return conf


if __name__ == '__main__':
    cnf = Config({'a': 10, 'b': 12})
    print(cnf.a, cnf.b)
    try:
        print(cnf.c)
    except Exception as err:
        print(err)
    try:
        cnf.c = 11
    except Exception as err:
        print(err)
    try:
        cnf.a = 101
    except Exception as err:
        print(err)
    try:
        cnf = Config({'a': 10, 'b': 12}, ('a', 'c'))
    except Exception as err:
        print(err)

