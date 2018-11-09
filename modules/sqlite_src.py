""" source of tokens from SQL text. See ... for example """

import logging
import pickle
import os
import sqlite3
from .doc_base import DocBase

class SQLite_src(DocBase):
    """ The source of texts by topics. A text in a database includes ID and raw text.
        All other attributes are optional.
        The DB is asked about categories only once on the first request.
        Then it reads documents for a category from DB
    """
    def __init__(self, db_fname):
        self._log = logging.getLogger(self.__class__.__name__)
        self._db_fname = db_fname
        self._curs = None
        self._conn = None
        self._cats = None
        self._once_warn = False

    @property
    def categories(self):
        """ returns a tuple of categories names """
        self._read_db()
        return self._cats

    def _db_connect(self):
        if self._curs:
            return
        if not os.path.exists(self._db_fname):
            raise ValueError("database file {} doesn't exists".format(self._db_fname))
        self._conn = sqlite3.connect(self._db_fname)
        self._curs = self._conn.cursor()

    def doc_by_category(self, cat, stype='train'):
        """ returns a set of documents for a category """
        super()._check_set_type(stype)
        self._log.debug("get documents for category '%s/%s'", cat, stype)
        if not self._once_warn:
            self._log.warning("This class doesn't use 'stype'")
            self._once_warn = True
        self._read_db()
        if cat not in self._cats:
            raise ValueError("category '{}' not in DB categories '{}'".format(cat, self._cats))
        ret = []
        res = self._curs.execute("select body from articles where category = '{}'".format(cat))
        while True:
            rows = res.fetchmany(200)
            if not rows:
                self._log.debug("%d documents received for category '%s/%s'", len(ret), cat, stype)
                break
            for rlst in rows:
                ret.append(rlst[0])
        return ret

    def _read_db(self):
        if self._cats:
            return
        self._db_connect()
        res = self._curs.execute("select category, count(id) from articles group by category")
        cat = {key:val for key, val in res.fetchall()}
        data_sz = sum(list(cat.values()))
        self._log.info('data size: %d', data_sz)
        self._log.info('number of categories: %d', len(cat))
        self._log.info('categories: %s', cat)
        self._log.info("The 'Real Estate' category has a small number items and removed")
        self._log.info('number of categories: %d', len(cat)-1)
        self._cats = [elem for elem in cat if elem != 'Real Estate']

    def __del__(self):
        if self._conn:
            self._conn.close()
