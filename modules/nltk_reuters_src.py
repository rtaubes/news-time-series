""" Access to categories and documents in NLTK corpora.
    See https://www.nltk.org/book/ch02.html for details.
    Add required classes using NLTK as template for other corpuses if needed.
"""

from nltk.corpus import reuters
import nltk
import logging
from .doc_base import DocBase


class NLTK_Reuters_src(DocBase):
    def __init__(self, _):
        nltk.download('reuters')
        from nltk.corpus import reuters
        self._log = logging.getLogger(self.__class__.__name__)

    @property
    def categories(self):
        """ returns a tuple of categories names """
        return reuters.categories()

    def doc_by_category(self, cat, stype='train'):
        super()._check_set_type(stype)
        doc_ids = reuters.fileids(cat)
        doc_ids_stype = [elem for elem in doc_ids if elem.startswith(stype)]
        doc_set = []
        for doc_id in doc_ids_stype:
            txt = reuters.raw(fileids=doc_id)
            doc_set.append(txt)
        self._log.debug('found %d documents in the category "%s", set type "%s"',
                        len(doc_ids), cat, stype)
        return doc_set
