""" Base class for documents """

from abc import ABCMeta, abstractmethod


class DocBase(metaclass=ABCMeta):
    """ The interface to Document """
    @property
    @abstractmethod
    def categories(self):
        pass

    @abstractmethod
    def categories(self):
        """ returns a tuple of categories names """
        self._read_db()
        return self._cats

    @abstractmethod
    def doc_by_category(self, category, ctype='train'):
        """ Get a list of documents for the category for 'train' or 'test' """
        pass

    def _check_set_type(self, stype):
        STYPES = ('test', 'train')
        if stype not in STYPES:
            raise ValueError('set type must be one of "{}". Got "{}"'.format(STYPES, stype))
