""" Access to categories and documents in NLTK reuters corpus using custom categories.
    Each custom category is a set of reuters categories.
    This is used to train a model for financial news.
    Other parts of the reuters corpus divided to categories using a similar count of words.
    This allows:
    1) to use all words from the reuters corpus,
    2) to have a balanced data set.

    How to calculate OTHER_CATS:
    Run the script from a folder one level above(word2vec) as
    $ python -m modules.nltk_reuters_custom
    and copy output categories to the 'OTHER_CATS' values

"""

from nltk.corpus import reuters
import nltk
import logging
from .doc_base import DocBase

MONEY_CATS = ['gold', 'income', 'money-fx', 'money-supply', 'silver', 'trade']
OTHER_CATS = [
    ['acq'],
    ['alum', 'barley', 'bop', 'carcass', 'castor-oil', 'cocoa', 'coconut', 'coconut-oil',
     'coffee', 'copper', 'copra-cake', 'corn', 'cotton', 'cotton-oil', 'cpi', 'cpu', 'crude'],
    ['dfl', 'dlr', 'dmk', 'earn'],
    ['fuel', 'gas', 'gnp', 'grain', 'groundnut', 'groundnut-oil', 'heat', 'hog', 'housing',
     'instal-debt', 'interest'],
    ['ipi', 'iron-steel', 'jet', 'jobs', 'l-cattle', 'lead', 'lei', 'lin-oil', 'livestock',
     'lumber', 'meal-feed', 'naphtha', 'nat-gas', 'nickel', 'nkr', 'nzdlr', 'oat', 'oilseed',
     'orange', 'palladium', 'palm-oil', 'palmkernel', 'pet-chem', 'platinum', 'potato',
     'propane', 'rand', 'rape-oil', 'rapeseed', 'reserves', 'retail', 'rice',
     'rubber', 'rye', 'ship'],
    ['sorghum', 'soy-meal', 'soy-oil', 'soybean', 'strategic-metal', 'sugar', 'sun-meal',
     'sun-oil', 'sunseed', 'tea', 'tin', 'veg-oil', 'wheat', 'wpi', 'yen', 'zinc']
]

# Assign names 'other-0', 'other-1', and so on for categories from OTHER_CATS
CATEGORIES = {'money': MONEY_CATS}
for idx, cat_lst in enumerate(OTHER_CATS):
    cname = 'other-{:d}'.format(idx)
    CATEGORIES[cname] = cat_lst


def make_custom_categories():
    """ create a list of categories that have a similar to MONEY_CATS words count """
    mwords = reuters.words(categories=MONEY_CATS)
    bound = int(len(mwords) * 0.9)
    other_cats = [item for item in reuters.categories() if item not in MONEY_CATS]
    cust_cat_lst, sz_lst = [], []
    wcat, cust_cat = 0, []
    for cat in other_cats:
        cust_cat.append(cat)
        cat_len = len(reuters.words(categories=cat))
        if wcat + cat_len > bound:
            cust_cat_lst.append(cust_cat)
            sz_lst.append((wcat, wcat + cat_len))
            cust_cat = []
            wcat = 0
        else:
            wcat += cat_len
    if cust_cat:
        cust_cat_lst.append(cust_cat)
    sz_lst.append((-1, wcat))
    return cust_cat_lst, sz_lst


class NLTK_Reuters_custom_src(DocBase):
    def __init__(self, _):
        nltk.download('reuters')
        from nltk.corpus import reuters
        self._log = logging.getLogger(self.__class__.__name__)

    @property
    def categories(self):
        """ returns a tuple of categories names """
        return list(CATEGORIES.keys())

    def doc_by_category(self, cat, stype='train'):
        super()._check_set_type(stype)
        reuters_cats = CATEGORIES[cat]
        doc_ids = reuters.fileids(reuters_cats)
        doc_ids_stype = [elem for elem in doc_ids if elem.startswith(stype)]
        doc_set = []
        for doc_id in doc_ids_stype:
            txt = reuters.raw(fileids=doc_id)
            doc_set.append(txt)
        self._log.debug('found %d documents in the category "%s", set type "%s"',
                        len(doc_ids), cat, stype)
        return doc_set

if __name__ == '__main__':
    cust_cat_lst, sz_lst = make_custom_categories()
    print("groups of custom categories:", cust_cat_lst)
    print("size of 'MONEY_CATS':", len(MONEY_CATS))
    print("sizes of custom categories:", sz_lst)