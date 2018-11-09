#!/usr/bin/env python
''' Create, train, and save the LDA models from a financial corpus '''

import logging
import os
import argparse
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
from modules.config import Config
from modules.nltk_reuters_src import NLTK_Reuters_src
from modules.sqlite_src import SQLite_src
from modules.nltk_reuters_custom import NLTK_Reuters_custom_src
import numpy as np

_CONF_TYPES = {'sqlite': SQLite_src,
               'nltk': NLTK_Reuters_src,
               'nltk_custom': NLTK_Reuters_custom_src
}
VECTORIZER_TYPES = ("count", "tf", "tfidf")


class LDA_model:
    CONF_KEYS = ("model_file", "vectorizer_file", "token_file", "data_src", "data_type",
                 "categories", "data_subset", "vectorizer_type", "train_iter")
    def __init__(self, conf_fname):
        self._log = logging.getLogger(self.__class__.__name__)
        self._categories = None
        self._ldamodel = None
        self._token_set = None

        self._conf = Config.from_json(conf_fname, LDA_model.CONF_KEYS)
        if self._conf.data_type not in _CONF_TYPES:
            raise ValueError("configuration.data_type: '{}' not in '{}'"
                             .format(self._conf.data_type, _CONF_TYPES.keys()))
        self._data_set = _CONF_TYPES[self._conf.data_type](self._conf.data_src)
        if self._conf.vectorizer_type not in VECTORIZER_TYPES:
            raise ValueError("configuration.vectorizer_type: {} not in '{}'"
                             .format(self.conf.vectorizer_type, VECTORIZER_TYPES))
        if self._conf.vectorizer_type == 'count':
            self._vectorizer = CountVectorizer(min_df=2, max_df=1.0, stop_words='english',
                                               lowercase=True,
                                               token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
        else:
            use_idf = self._conf.vectorizer_type == 'tfidf'
            self._vectorizer = TfidfVectorizer(min_df=2, max_df=1.0, stop_words='english',
                                               use_idf=use_idf, lowercase=True,
                                               token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
        self._max_iter = int(self._conf.train_iter)

    def exists(self):
        ''' Return True if both Word2Vec dictionary and the model exist '''
        _exists = os.path.exists(self._conf.model_file)
        self._log.debug("model '%s' exists: %s", self._conf.model_file, _exists)
        return _exists

    def prepare_tokens(self):
        # Create tokens from text
        self._log.info("create tokens from texts")
        if self._conf.categories != 'all':
            if not isinstance(self._conf.categories, (list, tuple)):
                raise ValueError("configuration.categories expected as list or tuple. Got '{}'"
                                 .format(type(self._conf.categories)))
            if not set(self._conf.categories).issubset(self._data_set.categories):
                err = ("Required categories '{}' are not subset of data categories '{}'"
                       .format(self._conf.categories, self._data_set.categories))
                raise ValueError(err)
            self._categories = self._conf.categories
        else:
            self._categories = self._data_set.categories
        # get a list of raw documents
        docs = []
        for cat in self._categories:
            docs += self._data_set.doc_by_category(cat, self._conf.data_subset)
        # transform to tokens
        self._token_set = self._vectorizer.fit_transform(docs)
        self._log.info("transformation finished")

    def create(self):
        ''' Create if not exists and train the model '''
        # create dictionary from tokens
        NUM_TOPICS = len(self._categories)
        # Build a Latent Dirichlet Allocation Model
        self._lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS,
                                                    max_iter=self._max_iter,
                                                    evaluate_every=10,
                                                    learning_method='online')

        self._log.info("training the model")
        self._lda_model.fit(self._token_set)
        self._log.debug("training finished")
        with open(self._conf.model_file, 'wb') as mwfile:
            pickle.dump(self._lda_model, mwfile)
        with open(self._conf.vectorizer_file, 'wb') as mwfile:
            pickle.dump(self._vectorizer, mwfile)
        with open(self._conf.token_file, 'wb') as mwfile:
            pickle.dump(self._token_set, mwfile)
        self._log.info("model has been saved as '%s'", self._conf.model_file)
        self._log.info("vectorizer has been saved as '%s'", self._conf.vectorizer_file)
        self._log.info("tokens has been saved as '%s'", self._conf.token_file)

        tf_feature_names = self._vectorizer.get_feature_names()
        self._log.debug("the first 20 feature names: %s", tf_feature_names[:20])
        n_top_words = 15
        self._print_top_words(tf_feature_names, n_top_words)

    def _print_top_words(self, feature_names, n_top_words):
        self._log.info("topics in LDA model:")
        for topic_idx, topic in enumerate(self._lda_model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            self._log.info(message)

    def test(self):
        text = ["The economy is working better than ever", "I earn some money", "how are you?"]
        x = self._lda_model.transform(self._vectorizer.transform(text))[0]
        self._log.info("test result: %s, %s", x, x.sum())
        idx = np.matrix(x).argmax()
        self._log.info("maximum: idx: %d, val: %f", idx, x[idx])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
            help="be verbose")
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='overwrite existed model, tokens, and vectorizer')
    parser.add_argument('jfile', nargs=1, help="json configuration")
    args = parser.parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s %(name)-15s %(message)s')
    lda_model = LDA_model(args.jfile[0])
    if lda_model.exists() and not args.force:
        logging.info("Nothing todo - model exist")
    else:
        # lda_model.read_text()
        lda_model.prepare_tokens()
        lda_model.create()
        lda_model.test()
