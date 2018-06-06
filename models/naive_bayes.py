import logging
import re
import string
import typing
import spacy

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from typing import List

from etc.types import SKLearnPipeline, DataFrame, Series

class MNBDocumentClassifier():

    _nlp = spacy.load('en')

    def __init__ (self, *nargs, **kwargs):
        self._model = self._build_pipeline()

    @classmethod
    def _build_pipeline(cls, fit_prior: bool = True)->SKLearnPipeline:
        """ Create MultinomialNB model pipeline using sklearn. """
        return Pipeline([('vect', CountVectorizer(analyzer='word',
                                                  stop_words='english',
                                                  tokenizer=cls._tokenize_doc,
                                                  ngram_range=(1, 2))),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB(alpha=.01, fit_prior=fit_prior)),
                        ])
    @classmethod
    def _tokenize_doc(cls, doc: str)->List[str]:
        """ Take in document, remove punctuation and tokenize. """
        tokens = []
        try:
            all_tokens = (w.text for w in cls._nlp(doc))
            tokens = [token for token in all_tokens if token not in string.punctuation]
        except:
            logging.warn("Couldn't tokenize description: %s", repr(doc),
                         exc_info=True)

        return tokens

    def fit(self, X: Series = None, y: Series = None):
        """ Take in X of documents, y of labels. Fit model. """
        self._model = self._model.fit(X, y)

    def predict(self, docs: Series = None)->str:
        """ Classify documents """
        classifications = self._model.predict(docs)
        return classifications
