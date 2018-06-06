import unittest
import pandas as pd
import timeit

from models import MNBDocumentClassifier
from .base import BaseClassifierTest

class MNBDocumentClassifierTest(BaseClassifierTest):

    def test_predict(self):

        x, y = self.load_test_data()
        classifier = MNBDocumentClassifier()

        classifier.fit(x, y)

        classification = classifier.predict(['This is  a test.'])

        self.assertEqual('obj', classification[0])

    def test_fit_time(self):
        times = []
        for i in range(10, 101, 10):
            fit_timer = timeit.Timer(f'fit_classifier({i})',
                                     setup='from tests.naive_bayes_test import fit_classifier')
            times.append([i, fit_timer.timeit(10)])

        print('Fit runtimes:\n')
        for n, t in times:
            print(f'{n} docs:\t{t}')

def fit_classifier(range_end: int)->None:
    import pandas as pd
    from models import MNBDocumentClassifier

    df_test = pd.read_csv('data/test.csv').sample(n=100, replace=False)
    x = df_test['sentence']
    y = df_test['class']

    classifier = MNBDocumentClassifier()
    classifier.fit(x[:range_end], y[:range_end])



if __name__ == "__main__":
    unittest.main()
