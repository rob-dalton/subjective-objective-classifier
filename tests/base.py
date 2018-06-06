import unittest

import pandas as pd

class BaseClassifierTest(unittest.TestCase):

    def load_test_data(self)->tuple:
        df_test = pd.read_csv('data/test.csv')
        x = df_test['sentence']
        y = df_test['class']
        return (x, y)
