import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from models import MNBDocumentClassifier


if __name__ == "__main__":

    # setup x_lab, y_lab
    x_lab = 'sentence'
    y_lab = 'class'

    # load training data
    with open('data/rotten_imdb/plot.tok.gt9.5000', encoding="ISO-8859-1") as csv:
        df_subj = pd.read_csv(csv, delimiter='\n', header=None,
                              names=['sentence'])
        df_subj['class'] = 'subj'

    with open('data/rotten_imdb/quote.tok.gt9.5000', encoding="ISO-8859-1") as csv:
        df_obj = pd.read_csv(csv, delimiter='\n', header=None,
                             names=['sentence'])
        df_obj['class'] = 'obj'

    df_train = df_subj.append(df_obj)

    # create model
    model = MNBDocumentClassifier()

    # TODO: Add grid search for model parameter values
    # TODO: Add option to pass parameter values to MNBDocumentClassifier

    # get cross validated error of model
    kf = KFold(n_splits=10)
    scores = []
    for train_i, test_i in kf.split(df_train):
        # get split
        train = df_train.iloc[train_i]
        test = df_train.iloc[test_i]

        # fit model
        model.fit(train[x_lab], train[y_lab])

        # get estimated accuracy score
        y_pred = model.predict(test[x_lab])
        score = accuracy_score(test[y_lab], y_pred)
        scores.append(score)

    # print accuracy metrics
    print(f'K-Fold CV scores:\n')
    for i, score in enumerate(scores):
        print(f'{i}:\t{round(score, 3)}')

    print(f'\nMean score:\t{sum(scores) / len(scores)}')
