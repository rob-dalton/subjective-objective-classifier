#!/bin/bash python
''' Script to randomly sample sentences from csv of reviews '''

import json
import typing
import re

import pandas as pd
import numpy as np

from typing import List

def split_review_text(text: str)->List[str]:
    # TODO: Move regex pattern to main, compile there, pass here
    sents = re.split('\.|\!|\?', text)
    return [s.strip() for s in sents if s.strip()]

if __name__ == "__main__":

    # load schema (get column names)
    with open('etc/reviews_schema.json') as f:
        schema = json.loads(f.read())

    # load reviews
    df_reviews = pd.read_csv('data/sample_reviews.csv',
                             header=None,
                             names=schema.keys())

    # crete df of sentences
    df_reviews['Sentences'] = df_reviews.reviewText.apply(split_review_text)
    num_sents = [len(el) for el in df_reviews.Sentences]

    df_sents = pd.DataFrame({'review_id': np.repeat(df_reviews.id, num_sents),
                             'sentence': np.hstack(df_reviews.Sentences)})

    # sample sentences
    df_sample = df_sents.sample(n=1000, replace=False, random_state=42)

    # save to csv
    df_sample.to_csv('data/sample_sentences.csv', index=False)
