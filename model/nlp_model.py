import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def get_model():
    df = pd.read_csv('Cleaned_Dataset.tsv', sep='\t')

    X, y = df['text'], df['sentiment']

    pipeline = Pipeline([
        ('bow', CountVectorizer()),  # Transforming text documents to counts.
        ('tf-idf', TfidfTransformer()),  # Weighing counts through TF-IDF.
        ('classifier', MultinomialNB())
    ])

    pipeline.fit(X, y)

    return pipeline
