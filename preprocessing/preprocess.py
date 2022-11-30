import pandas as pd
import numpy as np

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')


class Preprocesser:
    """
    Preprocesser for the paper abstracts dataset. Set parameters to customize preprocessing operations. 

    PARAMETERS
    ----------
    data_path: str
    remove_punt: bool, optional
    lower_case: bool, optional
    remove_stop_words: bool, optional
    stemming: bool optional

    """

    def __init__(self, data_path, remove_punct=True, lower_case=True, remove_stop_words=False, stemming=False):
        self.data_path = data_path
        self.remove_punct = remove_punct
        self.lower_case = lower_case
        self.remove_stop_words = remove_stop_words
        self.stemming = stemming

    def _get_data(self):
        """
        Reads data from the csv file and loads into X/y splitted ndarrays.
        """
        data = pd.read_csv(self.data_path, index_col=0)
        X = data.iloc[:, 0].to_numpy()
        y = data.iloc[:, 1:].to_numpy().astype(int)
        mask = (y == 0).all(axis=1)
        mask = np.invert(mask)
        y = y[mask, :]
        X = X[mask]
        return X, y

    def preprocess(self, sen):
        """
        Preprocessing pipeline with chosen operations for a single sentence. 

        PARAMETERS
        ----------
        sen: str
            sentence to be preprocessed in raw format
        """
        # lowercase
        if self.lower_case:
            sen = sen.lower()

        # remove punctuation
        if self.remove_punct:
            sen = "".join(
                [char for char in sen if char not in string.punctuation])

        # tokenization
        sen = word_tokenize(sen)

        # remove stop words
        if self.remove_stop_words:
            stop_words = stopwords.words('english')
            sen = [word for word in sen if word not in stop_words]

        # stemming
        if self.stemming:
            porter = PorterStemmer()
            sen = [porter.stem(word) for word in sen]

        return sen

    def get_data_features_labels(self):
        """
        Gets preprocessed data splitted in X/y ndarrays ready to be fed into classifiers. 
        """
        X, y = self._get_data()

        X_s = pd.Series(X)
        X_s = X_s.apply(lambda s: self.preprocess(s))
        X = X_s.to_numpy()

        return X, y
