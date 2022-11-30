from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import fbeta_score, make_scorer
import numpy as np


class Classifier(ABC):
    """
    Abstract base class for all classifier model implementations defining common steps for training and evaluation.
    """

    @abstractmethod
    def __init__(self, X, y):
        """
        Constructor for the classifier. Performs algorithm specific operations (e.g. vectorizing input using TF-IDF) to prepare for training.

        Parameters
        ----------
        X: ndarray
            preprocessed sentences in tokenized format
        y: ndarray
            targets i.e. multilabel sentence classification
        """
        pass

    @abstractmethod
    def train(self):
        """
        Trains classifier on vectorized data.
        """
        pass

    @abstractmethod
    def predict(self, sentence):
        """
        Performs multilabel classification on a single sentence.

        Parameters
        ----------
        sentence: list of str
            Sentence to be labeled in tokenized format

        Returns
        -------
        predictions: dict

        Examples
        --------
        >>> s = "Our results show that both two tools suffer from similar problems such as the inability to find smaller-sized images"

        >>> s = preprocesser.preprocess(s)

        >>> model.predict(s)

        background_motivation: 0
        aim_contribution_research_object: 0
        research_method: 0
        results_findings_summary: 1
        """

    def test(self, X, y, model, classifier_name, category):
        """
        Performs cross validated tests on the dataset and saves results in csv format.
        """
        f2 = make_scorer(fbeta_score, beta=2)
        f0_5 = make_scorer(fbeta_score, beta=0.5)
        scoring = {"accuracy": "accuracy",
                   "precision": "precision",
                   "recall": "recall",
                   "f1":  "f1",
                   "f2": f2,
                   "f0_5": f0_5}

        cv_results = cross_validate(model, X, y, cv=10, scoring=scoring)
        train_time = np.sum(cv_results["fit_time"])
        test_time = np.sum(cv_results["score_time"])
        accuracy_score = np.mean(cv_results["test_accuracy"])
        precision_score = np.mean(cv_results["test_precision"])
        recall_score = np.mean(cv_results["test_recall"])
        f1_score = np.mean(cv_results["test_f1"])
        f2_score = np.mean(cv_results["test_f2"])
        f0_5_score = np.mean(cv_results["test_f0_5"])

        print({"category": category, "train_time": train_time, "test_time": test_time, "accuracy_score": accuracy_score, "precision_score": precision_score,
              "recall_score": recall_score, "f1_score": f1_score, "f2_score": f2_score, "f0_5_score": f0_5_score})

        self.save_results(classifier_name, "TF-IDF",
                          category, accuracy_score, precision_score, recall_score, f1_score, f0_5_score, f2_score, train_time, test_time)

    def train_test_split(self, X, y):
        """
        Centralized train test split logic for all implementing classifier classes using sklearn.model_selection.train_test_split.

        Parameters
        ----------
        X: ndarray
            preprocessed sentences in tokenized format
        y: ndarray
            targets i.e. multilabel sentence classification
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=False, random_state=42)
        return X_train, X_test, y_train, y_test

    def save_results(self, classifier, word_vectors, category, acc, precision, recall, f1, f0_5, f2, train_time, test_time):
        """
        Saves classifier evaluation results in a csv file with given scores and meta information.

        Parameters
        ----------
        classifier: str
            name of the classifier model (SVM, RandomForest, LSTM etc.)
        word_vectors: str
            vectorizing strategy (e.g. TF-IDF)
        category: str
            category i.e. target class the evaluation results are calculated against
        acc: float
            accuracy score for the category
        precision: float
            precision score for the category
        recall: float
            recall score for the category
        f1: float
            f1 score for the category
        f0_5: float
            f0_5 score for the category
        f2: float
            f2 score for the category
        train_time: str
            formatted time duration string recorded during training
        test_time: str
            formatted time duration string recorded during testing
        """
        df = pd.read_csv("evaluation/results.csv", index_col="Index")
        new_row = [classifier, word_vectors, category, acc,
                   precision, recall, f1, f0_5, f2, train_time, test_time]
        df.loc[len(df)] = new_row
        df.to_csv("evaluation/results.csv")
