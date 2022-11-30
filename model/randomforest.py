from sklearn.ensemble import RandomForestClassifier as RandomForestClf
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from model.classifier import Classifier
from model.utils import cats, cats_merged
from imblearn.pipeline import Pipeline


class RandomForestClassifier(Classifier):
    """
    Classifier implementation with RandomForest.  
    """
    __doc__ = Classifier.__doc__

    def __init__(self, X, y):
        """
        Initialiazes the classifier by setting dataset and classifier objects 

        Parameters
        ----------
        X: ndarray
            preprocessed sentences in tokenized format 
        y: ndarray
            targets i.e. multilabel sentence classification 
        """
        self.X = X
        self.y = y
        self.tfidf = TfidfVectorizer(analyzer=(lambda x: x),
                                     tokenizer=(lambda x: x))

        if y.shape[1] == 4:
            self.cats = cats_merged
        else:
            self.cats = cats

        self.y_by_cat = [[label[i] for label in self.y]
                         for i in range(len(self.cats))]

        self.classifiers = [None] * len(self.cats)
        for i in range(len(self.cats)):
            self.classifiers[i] = (
                Pipeline([('vect', self.tfidf), ('rus', RandomUnderSampler(random_state=42)), ('model', RandomForestClf(criterion="entropy", max_depth=None, max_features="auto",
                                                                                                                        n_estimators=500, random_state=42))]))

    def train(self):
        for i in range(len(self.cats)):
            y = self.y_by_cat[i]
            self.classifiers[i] = self.classifiers[i].fit(self.X, y)

    def test(self):
        for i in range(len(self.cats)):
            clf = self.classifiers[i]
            X, y = self.X, self.y_by_cat[i]
            super().test(X, y, clf, "RandomForest", self.cats[i])

    def predict(self, sentence):
        labels = []
        for i in range(len(self.cats)):
            clf = self.classifiers[i]
            X = clf["vect"].transform([sentence]).toarray()
            prediction = clf["model"].predict(X)
            if prediction[0] == 1:
                labels.append(self.cats[i])
        return labels
