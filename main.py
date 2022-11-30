import sys
from preprocessing.preprocess import Preprocesser
from model.svm import SVMClassifier as SVMClf
from model.randomforest import RandomForestClassifier as RandomForestClf
from distutils.util import strtobool
from model.lstm import LSTMClassifier as LSTMClf
from dill import dump

# Input parameter values
SVM = "svm"
RANDOMFOREST = "randomforest"
LSTM = "lstm"
TFIDF = "tfidf"
EMBEDDING = "embedding"
GLOVE = "glove"

# Path to dataset to be trained for
DATASET = "data/sentences_all_merged_categories.csv"


def _prepare_trainer(model, X, y, lstm_encoding):
    """
    Initializes the trainer with the chosen classifier model.

    PARAMETERS
    ----------
    model: str
        name of the model
    X: ndarray
        preprocessed sentences
    y: ndarray
        target labels for each sentence
    """
    if model == SVM:
        return SVMClf(X, y)
    if model == RANDOMFOREST:
        return RandomForestClf(X, y)
    if model == LSTM and lstm_encoding == TFIDF:
        return LSTMClf(X, y)
    else:
        sys.exit("not implemented")


def _to_bool(s):
    return bool(strtobool(s))


def main():
    """
    Main function for reading command line parameters, initializing the trainer, training, testing and saving the model.
    """
    print("setting parameters...")
    # choose model
    MODEL = sys.argv[1]
    if MODEL not in [SVM, RANDOMFOREST, LSTM]:
        sys.exit("invalid model")

    # preprocessing options
    REMOVE_PUNCT = _to_bool(sys.argv[2])
    LOWER_CASE = _to_bool(sys.argv[3])
    REMOVE_STOP_WORDS = _to_bool(sys.argv[4])
    STEMMING = _to_bool(sys.argv[5])

    # TFID, word embedding, GloVe
    ENCODING = sys.argv[6]
    if ENCODING not in [TFIDF, EMBEDDING, GLOVE]:
        sys.exit("invalid encoding")

    print("preprocessing...")
    # set preprocess parameters
    preprocesser = Preprocesser(data_path=DATASET, remove_punct=REMOVE_PUNCT, lower_case=LOWER_CASE,
                                remove_stop_words=REMOVE_STOP_WORDS, stemming=STEMMING)

    # get preprocessed data
    X, y = preprocesser.get_data_features_labels()

    print("training...")
    # train
    trainer = _prepare_trainer(MODEL, X, y, ENCODING)

    if MODEL != LSTM:
        trainer.test()
    trainer.train()

    print("saving trained model...")
    with open("{}.obj".format(MODEL), 'wb') as f:
        dump(trainer, f)


if __name__ == "__main__":
    main()
