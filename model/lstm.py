import time
from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchmetrics
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from model.classifier import Classifier
from model.utils import cats, cats_merged
from sklearn.metrics import accuracy_score, fbeta_score, precision_recall_fscore_support


class Data(Dataset):
    """
    Data preparation abstraction using torch.utils.data.Dataset
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(pl.LightningModule):
    """
    Pytorch Lightning module defining the LSTM network model. 
    """

    def __init__(self, cats, input_size, hidden_size, n_layers, output_size, bidirectional, dropout):
        super().__init__()
        self.cats = cats
        PROB_THRESHOLD = 0.5

        self.accuracy_sep = torchmetrics.Accuracy(
            num_classes=len(self.cats), average="none", threshold=PROB_THRESHOLD)
        self.precision_sep = torchmetrics.Precision(
            num_classes=len(self.cats), average="none", threshold=PROB_THRESHOLD)
        self.recall_sep = torchmetrics.Recall(
            num_classes=len(self.cats), average="none", threshold=PROB_THRESHOLD)
        self.f1_sep = torchmetrics.F1Score(
            num_classes=len(self.cats), average="none", threshold=PROB_THRESHOLD)
        self.f0_5_sep = torchmetrics.FBetaScore(
            beta=0.5, num_classes=len(self.cats), average="none", threshold=PROB_THRESHOLD)
        self.f2_sep = torchmetrics.FBetaScore(
            beta=2, num_classes=len(self.cats), average="none", threshold=PROB_THRESHOLD)

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.criterion = nn.BCELoss()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)
        self.out = nn.Linear(in_features=hidden_size *
                             n_layers, out_features=output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = input.float()
        lstm_out, _ = self.lstm(input)
        output = self.out(lstm_out[:, -1, :])
        output = self.sigmoid(output)
        return output

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.accuracy_sep.reset()
        x, y = batch
        y_out = self(x)
        loss = self.criterion(y_out, y.float())

        self.accuracy_sep(y_out, y.type(torch.int64))
        self.precision_sep(y_out, y.type(torch.int64))
        self.recall_sep(y_out, y.type(torch.int64))
        self.f1_sep(y_out, y.type(torch.int64))
        self.f0_5_sep(y_out, y.type(torch.int64))
        self.f2_sep(y_out, y.type(torch.int64))

        output = {'loss': loss}
        return output

    def training_epoch_end(self, outs):
        acc = self.accuracy_sep.compute().data
        prec = self.precision_sep.compute().data
        recall = self.recall_sep.compute().data
        f1 = self.f1_sep.compute().data
        f0_5 = self.f0_5_sep.compute().data
        f2 = self.f2_sep.compute().data
        output = {}
        for i in range(len(self.cats)):
            name = "acc__{}".format(self.cats[i])
            output[name] = acc[i]
            name = "precision__{}".format(self.cats[i])
            output[name] = prec[i]
            name = "recall__{}".format(self.cats[i])
            output[name] = recall[i]
            name = "f1__{}".format(self.cats[i])
            output[name] = f1[i]
            name = "f0,5__{}".format(self.cats[i])
            output[name] = f0_5[i]
            name = "f2__{}".format(self.cats[i])
            output[name] = f2[i]
        self.log_dict(output)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.accuracy_sep.reset()
        x, y = batch
        y_out = self(x)
        loss = self.criterion(y_out, y.float())

        self.accuracy_sep(y_out, y.type(torch.int64))
        self.precision_sep(y_out, y.type(torch.int64))
        self.recall_sep(y_out, y.type(torch.int64))
        self.f1_sep(y_out, y.type(torch.int64))
        self.f0_5_sep(y_out, y.type(torch.int64))
        self.f2_sep(y_out, y.type(torch.int64))

        output = {'loss': loss}
        return output

    def test_epoch_end(self, outs):
        acc = self.accuracy_sep.compute().data
        prec = self.precision_sep.compute().data
        recall = self.recall_sep.compute().data
        f1 = self.f1_sep.compute().data
        f0_5 = self.f0_5_sep.compute().data
        f2 = self.f2_sep.compute().data
        output = {}
        for i in range(len(self.cats)):
            name = "acc__{}".format(self.cats[i])
            output[name] = acc[i]
            name = "precision__{}".format(self.cats[i])
            output[name] = prec[i]
            name = "recall__{}".format(self.cats[i])
            output[name] = recall[i]
            name = "f1__{}".format(self.cats[i])
            output[name] = f1[i]
            name = "f0,5__{}".format(self.cats[i])
            output[name] = f0_5[i]
            name = "f2__{}".format(self.cats[i])
            output[name] = f2[i]
        self.log_dict(output)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.005)


class LSTMClassifier(Classifier):
    """
    Classifier implementation with LSTM using Pytorch.
    """
    __doc__ = Classifier.__doc__

    def __init__(self, X, y):
        """
        Performs train_test_split and Tfidf vectorizing. 
        Creates dataloaders and the underlying LSTM network model with Pytorch Lightning. 

        Parameters
        ----------
        X: ndarray
            preprocessed sentences in tokenized format 
        y: ndarray
            targets i.e. multilabel sentence classification 
        """
        self.X_train, self.X_test, self.y_train, self.y_test = super().train_test_split(X, y)
        self.tfidf = TfidfVectorizer(analyzer=(lambda x: x),
                                     tokenizer=(lambda x: x))
        self.X_train = self.tfidf.fit_transform(
            list(self.X_train)).toarray()[:, None, :]
        self.X_test = self.tfidf.transform(
            list(self.X_test)).toarray()[:, None, :]

        if y.shape[1] == 4:
            self.cats = cats_merged
        else:
            self.cats = cats

        self.NUM_LSTM_LAYERS = 2
        self.BIDIRECTIONAL = True
        self.N_EPOCHS = 30
        self.INPUT_SIZE = self.X_train.shape[2]
        self.OUTPUT_SIZE = len(self.cats)
        self.HIDDEN_SIZE = 256
        self.DROPOUT = 0.1

        self.train_dataloader = DataLoader(
            Data(self.X_train, self.y_train), batch_size=64, shuffle=True)
        self.test_dataloader = DataLoader(
            Data(self.X_test, self.y_test), batch_size=64, shuffle=True)

        self.model = LSTMModel(cats=self.cats, input_size=self.INPUT_SIZE, hidden_size=self.HIDDEN_SIZE,
                               n_layers=self.NUM_LSTM_LAYERS, bidirectional=self.BIDIRECTIONAL, dropout=self.DROPOUT, output_size=self.OUTPUT_SIZE)

    def train(self):
        trainer = pl.Trainer(max_epochs=self.N_EPOCHS,
                             log_every_n_steps=5, check_val_every_n_epoch=1)

        start_train = time.time()
        trainer.fit(model=self.model, train_dataloaders=self.train_dataloader)
        end_train = time.time()
        train_time = timedelta(seconds=end_train - start_train)

        # EVALUATION
        start_test = time.time()

        self.model.eval()
        X = self.X_test
        X = torch.from_numpy(X)
        y = self.y_test
        predictions = self.model(X)
        predictions = torch.round(predictions)
        predictions = predictions.detach().numpy()

        end_test = time.time()
        test_time = timedelta(seconds=end_test - start_test)

        y_predict = [[label[i] for label in predictions]
                     for i in range(len(self.cats))]
        y_test = [[label[i] for label in y]
                  for i in range(len(self.cats))]

        for i in range(len(self.cats)):
            acc = accuracy_score(y_test[i], y_predict[i])
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test[i], y_predict[i], beta=1, average="binary", pos_label=1)
            f0_5 = fbeta_score(y_test[i], y_predict[i], beta=0.5,
                               average="binary", pos_label=1)
            f2 = fbeta_score(y_test[i], y_predict[i], beta=2,
                             average="binary", pos_label=1)

            print("accuracy for {}: {}".format(self.cats[i], acc))
            print("precision for {}: {}".format(self.cats[i], precision))
            print("recall for {}: {}".format(self.cats[i], recall))
            print("f1 for {}: {}".format(self.cats[i], f1))
            print("f0.5 for {}: {}".format(self.cats[i], f0_5))
            print("f2 for {}: {}".format(self.cats[i], f2))
            print("train time: {}".format(train_time))
            print("test time: {}".format(test_time))

            super().save_results("LSTM", "TF-IDF",
                                 self.cats[i], acc, precision, recall, f1, f0_5, f2, train_time, test_time)

    def predict(self, sentence):
        X = self.tfidf.transform([sentence]).toarray()
        X = X[:, None, :]
        X = torch.from_numpy(X)
        self.model.eval()
        prediction = self.model(X)[0]
        labels = []
        for i in range(len(self.cats)):
            if prediction[i] >= 0.5:
                labels.append(self.cats[i])
        return labels
