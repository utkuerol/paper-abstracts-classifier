import PySimpleGUI as sg
from preprocessing.preprocess import Preprocesser
from dill import load
from nltk.tokenize import sent_tokenize


def predict(classifier, sentence):
    preprocesser = Preprocesser(data_path="", remove_punct=True, lower_case=True,
                                remove_stop_words=False, stemming=False)

    with open("pretrained/{}.obj".format(classifier), 'rb') as f:
        model = load(f)
        s = preprocesser.preprocess(sentence)
        prediction = model.predict(s)
        return prediction


def label_abstract(classifier, abstract):
    sentences = sent_tokenize(abstract)
    new_sentences = ""
    for s in sentences:
        labels = predict(classifier, s)
        new_sentences = new_sentences + "{}\n {}\n\n".format(labels, s)
    return new_sentences


font = "arial 16 normal"

# Define the window's contents
layout = [[sg.Text("Enter abstract", font=font)],
          [sg.Multiline(key="input", font=font, size=(None, 20))],
          [sg.Text("Enter classifier (svm, svm-merged, randomforest, randomforest-merged, lstm, lstm-merged)", font=font)],
          [sg.Input(key="classifier", font=font)],
          [sg.Button('Ok'), sg.Button('Quit')]]

# Create the window
window = sg.Window('Paper Abstracts Sentence Classification', layout)

# Display and interact with the Window using an Event Loop
while True:
    event, values = window.read()
    # See if user wants to quit or window was closed
    if event == sg.WINDOW_CLOSED or event == 'Quit':
        break
    # Output a message to the window
    window["input"].update(label_abstract(values["classifier"],
                                          values["input"]))

# Finish up by removing from the screen
window.close()
