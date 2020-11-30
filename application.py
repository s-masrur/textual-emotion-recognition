from tkinter import *
import tkinter.font as tkfont
import nltk
from keras.models import load_model
import re
from keras_preprocessing.sequence import pad_sequences
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import preprocessor as p
import pickle
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter("ignore", UserWarning)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('wordnet_ic')
nltk.download('punkt')

stopwords = stopwords.words('english')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

not_stopwords = {'couldnt', 'not', 'didn', 'didnt', 'doesn', 'doesnt', 'hadn', 'hadnt', 'hasn', 'hasnt', 'haven',
                 'havent', 'isn', 'isnt', 'ma', 'mightn', 'mightnt', 'mustn', 'mustnt', 'needn', 'neednt',
                 'shan', 'shant', 'shouldn', 'shouldnt', 'wasn', 'wasnt', 'weren', 'werent', 'won', 'wont',
                 'wouldn', 'wouldnt'}
stop_words = set([word for word in stopwords if word not in not_stopwords])


def preproc(text):
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION)

    text = p.clean(text)                                            # Remove URLs, emojis, and usernames
    text = re.sub('[^A-Za-z0-9 ]+', '', text)                       # Removing special characters
    text = word_tokenize(text)                                      # Tokenization
    text = [word.lower() for word in text]                          # Normalizing case (lower case)
    text = [w for w in text if w not in stop_words]                 # Removing stopwords
    text = [i for i in text if len(i) > 1]                          # Removing single characters
    words_lemmed = [lemmatizer.lemmatize(word) for word in text]    # Lemmatization
    text = words_lemmed

    return text


# loading models:
model_2 = load_model('models/LSTM_2.h5')
model_3 = load_model('models/LSTM_3.h5')
model_6 = load_model('models/LSTM_6.h5')

# loading tokenizers:
with open('models/tokenizer_2', 'rb') as handle:
    tokenizer_2 = pickle.load(handle)

with open('models/tokenizer_3', 'rb') as handle:
    tokenizer_3 = pickle.load(handle)

with open('models/tokenizer_6', 'rb') as handle:
    tokenizer_6 = pickle.load(handle)


class Window:
    def __init__(self):
        self.root = Tk()
        self.root.title("Emotion Recognition")
        self.root.geometry('398x450')
        self.root.iconbitmap('icon.ico')  # the icon retrieved from the Noun Project by Vectors Market:
        self.root.resizable(False, False)  # https://thenounproject.com/term/machine-learning/1927709/
        # text label:
        self.enterLbl = Label(self.root, text="Please enter your text: ")
        self.enterLbl.grid(row=0, column=0)
        # textbox:
        self.textbox = Text(self.root, width=49, height=10)
        self.textbox.grid(row=1, column=0)
        # exit button:
        self.extBtn = Button(self.root, width=6, text="Exit", command=self.close_window)
        self.extBtn.place(x=340, y=420)
        # text label
        self.chooseLbl = Label(self.root, text="Choose a classifier: ")
        self.chooseLbl.place(x=12, y=210)
        # radio buttons:
        self.state = IntVar()
        self.binary = Radiobutton(self.root, text="Binary classifier (positive, negative)",
                                  variable=self.state, value=1)
        self.binary.place(x=7, y=240)

        self.ternary = Radiobutton(self.root, text="Ternary classifier (positive, negative, neutral)",
                                   variable=self.state, value=2)
        self.ternary.place(x=7, y=260)

        self.sixEmotion = Radiobutton(self.root, variable=self.state, value=3,
                                      text="Six emotion classifier (neutral, worry, happiness, sadness, love, fun)")
        self.sixEmotion.place(x=7, y=280)
        # text label:
        self.prediction = Label(self.root, text="Prediction:")
        self.prediction.place(x=12, y=340)
        # output text label:
        fontStyle = tkfont.Font(size=15)
        self.output = Label(self.root, font=fontStyle)
        self.output.place(x=22, y=375)
        # predict button:
        self.predictBtn = Button(self.root, width=9, text="Predict", command=self.predict)
        self.predictBtn.place(x=265, y=420)

        self.root.mainloop()

    def close_window(self):  # exit function for exit button
        self.root.destroy()

    def predict(self):  # predict function
        sentence = self.textbox.get("1.0", 'end-1c')  # get ...
        inp = preproc(sentence)  # ... and preprocess the input

        chosenClassifier = self.state.get()

        if chosenClassifier == 1:   # If chosen BINARY CLASSIFIER
            # integer encoding the input:
            x_encoded = tokenizer_2.texts_to_sequences([inp])
            # pre-padding the vectors:
            x_padded = pad_sequences(x_encoded, maxlen=35, padding='pre', value=0)
            # making prediction using the model:
            ynew = model_2.predict_classes(x_padded)

            if ynew == 0:
                self.output['text'] = 'Negative'  # display the output
            elif ynew == 1:
                self.output['text'] = 'Positive'

        if chosenClassifier == 2:   # If chosen 3-CLASS CLASSIFIER
            x_encoded = tokenizer_3.texts_to_sequences([inp])
            x_padded = pad_sequences(x_encoded, maxlen=22, padding='pre', value=0)
            ynew = model_3.predict_classes(x_padded)

            if ynew == 0:
                self.output['text'] = 'Positive'
            elif ynew == 1:
                self.output['text'] = 'Negative'
            elif ynew == 2:
                self.output['text'] = 'Neutral'

        if chosenClassifier == 3:   # if chosen SIX EMOTION CLASSIFIER:
            x_encoded = tokenizer_6.texts_to_sequences([inp])
            x_padded = pad_sequences(x_encoded, maxlen=22, padding='pre', value=0)
            ynew = model_6.predict_classes(x_padded)

            if ynew == 0:
                self.output['text'] = 'Neutral'
            elif ynew == 1:
                self.output['text'] = 'Worry'
            elif ynew == 2:
                self.output['text'] = 'Happiness'
            elif ynew == 3:
                self.output['text'] = 'Sadness'
            elif ynew == 4:
                self.output['text'] = 'Love'
            elif ynew == 5:
                self.output['text'] = 'Fun'


app = Window()
