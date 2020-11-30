import re
import nltk
from keras.utils import to_categorical
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
import preprocessor as p
import pickle


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('wordnet_ic')
nltk.download('punkt')

stopwords = stopwords.words('english')
not_stopwords = {'not', 'couldnt', 'didn', 'didnt', 'doesn', 'doesnt', 'hadn', 'hadnt', 'hasn', 'hasnt', 'haven',
                 'havent', 'isn', 'isnt', 'ma', 'mightn', 'mightnt', 'mustn', 'mustnt', 'needn', 'neednt',
                 'shan', 'shant', 'shouldn', 'shouldnt', 'wasn', 'wasnt', 'weren', 'werent', 'won', 'wont',
                 'wouldn', 'wouldnt'}
# words that should be removed from stopwords:
stop_words = set([word for word in stopwords if word not in not_stopwords])

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# Text Preprocessing:
def preproc(data):  # dataset 1 or 2:
    train_x = data[:, 3]    # 3 or 5     MODIFY THESE PARAMETERS
    train_y = data[:, 1]    # 1 or 0     DEPENDING ON THE DATASET
    vocab = []

    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION)

    for ind in range(len(train_x)):
        train_x[ind] = p.clean(train_x[ind])                                  # Remove URLs, emojis, and usernames
        train_x[ind] = re.sub('[^A-Za-z ]+', '', train_x[ind])                # Removing special characters and numbers
        train_x[ind] = word_tokenize(train_x[ind])                            # Tokenization
        train_x[ind] = [word.lower() for word in train_x[ind]]                # Normalizing case (lower case)
        train_x[ind] = [w for w in train_x[ind] if w not in stopwords]        # Removing stopwords
        # train_x[ind] = [stemmer.stem(word) for word in train_x[ind]]        # Stemming
        train_x[ind] = [lemmatizer.lemmatize(word) for word in train_x[ind]]  # Lemmatizing
        train_x[ind] = [i for i in train_x[ind] if len(i) > 1]                # Removing single characters
        vocab.extend(train_x[ind])                                            # Add the words to vocabulary

    vocab = list(set(vocab))  # Removing duplicate words from vocabulary

    maxlen = len(max(train_x, key=len))  # max length of sentences
    vocablen = len(vocab)

    # print("MAX LENGTH: ", maxlen)
    # print("VOCAB LENGTH: ", vocablen)

    return train_x, train_y, vocablen, maxlen


# Encoding the dataset:
def encode(x, y, vocablen, maxlen, num_classes):
    for i in range(len(x)):
        x[i] = ' '.join(x[i])

    tk = Tokenizer(num_words=vocablen, lower=False)
    tk.fit_on_texts(x)

    x_encoded = tk.texts_to_sequences(x)     # integer encoding
    x_padded = pad_sequences(x_encoded, maxlen=maxlen, padding='pre', value=0)  # pre-padding the vectors
    y_encoded = to_categorical(y, num_classes=num_classes)  # one hot encoding the labels

    # saving the tokenizer
    with open('models/tokenizer', 'wb') as handle:
        pickle.dump(tk, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return x_padded, y_encoded
