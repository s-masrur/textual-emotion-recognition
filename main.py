import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import preprocess as pre
from LSTM import create_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # remove unnecessary tensorflow warnings
warnings.simplefilter("ignore", UserWarning)
np.set_printoptions(threshold=np.inf)


df = pd.read_csv('datasets/text_emotion.csv', delimiter=',', encoding='latin-1')
df = df[~df["sentiment"].str.contains('surprise|relief|enthusiasm|anger|boredom|hate|empty')]

# label codes for encoding the labels of the FIRST dataset for classification of three classes:
# label_codes = {"sentiment": {"worry": 1, "hate": 1, "sadness": 1, "boredom": 1, "anger": 1,
#                              "relief": 0, "happiness": 0, "enthusiasm": 0, "love": 0, "fun": 0,
#                              "neutral": 2, "empty": 2}}

# label codes for classification of the six emotions
label_codes = {"sentiment": {"neutral": 0, "worry": 1, "happiness": 2, "sadness": 3, "love": 4, "fun": 5}}

# label codes for encoding the labels of the SECOND dataset:
# label_codes = {"sentiment": {4: 1}}

df.replace(label_codes, inplace=True)
print(df["sentiment"].value_counts())

dataset = df.to_numpy()
num_classes = df['sentiment'].nunique()


# Preprocess, encode and split the dataset:
train_X, train_Y, vocab_len, max_len = pre.preproc(dataset)
train_X, train_Y = pre.encode(train_X, train_Y, vocab_len, max_len, num_classes)  # set shuffle=True for binary classif
train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.30, shuffle=False)

# Create, fit and evaluate the model:
model = create_model(vocab_len, max_len, num_classes)
hist = model.fit(train_X, train_Y, shuffle=False, epochs=4, batch_size=8, validation_data=(test_X, test_Y))
pred = model.predict_classes(test_X)  # make predictions on test set
results = model.evaluate(test_X, test_Y, batch_size=8)
print('test loss, test acc:', results)

# Classification report and confusion matrix:
test_Y = np.argmax(test_Y, axis=1)  # reverse one-hot encoding
print("Classification report for classifier %s:\n%s\n"  % (model, metrics.classification_report(test_Y, pred)))
print(confusion_matrix(list(test_Y), list(pred)))

model.save('models/LSTM.h5')


# Accuracy and loss plots:
# list all data in history
# print(hist.history.keys())
# summarize history for accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('plots/accuracy.png')
plt.clf()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('plots/loss.png')
