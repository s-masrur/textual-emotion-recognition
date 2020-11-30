Textual Emotion Recognition using Deep Learning.

For the extended overview of the project see Dissertation.pdf

TL:DR...

Long short-term memory (LSTM) is a type of Deep Learning model which is capable of "adding" or "removing" information to/from the memory. The idea of the project was to see how effective LSTM (and its strength) would be in emotion recognition using text documents. A simple application was implemented using Tkinter package to showcase three trained models: binary classifier (positive/negative), ternary classifier (positive, negative, and neutral) and six emotions classifier (neutral, worry, happiness, sadness, love, and fun). 

"Sentiment140" a dataset with 1,6 million samples divided into two classes (positive and negative) evenly, helped in training the binary classifier. The achieved accuracy of 79% for binary classifier was an adequate result but did not improve the state-of-the-art performance. Since the dataset has a large number of samples, the file was too large in size to add to the repository.

The second dataset which is uploaded to this repository has much smaller number of samples but divided into many classes of emotions. This dataset helped to experiment with the other two classifiers achieving 61% and 41% accuracy for ternary and six emotions classifiers, respectively.
