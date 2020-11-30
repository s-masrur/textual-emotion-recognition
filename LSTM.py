import keras
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense


def create_model(input_dim, input_length, num_classes):

    model = Sequential()
    model.add(Embedding(input_dim=input_dim, input_length=input_length, output_dim=32))  # Embedding layer
    model.add(LSTM(32, dropout=0.4, recurrent_dropout=0.4))                              # LSTM layer
    model.add(Dense(num_classes, activation='softmax'))                                  # output Dense layer
    adam = keras.optimizers.Adam(lr=0.0002)                                              # optimizer
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    print(model.summary())

    return model
