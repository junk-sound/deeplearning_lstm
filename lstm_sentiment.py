from keras.layers import Input, Embedding, Reshape, Conv2D, MaxPool2D, concatenate, Dropout, Dense, BatchNormalization
from keras.layers import LSTM
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
import numpy as np
import pandas as pd
import time
import os

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

date = time.strftime('%x', time.localtime(time.time())).replace('/', '')

rd = 8
train_file = 'train_rd_' + str(rd) + '.xlsx'
test_file = 'test_rd_' + str(rd) + '.xlsx'
model_name = "model_multi_7_cnn_lstm_1_" + str(rd) + '_' + date + ".h5"

model_file_path = '/home/ubuntu/junksound/sentiment_analysis/weight/'
data_file_path = '/home/ubuntu/junksound/sentiment_analysis/sampling/'

filter_num = 130
embed_size = 128

max_length = 165
min_length = 5

batch_size = 500
epoch_num = 200

n_class = 7
n_pooling = 30


def create_model(embed_size=embed_size, max_length=max_length, filter_sizes=(2, 3, 4, 5), filter_num=filter_num,
                 n_class=n_class, n_pooling=n_pooling):
    inp = Input(shape=(max_length,))
    emb = Embedding(0xffff, embed_size)(inp)
    emb_ex = Reshape((max_length, embed_size, 1))(emb)

    layer_merged = []
    for filter_size in filter_sizes:

        conv = Conv2D(filter_num, (filter_size, embed_size), activation="relu", strides=(1, 1))(emb_ex)
        pool = MaxPool2D(pool_size=(int(max_length / n_pooling), 1), strides=(int(max_length / n_pooling), 1))(conv)
        reshape_pool = Reshape((filter_num, int((pool.shape[1])),))(pool)
        lstm = LSTM(units=100, return_sequences=True)(reshape_pool)
        lstm_reshape = Reshape((filter_num*100,))(lstm)
        layer_merged.append(lstm_reshape)
    feature = concatenate(layer_merged)

    reshape = Reshape((int(feature.shape[1]),))(feature)
    fc1 = Dense(128, activation="relu")(reshape)
    bn1 = BatchNormalization()(fc1)
    do1 = Dropout(0.5)(bn1)
    fc2 = Dense(128, activation="relu")(do1)
    bn2 = BatchNormalization()(fc2)
    do2 = Dropout(0.5)(bn2)
    fc3 = Dense(n_class, activation='softmax')(do2)
    model = Model(inputs=inp, outputs=fc3)
    print(model.output)
    return model


def load_data(filepath, max_length=max_length, min_length=min_length, n_class=n_class):
    count = 0
    comments, tmp_comments, emotions, tmp_emotion = [], [], [], []
    filebody = pd.read_excel(filepath, header=0)
    texts = filebody.values[:, 0]
    emotion_set = filebody.values[:, 1]
    emotion_set = emotion_set.astype('float32')
    emotion_set = np_utils.to_categorical(emotion_set, n_class)

    for text in texts:
        text = text.replace("\t", "")
        text = text.replace("\n", "")
        text = [ord(x) for x in text.strip()]
        text = text[:max_length]
        text_len = len(text)
        count += 1
        if text_len < min_length:
            continue
        if text_len < max_length:
            text = ([0] * (max_length - text_len)) + text
        tmp_comments.append(text)
        tmp_emotion.append(emotion_set[count - 1])
    comments.extend(tmp_comments[:len(texts)])
    emotions.extend(tmp_emotion[:len(texts)])
    return comments, emotions


def train(inputs, targets, batch_size=batch_size, epoch_count=epoch_num, max_length=max_length,
          model_filepath=model_file_path + model_name, learning_rate=0.001):
    start = learning_rate
    stop = learning_rate * 0.01
    learning_rates = np.linspace(start, stop, epoch_count)

    model = create_model(max_length=max_length)
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.fit(inputs, targets,
              epochs=epoch_count,
              batch_size=batch_size,
              verbose=1,
              # validation_split=0.1,
              shuffle=True,
              callbacks=[LearningRateScheduler(lambda epoch: learning_rates[epoch]), ])

    model.save(model_filepath)


if __name__ == "__main__":
    t_start = time.clock()
    comments, emotions = load_data(data_file_path + train_file)

    input_values = np.array(comments)
    target_values = np.array(emotions)

    # create_model(max_length=max_length)

    train(input_values, target_values, batch_size=batch_size, epoch_count=epoch_num)

    t_end = time.clock()
    print("running_time : %0.4f" % (t_end - t_start))

    model = load_model(model_file_path + model_name)
    X_test, Y_test = load_data(data_file_path + test_file)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    score = model.evaluate(X_test, Y_test, verbose=0)
    print(model.metrics_names)
    print(score)

    print(model_name)

    t_end = time.clock()
    print("running_time : %0.4f" % (t_end - t_start))