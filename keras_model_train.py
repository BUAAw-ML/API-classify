import pickle as pk

from tensorflow.keras import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from datasets import program_web
from keras_models.serve_net import ServeNet
from keras_models.CNN import CNN
from keras_models.LSTM import ServiceLSTM
from keras_models.RCNN import RCNN
from keras_models.BiLSTM import BiLSTM
from keras_utils.dictionary import GloveDictionary


def train_keras_model(model_type="ServeNet", data=None, glove_path="./glove.6B.200d.txt", save_model_path="./data/best.hdf5"):
    glove_dict = GloveDictionary(*GloveDictionary.read_glove_vecs(glove_path))
    train_X, train_Y = data['train']
    test_X, test_Y = data['test']
    if model_type == "ServeNet":
        net = ServeNet(train_X.shape[1:], train_Y.shape[-1], glove_dict)
    elif model_type == "CNN":
        net = CNN(train_X.shape[1:], train_Y.shape[-1], glove_dict, 0)
    elif model_type == "LSTM":
        net = ServiceLSTM(train_X.shape[1:], train_Y.shape[-1], glove_dict)
    elif model_type == "RCNN":
        net = RCNN(train_X.shape[1:], train_Y.shape[-1], glove_dict)
    elif model_type == "BiLSTM":
        net = BiLSTM(train_X.shape[1:], train_Y.shape[-1], glove_dict, 0)

    checkpointer = ModelCheckpoint(filepath=save_model_path,
        monitor='val_auc', verbose=1,
        save_best_only=True, mode="max")

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    net.compile(loss='binary_crossentropy', optimizer=adam,
        metrics=[metrics.AUC(multi_label=True, name="auc")])

    net.fit([train_X, train_X, train_X], train_Y,
        validation_data=([test_X, test_X, test_X], test_Y),
        epochs = 50, batch_size = 64,
        verbose = 1, shuffle=True,
        callbacks=[checkpointer])