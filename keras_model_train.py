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
from utils.dictionary import GloveDictionary


train_X, train_Y = program_web.load_data("data/train.pkl",
    "data/glove.6B.200d.txt",
    "data/index_to_tag.pkl",
    "data/tag_to_index.pkl")
max_len = len(train_X[0])
test_X, test_Y = program_web.load_data("data/test.pkl",
    "data/glove.6B.200d.txt",
    "data/index_to_tag.pkl",
    "data/tag_to_index.pkl",
    max_len=max_len)

glove_dict = GloveDictionary(*GloveDictionary.read_glove_vecs("data/glove.6B.200d.txt"))
# serve_net = ServeNet(train_X.shape[1:], train_Y.shape[-1], glove_dict)
# serve_net = CNN(train_X.shape[1:], train_Y.shape[-1], glove_dict, 0)
# serve_net = ServiceLSTM(train_X.shape[1:], train_Y.shape[-1], glove_dict)
# serve_net = RCNN(train_X.shape[1:], train_Y.shape[-1], glove_dict)
serve_net = BiLSTM(train_X.shape[1:], train_Y.shape[-1], glove_dict, 0)


checkpointer = ModelCheckpoint(filepath='data/BiLSTM.hdf5',
    monitor='val_auc', verbose=1,
    save_best_only=True, mode="max")

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
serve_net.compile(loss='binary_crossentropy', optimizer=adam,
    metrics=[metrics.AUC(multi_label=True, name="auc")])
# history = serve_net.fit(train_X, train_Y,
#     validation_data=(test_X, test_Y),
#     epochs = 50, batch_size = 64,
#     verbose = 1, shuffle=True,
#     callbacks=[checkpointer])
history = serve_net.fit([train_X, train_X, train_X], train_Y,
    validation_data=([test_X, test_X, test_X], test_Y),
    epochs = 50, batch_size = 64,
    verbose = 1, shuffle=True,
    callbacks=[checkpointer])
pk.dump(history.history, open("data/history.pkl", "wb"))
