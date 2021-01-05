from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input, Dropout, LSTM, Bidirectional, Activation, Conv2D, Reshape, Average, Flatten, GlobalAvgPool1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform

from keras_utils.dictionary import GloveDictionary
from .serve_net import trainable_embedding_layer


def ServiceLSTM(input_shape, out_dim, dictionary):
    
    
    sentence_indices = Input(shape=input_shape, dtype='int32')
    
    embedding_layer = trainable_embedding_layer(dictionary)
    embeddings = embedding_layer(sentence_indices)
    emb_dim = embedding_layer.get_weights()[0].shape[1]
     
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(256, return_sequences=False)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(0.1)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(out_dim, activation='sigmoid')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
    
    return model