from tensorflow.keras.models import Model
from tensorflow.keras import backend
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense, Embedding, Input, Dropout, LSTM, Bidirectional, Activation, Conv1D, Reshape, Average, Flatten, GlobalAvgPool1D, SimpleRNN, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform

from utils.dictionary import GloveDictionary
from .serve_net import trainable_embedding_layer


def RCNN(input_shape, out_dim, dictionary):
    
    embedding_layer = trainable_embedding_layer(dictionary)
    
    document = Input(shape = input_shape, dtype = "int32")
    left_context = Input(shape = input_shape, dtype = "int32")
    right_context = Input(shape = input_shape, dtype = "int32")
    
    doc_embedding = embedding_layer(document)
    l_embedding = embedding_layer(left_context)
    r_embedding = embedding_layer(right_context)
    
    forward = SimpleRNN(50, return_sequences = True)(l_embedding) # See equation (1).
    backward = SimpleRNN(50, return_sequences = True, go_backwards = True)(r_embedding) # See equation (2).
    # Keras returns the output sequences in reverse order.
    backward = Lambda(lambda x: backend.reverse(x, axes = 1))(backward)
    together = Concatenate(axis = 2)([forward, doc_embedding, backward]) # See equation (3).
    
    semantic = Conv1D(50, kernel_size = 1, activation = "relu")(together) # See equation (4).

    # Keras provides its own max-pooling layers, but they cannot handle variable length input
    # (as far as I can tell). As a result, I define my own max-pooling layer here.
    pool_rnn = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (50, ))(semantic) # See equation (5).

    output = Dense(out_dim, activation="sigmoid")(pool_rnn) # See equations (6) and (7).

    model = Model(inputs = [document, left_context, right_context], outputs = output)
         
    
    return model
