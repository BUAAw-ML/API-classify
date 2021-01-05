from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input, Dropout, LSTM, Bidirectional, Activation, Conv2D, Reshape, Average, Flatten, GlobalAvgPool1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform

from keras_utils.dictionary import GloveDictionary
from .serve_net import trainable_embedding_layer


def CNN(input_shape, out_dim, dictionary, seed):
    sentence_indices = Input(shape=input_shape, dtype='int32')
    
    embedding_layer = trainable_embedding_layer(dictionary)
    embeddings = embedding_layer(sentence_indices)
    emb_dim = embedding_layer.get_weights()[0].shape[1] 
     
    embeddings = Reshape((input_shape[0], emb_dim, 1))(embeddings)
    
    cnn1 = Conv2D(64, kernel_size=(5, 5), padding='same', kernel_initializer=glorot_uniform(seed=seed))(embeddings)
    cnn1 = Dropout(0.1)(cnn1)
    cnn2 = Conv2D(32, kernel_size=(3, 3), padding='same', kernel_initializer=glorot_uniform(seed=seed))(cnn1)
    cnn2 = Dropout(0.1)(cnn2)
    cnn3 = Conv2D(1, kernel_size=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=seed))(cnn2)
    features_cnn = Reshape((input_shape[0], emb_dim))(cnn3)
     
    flat = GlobalAvgPool1D()(features_cnn)  
      
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(1024, activation='relu', kernel_initializer=glorot_uniform(seed))(flat)
    X = Dropout(0.1)(X)
    X = Dense(400, activation='relu', kernel_initializer=glorot_uniform(seed))(X)
    X = Dropout(0.1)(X)
    X = Dense(out_dim, activation='sigmoid', kernel_initializer=glorot_uniform(seed))(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
    
    ### END CODE HERE ###
    
    return model