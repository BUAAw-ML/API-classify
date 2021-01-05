from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input, Dropout, LSTM, Bidirectional, Activation, Conv2D, Reshape, Average, Flatten, GlobalAvgPool1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform, Orthogonal


from keras_utils.dictionary import GloveDictionary
from .serve_net import trainable_embedding_layer
    

def BiLSTM(input_shape, out_dim, dictionary, seed):
    sentence_indices = Input(shape=input_shape, dtype='int32')
    
    embedding_layer = trainable_embedding_layer(dictionary)
    embeddings = embedding_layer(sentence_indices) 
    emb_dim = embedding_layer.get_weights()[0].shape[1]
          
    lstm1 = Bidirectional(LSTM(512, return_sequences=False, kernel_initializer=glorot_uniform(seed=seed), recurrent_initializer=Orthogonal(seed=seed)))(embeddings)
    features = Dropout(0.1, seed = seed)(lstm1)
    X = Dense(1024, activation='relu', kernel_initializer=glorot_uniform(seed))(features)
    X = Dropout(0.1, seed = seed)(X)
    X = Dense(400, activation='relu', kernel_initializer=glorot_uniform(seed))(X)
    X = Dropout(0.1, seed = seed)(X)
    X = Dense(out_dim, activation='sigmoid', kernel_initializer=glorot_uniform(seed))(X)
    
    model = Model(inputs=sentence_indices, outputs=X)
    
    return model