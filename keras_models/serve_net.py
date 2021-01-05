from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input, Dropout, LSTM, Bidirectional, Activation, Conv2D, Reshape, Average
from tensorflow.keras.optimizers import Adam

from utils.dictionary import GloveDictionary


def trainable_embedding_layer(dictionary):
    vocab_len = len(dictionary.word_to_index) + 1
    emb_dim = dictionary.word_to_vec_map[dictionary.index_to_word[1]].shape[0]
    emb_matrix = dictionary.build_emb_matrix()
    
    embedding_layer = Embedding(vocab_len, emb_dim, mask_zero=True)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


def ServeNet(input_shape, out_dim, dictionary):
    sentence_indices = Input(shape=input_shape, dtype='int32')
    
    # Create the embedding layer pretrained with GloVe Vectors
    embedding_layer = trainable_embedding_layer(dictionary)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)
    emb_dim = embedding_layer.get_weights()[0].shape[1]
     
    embeddings = Reshape((input_shape[0], emb_dim, 1))(embeddings)
    
    features1 = Conv2D(64, kernel_size=(3, 3), padding='same')(embeddings)
    features2 = Conv2D(1, kernel_size=(1, 1), padding='same')(features1)
    features = Reshape((input_shape[0], emb_dim))(features2)
     
    print(features)
      
    X = Bidirectional(LSTM(512, return_sequences=False))(features)
    # Add dropout with a probability of 0.5
    X = Dropout(0.1)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(200, activation='relu')(X)
    X = Dropout(0.1)(X)
    X = Dense(out_dim, activation='sigmoid')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
    
    return model
