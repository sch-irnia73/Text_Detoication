from keras.layers import Embedding, Dense
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


"""## Hypothesis 3"""

class Model3():

    def __init__(self):
        pass

    def create_model(self, ref_lang, tran_lang, max_length_inp, latent_dim=300, embedding_dim=100):

        self.embeddings_index = {}
        f = open('../../data/external/glove.6B.100d.txt', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()

        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim

        # Encoder
        self.encoder_inputs = Input(shape=(max_length_inp, ))

        embeddings_matrix = self.get_embeddings(ref_lang)

        # Embedding layer
        self.enc_emb = Embedding(len(ref_lang.word_index) + 1, embedding_dim, input_length=max_length_inp, weights=[embeddings_matrix],
                              trainable=False)(self.encoder_inputs)


        # Encoder LSTM 1
        self.encoder_lstm1 = LSTM(latent_dim, return_sequences=True,
                              return_state=True, dropout=0.4,
                              recurrent_dropout=0.4)
        (self.encoder_output, self.state_h, self.state_c) = self.encoder_lstm1(self.enc_emb)

        # Set up the decoder, using encoder_states as the initial state
        self.decoder_inputs = Input(shape=(None, ))

        embeddings_matrix = self.get_embeddings(tran_lang)
        # Embedding layer
        self.dec_emb = Embedding(len(tran_lang.word_index) + 1, embedding_dim, weights=[embeddings_matrix],
                              trainable=False)(self.decoder_inputs)

        # Decoder LSTM
        self.decoder_lstm = LSTM(latent_dim, return_sequences=True,
                              return_state=True, dropout=0.4,
                              recurrent_dropout=0.2)
        (decoder_outputs, decoder_fwd_state, decoder_back_state) = \
              self.decoder_lstm(self.dec_emb, initial_state=[self.state_h, self.state_c])

        # Dense layer
        self.decoder_dense = TimeDistributed(Dense(len(tran_lang.word_index) + 1, activation='softmax'))
        decoder_outputs = self.decoder_dense(decoder_outputs)

        # Define the model
        return Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)

    def get_embeddings(self, lang):
        word_index = lang.word_index
        vocab_size = len(word_index)

        embeddings_matrix = np.zeros((vocab_size+1, self.embedding_dim))
        for word, i in word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embeddings_matrix[i] = embedding_vector

        return embeddings_matrix

    def inference_model(self):

        # Encode the input sequence to get the feature vector
        encoder_model = Model(inputs=self.encoder_inputs, outputs=[self.encoder_output,
                                self.state_h, self.state_c])

        # Below tensors will hold the states of the previous time step
        decoder_state_input_h = Input(shape=(self.latent_dim, ))
        decoder_state_input_c = Input(shape=(self.latent_dim, ))
        decoder_hidden_state_input = Input(shape=(max_length_inp, self.latent_dim))

        # Get the embeddings of the decoder sequence
        dec_emb2 = self.dec_emb

        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        (decoder_outputs2, state_h2, state_c2) = self.decoder_lstm(dec_emb2,
                  initial_state=[decoder_state_input_h, decoder_state_input_c])

        # A dense softmax layer to generate prob dist. over the target vocabulary
        decoder_outputs2 = self.decoder_dense(decoder_outputs2)

        # Final decoder model
        decoder_model = Model([self.decoder_inputs] + [decoder_hidden_state_input,
                                decoder_state_input_h, decoder_state_input_c],
                                [decoder_outputs2] + [state_h2, state_c2])

        return encoder_model, decoder_model

if __name__ == "__main__":
    
    ref_lang, tran_lang, max_length_targ, max_length_inp, reference_tensor_train, reference_tensor_val, translation_tensor_train, translation_tensor_val = preprocessing()
    

    model3 = Model3()
    m3 = model3.create_model(ref_lang, tran_lang, max_length_inp)
    m3.summary()

    m3.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

    history = m3.fit(
        [reference_tensor_train, translation_tensor_train[:, :-1]],
        translation_tensor_train.reshape(translation_tensor_train.shape[0], translation_tensor_train.shape[1], 1)[:, 1:],
        epochs=10,
        callbacks=[es],
        batch_size=64,
        validation_data=([reference_tensor_val, translation_tensor_val[:, :-1]],
                        translation_tensor_val.reshape(translation_tensor_val.shape[0], translation_tensor_val.shape[1], 1)[:, 1:]),
        )

    m3.save('model3.h5')

    encoder_model3, decoder_model3 = model3.inference_model()



