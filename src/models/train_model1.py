from keras.layers import Embedding, Dense
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from ..data.make_dataset import preprocessing
"""## Hypothesis 1"""

class Model1():

    def __init__(self):
        pass

    def create_model(self, ref_lang, tran_lang, max_length_inp, latent_dim=300, embedding_dim=200):

        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim

        # Encoder
        self.encoder_inputs = Input(shape=(max_length_inp, ))

        # Embedding layer
        self.enc_emb = Embedding(len(ref_lang.word_index) + 1, embedding_dim,
                          trainable=True)(self.encoder_inputs)

        # Encoder LSTM 1
        self.encoder_lstm1 = LSTM(latent_dim, return_sequences=True,
                          return_state=True, dropout=0.4,
                          recurrent_dropout=0.4)
        (self.encoder_output, self.state_h, self.state_c) = self.encoder_lstm1(self.enc_emb)

        # Set up the decoder, using encoder_states as the initial state
        self.decoder_inputs = Input(shape=(None, ))

        # Embedding layer
        self.dec_emb_layer = Embedding(len(tran_lang.word_index) + 1, embedding_dim, trainable=True)
        dec_emb = self.dec_emb_layer(self.decoder_inputs)

        # Decoder LSTM
        self.decoder_lstm = LSTM(latent_dim, return_sequences=True,
                              return_state=True, dropout=0.4,
                              recurrent_dropout=0.2)
        (decoder_outputs, decoder_fwd_state, decoder_back_state) = \
        self.decoder_lstm(dec_emb, initial_state=[self.state_h, self.state_c])

        # Dense layer
        self.decoder_dense = TimeDistributed(Dense(len(tran_lang.word_index) + 1, activation='softmax'))
        decoder_outputs = self.decoder_dense(decoder_outputs)

        # Define the model
        return Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)

    def inference_model(self):

        # Encode the input sequence to get the feature vector
        encoder_model = Model(inputs=self.encoder_inputs, outputs=[self.encoder_output,
                            self.state_h, self.state_c])

        # Decoder setup

        # Below tensors will hold the states of the previous time step
        decoder_state_input_h = Input(shape=(self.latent_dim, ))
        decoder_state_input_c = Input(shape=(self.latent_dim, ))
        decoder_hidden_state_input = Input(shape=(max_length_inp, self.latent_dim))

        # Get the embeddings of the decoder sequence
        dec_emb2 = self.dec_emb_layer(self.decoder_inputs)

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
    model1 = Model1()
    m1 = model1.create_model(ref_lang, tran_lang, max_length_inp)
    m1.summary()

    m1.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

    history = m1.fit(
        [reference_tensor_train, translation_tensor_train[:, :-1]],
        translation_tensor_train.reshape(translation_tensor_train.shape[0], translation_tensor_train.shape[1], 1)[:, 1:],
        epochs=10,
        callbacks=[es],
        batch_size=64,
        validation_data=([reference_tensor_val, translation_tensor_val[:, :-1]],
                        translation_tensor_val.reshape(translation_tensor_val.shape[0], translation_tensor_val.shape[1], 1)[:, 1:]),
        )

    m1.save('model1.h5')

    encoder_model1, decoder_model1 = model1.inference_model()



