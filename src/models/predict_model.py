import numpy as np
from ..data.make_dataset import preprocessing
import sys
from tensorflow.keras.callbacks import EarlyStopping
from train_model1 import Model1


def predict_sequence(input_seq, encoder_model, decoder_model, tran_lang, ref_lang, max_length_targ):
    reverse_target_word_index = tran_lang.index_word
    reverse_source_word_index = ref_lang.index_word
    target_word_index = tran_lang.word_index

    # Encode the input as state vectors.
    (e_out, e_h, e_c) = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sos']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        (output_tokens, h, c) = decoder_model.predict([target_seq]
                + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index+1]

        if sampled_token != 'eos':
            decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find the stop word.
        if sampled_token == 'eos' or len(decoded_sentence.split()) >= max_length_targ - 1:
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        (e_h, e_c) = (h, c)

    return decoded_sentence

if __name__ == "__main__":
    ref_lang, tran_lang, max_length_targ, max_length_inp, reference_tensor_train, reference_tensor_val, translation_tensor_train, translation_tensor_val = preprocessing()
    input_seq = sys.argv[0]
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

    encoder_model, decoder_model = model1.inference_model()

    predict_sequence(input_seq, encoder_model, decoder_model, tran_lang, ref_lang, max_length_targ)
    