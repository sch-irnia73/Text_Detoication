import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
import re

def preprocessing():


    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))

    # Load data
    data = pd.read_csv('../../data/raw/filtered.tsv', sep='\t')



    data = data.iloc[:10000]

    def preprocess_sentence(s, put_tok=True):
        s = s.lower().strip()
        text = re.sub(r'[0-9]', '' ,s)
        tokens = nltk.word_tokenize(s)
        tokens = [word for word in tokens if (word not in stopwords.words()) and (len(word) > 2) \
                and (word not in string.punctuation)]

        if put_tok:
            s = 'sos ' + ' '.join(tokens) + ' eos'
        return s

    data['reference'] = data['reference'].apply(lambda x: preprocess_sentence(x, put_tok=False))
    data['translation'] = data['translation'].apply(preprocess_sentence)


    def tokenize(lang):
        lang_tokenizer = Tokenizer()
        lang_tokenizer.fit_on_texts(lang)

        tensor = lang_tokenizer.texts_to_sequences(lang)
        tensor = pad_sequences(tensor, padding='post')

        return tensor, lang_tokenizer

    reference_tensor, ref_lang = tokenize(data['reference'])
    translation_tensor, tran_lang = tokenize(data['translation'])

    max_length_targ, max_length_inp = translation_tensor.shape[1], reference_tensor.shape[1]

    reference_tensor_train, reference_tensor_val, translation_tensor_train, translation_tensor_val = train_test_split(reference_tensor, translation_tensor, test_size=0.2)

    return ref_lang, tran_lang, max_length_targ, max_length_inp, reference_tensor_train, reference_tensor_val, translation_tensor_train, translation_tensor_val
