
from nltk.tokenize import word_tokenize

"""## Hugging Face t5 Transformer"""

from simpletransformers.t5 import T5Model
from nltk.translate.bleu_score import sentence_bleu
from sklearn.model_selection import train_test_split
import pandas as pd
from pprint import pprint
import os

def predict_t5(trained_model_path, test_data):


    args = {
        "overwrite_output_dir": True,
        "max_seq_length": 256,
        "max_length": 50,
        "top_k": 50,
        "top_p": 0.95,
        "num_return_sequences": 5,
    }


    trained_model = T5Model("t5",trained_model_path,args=args)


    prefix = "paraphrase"
    pred = trained_model.predict([f"{prefix}: Now you're getting nasty."])
    pprint(pred)

    all_data_pred = [trained_model.predict([f"{prefix}: {i}"]) for i in test_data["input_text"]]

    all_data_pred_tokens = [word_tokenize(sent) for sent in all_data_pred]
    actual_values_tokens = [word_tokenize(sent) for sent in test_data["input_text"]]
    bleu_scores = [sentence_bleu([actual], pred) for actual, pred in zip(actual_values_tokens, all_data_pred_tokens)]

    average_bleu_score = sum(bleu_scores) / len(bleu_scores)

    print(f"Average BLEU Score: {average_bleu_score}")



if __name__ == "__main__":
    
    root_dir = os.getcwd()
    trained_model_path = os.path.join(root_dir,"outputs")

    data = pd.read_csv('../../data/raw/filtered.tsv', sep='\t')

    data["input_text"] = data["reference"]
    data["target_text"] = data["translation"]
    data = data[["input_text", "target_text"]]
    data["prefix"] = "paraphrase"
    train_data,test_data = train_test_split(data.iloc[:10000],test_size=0.2)

    predict_t5(trained_model_path, test_data)

