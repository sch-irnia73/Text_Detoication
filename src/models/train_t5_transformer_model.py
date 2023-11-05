
"""## Hugging Face t5 Transformer"""

import pandas as pd
from simpletransformers.t5 import T5Model
from sklearn.model_selection import train_test_split
import sklearn



def t5_train(data):

    train_data,test_data = train_test_split(data.iloc[:10000],test_size=0.2)

    args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": 256,
        "num_train_epochs": 4,
        "num_beams": 1,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "use_multiprocessing": False,
        "save_steps": -1,
        "save_eval_checkpoints": True,
        "evaluate_during_training": False,
        'adam_epsilon': 1e-08,
        'eval_batch_size': 6,
        'fp_16': False,
        'gradient_accumulation_steps': 16,
        'learning_rate': 0.0003,
        'max_grad_norm': 1.0,
        'n_gpu': 1,
        'seed': 42,
        'train_batch_size': 6,
        'warmup_steps': 0,
        'weight_decay': 0.0
    }

    model = T5Model("t5","t5-small", args=args)

    model.train_model(train_data, eval_data=test_data, use_cuda=True,acc=sklearn.metrics.accuracy_score)



    transformer_model = model.model
    model_name = 't5_transformer_model'
    save_directory = './saved_models/'
    transformer_model.save_pretrained(save_directory, model_name)

if __name__ == "__main__":
    
    data = pd.read_csv('../../data/raw/filtered.tsv', sep='\t')

    data["input_text"] = data["reference"]
    data["target_text"] = data["translation"]
    data = data[["input_text", "target_text"]]
    data["prefix"] = "paraphrase"

    t5_train(data)


