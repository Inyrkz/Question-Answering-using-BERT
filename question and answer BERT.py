import pandas as pd
import numpy as np
import zipfile
import os
import wget
import shutil

print(os.listdir("C:/Users/Ademola/Desktop/dataset/json_data/input"))

url= 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip'
filename = wget.download(url)


repo = 'model_repo'
with zipfile.ZipFile("uncased_L-12_H-768_A-12.zip","r") as zip_ref:
    zip_ref.extractall(repo)

dir( 'model_repo/uncased_L-12_H-768_A-12')

dir( 'model_repo')

wget.download('https://raw.githubusercontent.com/google-research/bert/master/modeling.py')
wget.download('https://raw.githubusercontent.com/google-research/bert/master/run_squad.py')
wget.download('https://raw.githubusercontent.com/google-research/bert/master/tokenization.py')
wget.download('https://raw.githubusercontent.com/google-research/bert/master/optimization.py') 

BERT_MODEL = 'uncased_L-12_H-768_A-12'
BERT_PRETRAINED_DIR = f'{repo}/uncased_L-12_H-768_A-12'
OUTPUT_DIR = f'{repo}/outputs'
print(f'***** Model output directory: {OUTPUT_DIR} *****')
print(f'***** BERT pretrained directory: {BERT_PRETRAINED_DIR} *****')

wget.download('https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json')
wget.download('https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json')
wget.download('https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/')

dir()

!move index.html 'evaluate-v2.0.py'
dir('model_repo/uncased_L-12_H-768_A-12')

os.mkdir('output')

!python run_squad.py \
--vocab_file=model_repo/uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file=model_repo/uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=model_repo/uncased_L-12_H-768_A-12/bert_model.ckpt \
--do_train=False \
--train_file=train-v2.0.json \
--do_predict=True \
--predict_file=dev-v2.0.json \
--train_batch_size=24 \
--learning_rate=3e-5 \
--num_train_epochs=2.0 \
--max_seq_length=384 \
--doc_stride=128 \
--version_2_with_negative=True \
--output_dir=/content/output



