'''
Fine-tunning BERT for SQuAD 2.0.
'''

#Installing wget
pip install wget

#Let's import our libraries
import zipfile
import os
import shutil
import pandas as pd
import numpy as np
import wget

#mounting google drive to colab
from google.colab import drive
drive.mount('/content/drive')
#print(os.listdir("/content/drive/My Drive/SQuAD JSON-v2.0"))
print(os.listdir("/content/drive/My Drive/Mydataset/SQuAD JSON-v2.0"))

#Downloading our uncased BERT large model
URL = 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip'
FILENAME = wget.download(URL)

REPO = 'model_repo'
with zipfile.ZipFile("uncased_L-24_H-1024_A-16.zip", "r") as zip_ref:
    zip_ref.extractall(REPO)

#Listing the files in the model
os.listdir('model_repo/uncased_L-24_H-1024_A-16')

#Listing files in the model_repo
os.listdir('model_repo')

#Downloading some python files from Google's BERT needed for training
URL1 = 'https://raw.githubusercontent.com/google-research/bert/master/modeling.py'
URL2 = 'https://raw.githubusercontent.com/google-research/bert/master/optimization.py'
URL3 = 'https://raw.githubusercontent.com/google-research/bert/master/run_squad.py'
URL4 = 'https://raw.githubusercontent.com/google-research/bert/master/tokenization.py'
FILENAME1 = wget.download(URL1)
FILENAME2 = wget.download(URL2)
FILENAME3 = wget.download(URL3)
FILENAME4 = wget.download(URL4)

#Creating an output directory for our output
#Creating a bert pretrained directory
#Available pretrained model checkpoints:
#uncased_L-12_H-768_A-12: uncased BERT base model
#uncased_L-24_H-1024_A-16: uncased BERT large model
#cased_L-12_H-768_A-12: cased BERT large model
BERT_MODEL = 'uncased_L-24_H-1024_A-16'
BERT_PRETRAINED_DIR = f'{REPO}/uncased_L-24_H-1024_A-16'
OUTPUT_DIR = f'{REPO}/outputs'
print(f'***** Model output directory: {OUTPUT_DIR} *****')
print(f'***** BERT pretrained directory: {BERT_PRETRAINED_DIR} *****')

#The wget command helps us to download the SQuAD 2.0 dataset from the internet
URL5 = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json'
URL6 = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json'
HN = 'https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/'
URL7 = HN
FILENAME5 = wget.download(URL5)
FILENAME6 = wget.download(URL6)
FILENAME7 = wget.download(URL7)
os.listdir()

os.listdir('model_repo/uncased_L-24_H-1024_A-16')

#Creating a new directory
os.mkdir('output')

#Code to run the run_squad.py file with training set and test set of SQuAD 2.0
!python3 run_squad.py \
--vocab_file=model_repo/uncased_L-24_H-1024_A-16/vocab.txt \
--bert_config_file=model_repo/uncased_L-24_H-1024_A-16/bert_config.json \
--init_checkpoint=model_repo/uncased_L-24_H-1024_A-16/bert_model.ckpt \
--do_train=True \
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
#--use_tpu=True \
#--tpu_name=v2-128 \

#Code to run evaluate-v2.0 to get f1 score
!python3 evaluate-v2.0.py \
dev-v2.0.json pred.json

#Code to run the run_squad.py file with predicting set and test set of SQuAD 2.0
!python3 run_squad.py \
--vocab_file=model_repo/uncased_L-24_H-1024_A-16/vocab.txt \
--bert_config_file=model_repo/uncased_L-24_H-1024_A-16/bert_config.json \
--init_checkpoint=model_repo/uncased_L-24_H-1024_A-16/bert_model.ckpt \
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
#--use_tpu=True \
#--tpu_name=v2-128 \
