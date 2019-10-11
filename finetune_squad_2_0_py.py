#Let's import our libraries
import numpy as np
import pandas as pd
import zipfile
import sys
import datetime
import os

#mounting google drive to colab
from google.colab import drive
drive.mount('/content/drive')
print(os.listdir("/content/drive/My Drive/SQuAD JSON-v2.0"))

!wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip

REPO = 'model_repo'
with zipfile.ZipFile("uncased_L-24_H-1024_A-16.zip","r") as zip_ref:
    zip_ref.extractall(REPO)
 
#Listing the files in the model
!dir 'model_repo/uncased_L-24_H-1024_A-16'

#Listing files in the model_repo
!dir 'model_repo'

#Downloading some python files from Google's BERT needed for training
!wget https://raw.githubusercontent.com/google-research/bert/master/modeling.py 
!wget https://raw.githubusercontent.com/google-research/bert/master/optimization.py 
!wget https://raw.githubusercontent.com/google-research/bert/master/run_squad.py 
!wget https://raw.githubusercontent.com/google-research/bert/master/tokenization.py

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
!wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
!wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
!wget https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
!dir

#Renaming our index.html file
!mv index.html evaluate-v2.0.py

!dir model_repo/uncased_L-24_H-1024_A-16

#Creating a new directory
!mkdir output

#print full system path of output directory
!pwd output/

#Code to run the run_squad.py file with training set and test set of SQuAD 2.0
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
--output_dir=/tmp/squad_large/

#python evaluate-v2.0.py dev-v2.0.json ./squad/predictions.json --na-prob-file ./squad/null_odds.json
