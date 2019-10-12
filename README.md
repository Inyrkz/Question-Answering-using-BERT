# Question-Answering-using-BERT
## BERT
BERT (Bidirectional Encoder Representations from Transformers) is a recent paper published by researchers at Google AI Language. It has caused a stir in the Machine Learning community by presenting state-of-the-art results in a wide variety of NLP tasks, including Question Answering SQuAD v2.0 which we finetune here.
## SQUAD 2.0
Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

SQuAD2.0 tests the ability of a system to not only answer reading comprehension questions, but also abstain when presented with a question that cannot be answered based on the provided paragraph.

## Pre-trained Model
The aim of using a pre-trained Model is because of limitation of training our BERT Model which will require use of TPU which we do not have adequate access to.
To achieve finetuning of BERT on SQUAD 2.0 we used BERT-Large, Uncased (Whole Word Masking): 24-layer, 1024-hidden, 16-heads, 340M parameters on Google Colab Notebook using its Cloud TPU to enable us finetune faster. We downloaded SQUAD_DIR.zip file from https://github.com/google-research/bert which contained needed SQUAD 2.0 datasets such as 'train-v2.0.json', evaluate.py and 'dev-v2.0.json'

## Implementation
The implementation is documented in https://github.com/Inyrkz/Question-Answering-using-BERT/blob/master/Fine_tuning_BERT%202.ipynb
Using Colob Notebook, we importing the appropriate libraries 
```bash
import pandas as pd
import numpy as np
import zipfile
import os
pip install wget
import wget
import shutil

```
We mounted the drive containing our dev2.0.json and train2.0.json datasets needed
```bash
from google.colab import drive
drive.mount('/content/drive')
print(os.listdir("/content/drive/My Drive/SQuAD JSON-v2.0"))
```

We downloaded the BERT pretrained model needed for our finetuning, we were also provided with some python files to help run our model, files like modeling.py, optimization.py, run_squad.py, tokenization.py which were all downloaded from https://github.com/google-research/bert using the Wget method
```bash
URL= 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip'
filename = wget.download(URL)

REPO = 'model_repo'
with zipfile.ZipFile("uncased_L-24_H-1024_A-16.zip","r") as zip_ref:
    zip_ref.extractall(REPO)
    
os.listdir('model_repo/uncased_L-24_H-1024_A-16')
 
URL1= 'https://raw.githubusercontent.com/google-research/bert/master/modeling.py'
URL2= 'https://raw.githubusercontent.com/google-research/bert/master/optimization.py'
URL3= 'https://raw.githubusercontent.com/google-research/bert/master/run_squad.py'
URL4= 'https://raw.githubusercontent.com/google-research/bert/master/tokenization.py'
FILENAME1 = wget.download(URL1)
FILENAME2 = wget.download(URL2)
FILENAME3 = wget.download(URL3)
FILENAME4 = wget.download(URL4)
````

Next we created an output directory for our output
```bash
BERT_MODEL = 'uncased_L-24_H-1024_A-16'
BERT_PRETRAINED_DIR = f'{REPO}/uncased_L-24_H-1024_A-16'
OUTPUT_DIR = f'{REPO}/outputs'
print(f'***** Model output directory: {OUTPUT_DIR} *****')
print(f'***** BERT pretrained directory: {BERT_PRETRAINED_DIR} *****')
```

We downloaded other needed files for our training using 
```bash
URL5= 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json'
URL6= 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json'
URL7= 'https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/'
FILENAME5 = wget.download(URL5)
FILENAME6 = wget.download(URL6)
FILENAME7 = wget.download(URL7)
os.listdir()
```

Next we run the run_squad.py file with the training set and test set of SQUAD 2.0 using the parameters
```bash
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
```
After some hours of training we got outputs of checkpoint, eval.tf_record, graph.pbtxt, train.tf_record and pred.json

We then tested our model to get the f1 score so we can rank on the leaderboard of the SQUAD2.0 found here https://rajpurkar.github.io/SQuAD-explorer/ using our pred.json, evaluate-v2.0.py and dev-v2.0

```bash
!python3 evaluate-v2.0.py \
dev-v2.0.json pred.json 
```

## Result
After running the evaluate-v2.0.py, dev-2.0.json and pred.json with the above code we were above to get the following score which ranked our training no 54 on the leadership board

```bash
{
  "exact": 64.81091552261434,
  "f1": 67.60971132981278,
  "total": 11873,
  "HasAns_exact": 59.159919028340084,
  "HasAns_f1": 64.7655368790259,
  "HasAns_total": 5928,
  "NoAns_exact": 70.4457527333894,
  "NoAns_f1": 70.4457527333894,
  "NoAns_total": 5945
}
```

