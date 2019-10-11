# Question-Answering-using-BERT
Use BERT for Question Answering and finetune train with SQuAD 2.0 

To use this pre-trained BERT model on your local desktop
You need to use Google Colab Notebook so you can run it using Cloud TPU
In your Colab workspace, click on 'Edit', Then on 'Notebook Settings'
Select 'TPU' as your hardware accelerator to enable you to train your model faster

You can download the $SQUAD_DIR.zip file from the repository.
That contains the 'train-v2.0.json', 'evaluate-v2.0.py' and 'dev-v2.0.json' file, i.e the SQuAD 2.0 dataset
Then you can either upload the files from your local desktop to your Colab Notebook or upload it to your Google Drive and access it from there.

We use the uncased BERT large model 'uncased_L-12_H-1024_A-16' to train our model
We also used the !wget to download our uncased BERT large model
and the !dir command was used to list files
!mv was used to rename files
!mkdir was used to create a directory
!pwd was used to print full system path of output directory

When you run the code, an output directory will be created
