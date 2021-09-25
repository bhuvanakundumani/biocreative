Download the training, development and test files from https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vii/track-1/

Unzip the files.

Run this command to generate train.tsv, dev.tsv and test.tsv in the folder processed_data.

```
python preprocess.py

```
# Chemprot data - For additional data
Register and manually download [ChemProt](https://biocreative.bioinformatics.udel.edu/news/corpora/) . Once you have downloaded chemprot data, unzip the data to get the train, test and development data in the ChemProt_Corpus folder. 

Run this command to generate train.tsv, dev.tsv and test.tsv in the folder processed_chemprotdata.
To preprocess the ChemProt data , run the command 
```
python preprocess_chemprot.py

```

## supported modeltype : ['biobert', 'pubmedbert','bioelectra']
## To run the model
```
python drugprot_pytorch.py -output model_aug16_biobert -modeltype biobert -overwrite true -lr 3e-5 -epochs 1 

```
Code for generation of predictions in the format required by the competiton is available at biobert.ipynb, pubmedbert.ipynb and bioelectra.ipynb for biobert, pubmedbert and bioelectra respectively. The predictions in the required format will be written to test_submission_sep15 folder. 



