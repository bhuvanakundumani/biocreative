download the trainign and development files from https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vii/track-1/

Unzip the files.

Run this command to generate train.tsv, dev.tsv and test.tsv in the folder processed_data.

```
python preprocess.py

```
### errors in bioelectra to be fixed

## supported modeltype : ['biobert', 'pubmedbert','bioelectra'].
## To run the model
```
python drugprot_pytorch.py -output model_aug16_biobert -modeltype biobert -overwrite true -lr 3e-5 -epochs 1 

```



