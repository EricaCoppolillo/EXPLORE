# EXPLORE

This is the official repository of our paper "**Relevance Meets Diversity: A User-Centric Framework for Knowledge Exploration Through Recommendations**", accepted at KDD2024.

## CITATION

If you want to use the code, please cite us:

```
@inproceedings{RelevanceMeetsDiversity,
author = {Coppolillo, Erica and Manco, Giuseppe and Gionis, Aristides},
title = {Relevance Meets Diversity: A User-Centric Framework for Knowledge Exploration Through Recommendations},
year = {2024},
publisher = {Association for Computing Machinery},
doi = {10.1145/3637528.3671949},
booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {490–501},
series = {KDD '24}
}
```


## USAGE

In order to reproduce the experiments, just follow the steps:

1. ### Install the requirements
   Run `$ pip install -r requirements.txt`

2. ### Specify the data folder
   Specify the path folder in which you store the data by filling the lines of code where `HERE THE DATA FOLDER` appears. The links for downloading all the datasets are provided in the paper.
    
3. ### Preprocess the dataset
   In `preprocessing.py` you can choose which dataset(s) to preprocess. All the necessary structures will be created.

4. ### Train the (black-box) model
   Run `matrix_factorization/main.py` to train the model that provides the relevance scores.

5. ### (Train DGREC)
   It is an optional step. In case you want to use this competitor in the exploration process, run `strategy/DGRec/main.py` to train the model. 

6. ### Launch the exploration process
   Run `strategy/main.py` to start the exploration process.
   You can specify the dataset(s) to adopt, the number of expected steps, the length of the recommendation list, and the strategy(-ies) to exploit.
   By default, the number of steps is \[5, 10\], the length of the list is equal to 10, and the strategies are the ones reported in the paper (competitors included).

  

