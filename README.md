# EXPLORE

This is the official repository of our paper "**Tackling Relevance and Diversity in Recommendations with a Novel User-Centric Framework**", submitted to the TKDD Journal.
The paper extends from the previous work "**Relevance Meets Diversity: A User-Centric Framework for Knowledge Exploration Through Recommendations**", accepted at KDD2024.

If you want to use the previous code, please switch to the "KDD2024" branch.

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
   Additionally, you can decide if allowing the user to refresh the recommendation list, which refreshing strategy adopting, and whether introducing a random factor when the list is constructed.

   By default, the number of steps is 5, the length of the list is equal to 10, refreshing the list is not allowed, and no random factor is considered.
   The strategies are the ones reported in the paper (competitors included).