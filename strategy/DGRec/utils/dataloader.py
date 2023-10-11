from collections import defaultdict

import pandas as pd
import sklearn.model_selection
import pdb
import torch
import logging
import numpy as np
import dgl
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import coo_matrix, csr_matrix

import sys

sys.path.append("../..")
from utility import get_items_genres_matrix, get_ratings, get_dictionaries


class TestDataset(Dataset):
    def __init__(self, dic):
        self.keys = torch.tensor(list(dic.keys()), dtype=torch.long)
        ls_values = [tensor for tensor in dic.values()]
        self.values = csr_matrix(torch.stack(ls_values))

    def __getitem__(self, index):
        key = self.keys[index]
        values = self.values[index]
        return {'key': key, 'value': values}

    def __len__(self):
        return len(self.keys)


class Dataloader(object):
    def __init__(self, args, data, device):
        self.historical_dict = None
        self.folder = args.folder
        logging.info("loading data")
        self.args = args

        dataset_folder = f"/mnt/nas/coppolillo/Serendipity/datasets/{data}"
        self.ratings = get_ratings(data, dataset_folder)

        self.users_dictionary, self.items_dictionary = get_dictionaries(self.ratings, self.folder, data)

        self.user_number = 0
        self.item_number = 0
        self.device = device
        logging.info('reading category information')

        train_data, val_data, test_data = self.get_dataset()
        items_genres_matrix, self.categories_values, self.categories_num = self.get_categories(data, dataset_folder)

        logging.info('reading train data')
        self.train_graph, self.dataloader_train = self.read_train_graph(train_data)
        logging.info('reading valid data')
        self.val_graph, self.dataloader_val = self.read_val_graph(val_data)
        logging.info('reading test data')
        self.test_dic, self.dataloader_test = self.read_test(test_data)
        logging.info('get weight for each sample')
        self.sample_weight = self.get_sample_weight(items_genres_matrix)

    def get_dataset(self):

        ratings_dataset = []

        for (user, item, _) in self.ratings:
            reindexed_user = self.users_dictionary[user]
            reindexed_item = self.items_dictionary[item]

            ratings_dataset.append([reindexed_user, reindexed_item])

        df = pd.DataFrame(ratings_dataset, columns=["user", "item"])

        train_data, test_data = sklearn.model_selection.train_test_split(ratings_dataset, test_size=0.2,
                                                                         stratify=df["user"], random_state=0)

        df = pd.DataFrame(train_data, columns=["user", "item"])
        train_data, val_data = sklearn.model_selection.train_test_split(train_data, test_size=0.2,
                                                                        stratify=df["user"], random_state=0)

        return train_data, val_data, test_data

    def get_csr_matrix(self, array):
        users = array[:, 0]
        items = array[:, 1]
        data = np.ones(len(users))
        # return torch.sparse_coo_tensor(array.t(), data, dtype = bool).to_sparse_csr().to(args.device)
        return coo_matrix((data, (users, items)), shape=(self.user_number, self.item_number), dtype=bool).tocsr()

    def get_categories(self, data, dataset_folder):

        items_genres_matrix_path = f"{self.folder}/items_genres_matrix_{data}.npy"

        items_genres_matrix = get_items_genres_matrix(self.items_dictionary, data, dataset_folder,
                                                      items_genres_matrix_path)

        categories_num = items_genres_matrix.shape[1]
        categories_values = []
        for i in range(items_genres_matrix.shape[0]):
            row = items_genres_matrix[i]
            first_category = np.argwhere(row > 0)[0][0]
            categories_values.append(first_category)

        return items_genres_matrix, categories_values, categories_num

    def get_sample_weight(self, items_genres_matrix):

        weight_tensor = torch.tensor(list(items_genres_matrix.sum(axis=1)), dtype=torch.float)
        effective_num = 1.0 - torch.pow(self.args.beta_class, weight_tensor)
        weight = (1 - self.args.beta_class) / effective_num
        weight = weight / weight.sum() * self.categories_num

        return weight[torch.tensor(list(self.categories_values))]

    def stacking_layers(self, array, num):
        pdb.set_trace()
        count, _ = array.shape
        data = np.ones(count)

        user2item = torch.sparse_coo_tensor(array.t(), data).to(self.args.device)
        item2user = user2item.t()
        trans = torch.sparse.mm(item2user, user2item)

        res = user2item
        for i in range(num):
            res = torch.sparse.mm(res, trans)

        return array

    def read_train_graph(self, train_data):
        self.historical_dict = defaultdict(set)

        for (user, item) in train_data:
            if user in self.historical_dict:
                self.historical_dict[user].add(item)
            else:
                self.historical_dict[user] = {item}

        train_data = torch.tensor(train_data)
        self.user_number = max(self.user_number, train_data[:, 0].max() + 1)
        self.item_number = max(self.item_number, train_data[:, 1].max() + 1)
        self.train_csr = self.get_csr_matrix(train_data)

        # train_data = self.stacking_layers(train_data, 1)

        graph_data = {
            ('user', 'rate', 'item'): (train_data[:, 0].long(), train_data[:, 1].long()),
            ('item', 'rated by', 'user'): (train_data[:, 1].long(), train_data[:, 0].long())
        }
        graph = dgl.heterograph(graph_data)
        category_tensor = torch.tensor(list(self.categories_values), dtype=torch.long).unsqueeze(1)
        graph.ndata['category'] = {'item': category_tensor, 'user': torch.zeros(self.user_number, 1) - 1}
        dataset = torch.utils.data.TensorDataset(train_data)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)

        return graph.to(self.device), dataloader

    def read_val_graph(self, val_data):

        val_data = torch.tensor(val_data)

        graph_data = {
            ('user', 'rate', 'item'): (val_data[:, 0].long(), val_data[:, 1].long()),
            ('item', 'rated by', 'user'): (val_data[:, 1].long(), val_data[:, 0].long())
        }
        number_nodes_dict = {'user': self.user_number, 'item': self.item_number}
        graph = dgl.heterograph(graph_data, num_nodes_dict=number_nodes_dict)

        dataset = torch.utils.data.TensorDataset(val_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

        return graph.to(self.device), dataloader

    def read_test(self, test_data):
        dic_test = {}
        for (user, item) in test_data:
            if user in dic_test:
                dic_test[user].append(item)
            else:
                dic_test[user] = [item]

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(list(dic_test.keys()), dtype=torch.long, device=self.device))

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)
        return dic_test, dataloader
        # test_data = torch.tensor(test_data)
        #
        # dataloader = torch.utils.data.DataLoader(test_data, batch_size=self.args.batch_size, shuffle=False)
        # return test_data, dataloader
