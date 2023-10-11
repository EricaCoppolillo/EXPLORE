import os

import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection

import torch
from torch.utils.data import DataLoader

from model import MatrixFactorization, train, test

import sys
sys.path.append("../")

from utility import get_ratings, get_dictionaries
from visualization import plot_loss
from results import compute_evaluations


def main():
    dataset_name = "HERE THE DATASET NAME IN [movielens-1m, coat, KuaiRec-2.0_small, netflix, yahoo-r2]"
    dataset_folder = "HERE THE DATA FOLDER"

    ratings = get_ratings(dataset_name, dataset_folder)

    users_dictionary, items_dictionary = get_dictionaries(ratings, "../outputs", dataset_name)

    ratings_dataset = []

    print("Creating ratings dataset...")

    for (user, item, rating) in ratings:
        reindexed_user = users_dictionary[user]
        reindexed_item = items_dictionary[item]

        ratings_dataset.append([reindexed_user, reindexed_item, rating])

    print(f"Ratings Length: {len(ratings_dataset)}")

    TRAIN = False

    device_id = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.cuda.set_device(device_id)

    batch_sizes_dict = {"movielens-1m": 1024, "KuaiRec-2.0_small": 4096, "coat": 16,
                        "yahoo-r2": 2048, "netflix": 1024}
    batch_size = batch_sizes_dict[dataset_name]

    df = pd.DataFrame(ratings_dataset, columns=["user", "item", "rating"])
    train_data, test_data = sklearn.model_selection.train_test_split(ratings_dataset, test_size=0.2,
                                                                     stratify=df["user"],
                                                                     random_state=0)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    n_users, n_items = len(users_dictionary), len(items_dictionary)

    print(f"Users: {n_users}\nItems: {n_items}")

    n_factors_dict = {"movielens-1m": 10, "KuaiRec-2.0_small": 10, "coat": 1,
                      "yahoo-r2": 5, "netflix": 5}
    model = MatrixFactorization(n_users, n_items, n_factors=n_factors_dict[dataset_name], device=device)
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

    model_folder = f"../models/checkpoints/"
    outputs_folder = f"../outputs/"
    images_folder = outputs_folder + "images/"

    model_path = os.path.join(model_folder, f"matrix_factorization_{dataset_name}.pth")

    if TRAIN:

        print("Starting training...")

        train_loss, test_loss = [], []

        best_test_loss = np.inf

        count_worst_loss = 0
        count_for_break = 5

        n_epochs = 100

        for epoch in range(n_epochs):

            train_epoch_loss = train(model, device, train_loader, epoch, optimizer, criterion)
            test_epoch_loss = test(model, device, test_loader, criterion)

            train_loss.append(train_epoch_loss)
            test_loss.append(test_epoch_loss)

            if test_epoch_loss < best_test_loss:
                best_test_loss = test_epoch_loss
                count_worst_loss = 0
                print("Best model saved")
                torch.save(model.state_dict(), model_path)
            else:
                count_worst_loss += 1

            if count_worst_loss == count_for_break:
                break

        plot_loss(train_loss, test_loss, images_folder, f"matrix_factorization_loss_{dataset_name}.png")

    else:
        print("Loading model...")
        model.load_state_dict(torch.load(model_path))

    test_data = pd.DataFrame(test_data, columns=["user", "item", "rating"])

    users = test_data["user"].to_numpy()
    items = test_data["item"].to_numpy()
    ratings = test_data["rating"].to_numpy()

    preds = model(torch.tensor(users, device=device, dtype=torch.int), torch.tensor(items, device=device, dtype=torch.int)).detach().cpu().numpy()
    mse = np.square(preds - ratings).mean()

    test_data["preds"] = preds

    sorted_test_data = test_data.sort_values(by=["preds"], ascending=False)

    k = 10

    grouped = sorted_test_data.groupby("user")

    final_users_recommendations = []
    for user, grouped in grouped:
        items = grouped["item"][:k]
        final_users_recommendations.append(items)

    hit_ratios, precisions, recalls = compute_evaluations(test_data, final_users_recommendations)

    print(f"MSE:", mse)
    print(f"HR@{k}: {np.mean(hit_ratios)}")
    print(f"Precision@{k}: {np.mean(precisions)}")
    print(f"Recall@{k}: {np.mean(recalls)}")

if __name__ == '__main__':
    main()
