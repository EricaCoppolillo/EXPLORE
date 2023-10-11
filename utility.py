import os
import pickle
import time

import numpy as np
import pandas as pd

from operator import itemgetter
from itertools import chain

from tqdm import tqdm


def get_dictionaries(ratings, outputs_folder, dataset_name, create=False):
    users_dictionary_path = os.path.join(outputs_folder, f"users_dictionary_{dataset_name}.pkl")
    items_dictionary_path = os.path.join(outputs_folder, f"items_dictionary_{dataset_name}.pkl")

    if not os.path.exists(users_dictionary_path) or create:
        print("Computing users/items dictionaries...")

        users_dict, items_dict = {}, {}

        user_count, item_count = 0, 0

        for (user, item, _) in ratings:

            if user not in users_dict:
                users_dict[user] = user_count
                user_count += 1

            if item not in items_dict:
                items_dict[item] = item_count
                item_count += 1

        # save dictionaries
        with open(users_dictionary_path, 'wb') as f, open(items_dictionary_path, 'wb') as g:
            pickle.dump(users_dict, f)
            pickle.dump(items_dict, g)

    else:

        print("Loading users/items dictionaries.")
        with open(users_dictionary_path, 'rb') as f, open(items_dictionary_path, 'rb') as g:
            users_dict = pickle.load(f)
            items_dict = pickle.load(g)

    return users_dict, items_dict


def get_structures(dataset_name, jaccard_distance, folder):
    jaccard_distances_fn = f"jaccard_{jaccard_distance}_distances_{dataset_name}.npy"

    users_dictionary_fn = f"users_dictionary_{dataset_name}.pkl"
    items_dictionary_fn = f"items_dictionary_{dataset_name}.pkl"

    files = [jaccard_distances_fn, users_dictionary_fn,
             items_dictionary_fn]  # [items_items_distances_fn, users_items_distances_fn, items_embeddings_fn]
    for f in files:
        path = os.path.join(folder, f)
        if not os.path.exists(path):
            print(f"{path} does not exist. Run preprocessing.py first")
            exit(0)

    items_items_distances = np.load(os.path.join(folder, jaccard_distances_fn))

    with open(os.path.join(folder, users_dictionary_fn), "rb") as f, \
            open(os.path.join(folder, items_dictionary_fn), "rb") as g:
        users_dictionary = pickle.load(f)
        items_dictionary = pickle.load(g)

    return users_dictionary, items_dictionary, items_items_distances


def get_jaccard_matrix(items_matrix, jaccard_path, create=False):
    if create or not os.path.exists(jaccard_path):

        print("Computing jaccard distances matrix...")
        start = time.time()

        jaccard_distances_matrix = np.zeros((len(items_matrix), len(items_matrix)))

        for i, row in tqdm(enumerate(items_matrix), total=len(items_matrix)):
            jaccard_similarities = np.minimum(row, items_matrix).sum(axis=-1) / np.maximum(row, items_matrix).sum(
                axis=-1)
            jaccard_similarities = np.nan_to_num(jaccard_similarities)

            jaccard_distances = 1 - jaccard_similarities

            jaccard_distances_matrix[i] = jaccard_distances

        np.save(jaccard_path, jaccard_distances_matrix)

        print(f"Time: {time.time() - start}")
        print("Jaccard distances matrix saved.")

    else:
        print("Loading distances matrix...")
        jaccard_distances_matrix = np.load(jaccard_path)

    return jaccard_distances_matrix


def create_items_users_matrix(ratings, items_dictionary, users_dictionary, items_users_path):
    items_users_matrix = np.zeros((len(items_dictionary), len(users_dictionary)))

    for (user, item, rating) in ratings:
        reindexed_user = users_dictionary[user]
        reindexed_item = items_dictionary[item]

        items_users_matrix[reindexed_item, reindexed_user] = rating

    np.save(items_users_path, items_users_matrix)


def create_items_genres_matrix(items_dictionary, dataset_name, dataset_path, items_genres_path):
    if dataset_name == "movielens-1m":
        items_genres = pd.read_csv(os.path.join(dataset_path, "movies.dat"), sep="::", engine="python", header=None)[
            [0, 2]]

        genres = list(set(np.concatenate(items_genres[2].str.split("|").to_numpy())))
        genres_dict = dict(zip(genres, range(len(genres))))

        items_genres_matrix = np.zeros((len(items_dictionary), len(genres)))
        items_genres = items_genres.to_numpy()

        for item, item_genres in items_genres:
            item_genres = item_genres.split("|")
            item_genres_ids = itemgetter(*item_genres)(genres_dict)
            if item in items_dictionary:  # discarding not interacted items
                items_genres_matrix[items_dictionary[item], item_genres_ids] = 1

        genres_dict_path = f"outputs/genres_dict_{dataset_name}.pkl"
        with open(genres_dict_path, "wb") as f:
            pickle.dump(genres_dict, f)

    elif "KuaiRec-2.0" in dataset_name:
        items_genres = pd.read_csv(os.path.join(dataset_path, "item_categories.csv"))

        genres = items_genres["feat"].apply(eval).to_numpy()
        unique_genres = np.unique(np.concatenate(genres))

        items_genres_matrix = np.zeros((len(items_dictionary), len(unique_genres)))

        items_genres = list(zip(items_genres["video_id"], genres))

        for item, genres in items_genres:
            if item in items_dictionary:
                items_genres_matrix[items_dictionary[item], genres] = 1

    elif dataset_name == "coat":
        with open(os.path.join(dataset_path, "user_item_features/item_features.ascii"), "r") as f:
            items_genres_matrix = np.loadtxt(f)

    elif dataset_name == "yahoo-r2":
        song_genres = pd.read_csv(os.path.join(dataset_path, "song-attributes.txt"), sep="\t",
                                  names=["itemId", "albumId", "artistId", "genreId"])
        n_song_genres = song_genres["genreId"].unique().size
        items_genres_matrix = np.zeros((len(items_dictionary), n_song_genres))

        genres_dict = dict(zip(song_genres["genreId"].unique(), range(n_song_genres)))
        filtered_song_genres = song_genres[song_genres["itemId"].isin(items_dictionary)]
        reindexed_items = itemgetter(*filtered_song_genres["itemId"].to_numpy())(items_dictionary)
        reindexed_genres = itemgetter(*filtered_song_genres["genreId"].to_numpy())(genres_dict)

        items_genres_matrix[reindexed_items, reindexed_genres] = 1

    elif dataset_name == "reviews-alaska":
        df_items_genres = pd.read_json(os.path.join(dataset_path, "meta-Alaska.json"), lines=True)[
            ["gmap_id", "category"]]
        df_items_genres.dropna(inplace=True)

        genres = list(set(chain.from_iterable(df_items_genres["category"].to_list())))
        genres_dict = dict(zip(genres, range(len(genres))))

        items_genres_matrix = np.zeros((len(items_dictionary), len(genres)))
        items_genres = df_items_genres.to_numpy()

        for item, item_genres in items_genres:
            item_genres_ids = itemgetter(*item_genres)(genres_dict)
            if item in items_dictionary:  # discarding not interacted items
                items_genres_matrix[items_dictionary[item], item_genres_ids] = 1

    elif dataset_name == "netflix":
        df_items_genres = pd.read_csv(os.path.join(dataset_path, "netflix_genres.csv"))

        genres = list(set(np.concatenate(df_items_genres["genres"].str.split("|").to_numpy())))
        genres_dict = dict(zip(genres, range(len(genres))))

        items_genres_matrix = np.zeros((len(items_dictionary), len(genres)))
        items_genres = df_items_genres.to_numpy()

        for item, item_genres in items_genres:
            item_genres = item_genres.split("|")
            item_genres_ids = itemgetter(*item_genres)(genres_dict)
            if item in items_dictionary:  # discarding not interacted items
                items_genres_matrix[items_dictionary[item], item_genres_ids] = 1

        genres_dict_path = f"outputs/genres_dict_{dataset_name}.pkl"
        with open(genres_dict_path, "wb") as f:
            pickle.dump(genres_dict, f)


    else:
        print(f"No gender matrix for dataset {dataset_name}")
        return -1

    np.save(items_genres_path, items_genres_matrix)
    return 0


def get_items_users_matrix(ratings, items_dictionary, users_dictionary, items_users_path, create=False):
    if not os.path.exists(items_users_path) or create:
        create_items_users_matrix(ratings, items_dictionary, users_dictionary, items_users_path)

    items_users_matrix = np.load(items_users_path)
    return items_users_matrix


def get_items_genres_matrix(items_dictionary, dataset_name, dataset_path, items_genres_path, create=False):
    result = 0
    if not os.path.exists(items_genres_path) or create:
        result = create_items_genres_matrix(items_dictionary, dataset_name, dataset_path, items_genres_path)

    items_genres_matrix = None
    if result != -1:
        items_genres_matrix = np.load(items_genres_path)

    return items_genres_matrix


def sample_dataset_items(df, sample_column, sample_size=3000):
    np.random.seed(1)
    items_sampled = np.random.choice(df[sample_column].unique(), size=sample_size, replace=False)
    new_df = df[df[sample_column].isin(items_sampled)]

    return new_df


def scale_ratings(df, dataset_name):
    if dataset_name == "KuaiRec-2.0":
        watch_ratio = df["watch_ratio"]
        scaled_ratings = np.interp(watch_ratio, (0, 2), (1, 5)).round().astype(int)

        df["watch_ratio"] = scaled_ratings

    return df.to_numpy()


def get_ratings(dataset_name, dataset_folder):
    if dataset_name == "movielens-1m":
        ratings = np.genfromtxt(os.path.join(dataset_folder, 'ratings.dat'),
                                skip_header=0,
                                skip_footer=1,
                                dtype=None,
                                delimiter='::')[:, :-1]

    elif "KuaiRec-2.0" in dataset_name:
        name, matrix = dataset_name.split("_")

        dataset_folder = os.path.join("HERE THE DATA FOLDER", name)
        dataset_folder = os.path.join(dataset_folder, "data")

        df = pd.read_csv(os.path.join(dataset_folder, f"{matrix}_matrix.csv"))[
            ["user_id", "video_id", "watch_ratio"]]
        ratings = scale_ratings(df, name)

    elif dataset_name == "coat":
        with open(os.path.join(dataset_folder, "train.ascii"), "r") as f:
            ratings_matrix = np.loadtxt(f)

        ratings = []
        for i in range(len(ratings_matrix)):
            user_row = ratings_matrix[i]
            user_ratings = user_row[user_row > 0]
            user_items = np.where(user_row > 0)[0]
            user = np.repeat(i, len(user_ratings))
            ratings.extend(np.array(list(zip(user, user_items, user_ratings))))

        ratings = np.array(ratings)

    elif dataset_name == "yahoo-r2":
        df = pd.read_csv(os.path.join(dataset_folder, "train_0.txt"), sep="\t", names=["userId", "itemId", "rating"])
        df = sample_dataset_items(df, sample_column="itemId", sample_size=3000)
        filtered_df = df.groupby("userId").filter(lambda x: len(x) >= 20)
        ratings = filtered_df.to_numpy()

    elif dataset_name == "netflix":
        df_items_with_genres = pd.read_csv(os.path.join(dataset_folder, "netflix_genres.csv"))["movieId"].to_list()
        df = pd.read_csv(os.path.join(dataset_folder, "Netflix_Dataset_Rating.csv"))[["User_ID", "Movie_ID", "Rating"]]
        df = df[df["Movie_ID"].isin(df_items_with_genres)]
        df = sample_dataset_items(df, sample_column="User_ID", sample_size=5000)
        filtered_df = df.groupby("User_ID").filter(lambda x: len(x) >= 20)
        ratings = filtered_df.to_numpy()

    else:
        print(f"Dataset {dataset_name} not preprocessed yet")
        exit(0)

    return ratings


def print_matrix_statistics(matrix):
    print("Mean:", np.mean(matrix))
    print("Median:", np.median(matrix))
    print("Min:", np.min(matrix))
    print("Max:", np.max(matrix))
