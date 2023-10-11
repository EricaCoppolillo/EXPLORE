from joblib import Parallel, delayed

from utility import *


def main():

    def preprocess_dataset(dataset_name):
        print(f"DATASET: {dataset_name}")
        dataset_folder = os.path.join("HERE THE DATA FOLDER", dataset_name)

        outputs_folder = "outputs/"

        ratings = get_ratings(dataset_name, dataset_folder)

        users_dictionary, items_dictionary = get_dictionaries(ratings, outputs_folder, dataset_name)

        print("#Users", len(users_dictionary))
        print("#Items", len(items_dictionary))
        print("#Ratings", len(ratings))

        items_users_matrix_path = f"outputs/items_users_matrix_{dataset_name}.npy"

        items_users_matrix = get_items_users_matrix(ratings, items_dictionary, users_dictionary,
                                                    items_users_matrix_path)
        users_jaccard_path = f"outputs/jaccard_users_distances_{dataset_name}.npy"
        users_jaccard_matrix = get_jaccard_matrix(items_users_matrix, users_jaccard_path)

        print("Jaccard-Users Matrix Statistics:")
        print_matrix_statistics(users_jaccard_matrix)

        items_genres_matrix_path = f"outputs/items_genres_matrix_{dataset_name}.npy"

        items_genres_matrix = get_items_genres_matrix(items_dictionary, dataset_name, dataset_folder,
                                                      items_genres_matrix_path)

        genres_jaccard_path = f"outputs/jaccard_genres_distances_{dataset_name}.npy"
        genres_jaccard_matrix = get_jaccard_matrix(items_genres_matrix, genres_jaccard_path)

        print("Jaccard-Genres Matrix Statistics:")
        print_matrix_statistics(genres_jaccard_matrix)

    datasets = ["movielens-1m", "coat", "KuaiRec-2.0_small", "netflix", "yahoo-r2"]

    Parallel(n_jobs=len(datasets), prefer="processes")(delayed(preprocess_dataset)(d) for d in datasets)


if __name__ == '__main__':
    main()