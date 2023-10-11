import pickle

from strategies import *
from competitors import *
from DGRec.models.models import *
from DGRec.utils.parser import *
from DGRec.utils.utils import *
from DGRec.utils.dataloader import *


def split_users_train_test(dataset_name):
    dataset_folder = f"HERE THE DATA FOLDER"

    actual_ratings = get_ratings(dataset_name, dataset_folder)

    items_dictionary_path = f"../outputs/items_dictionary_{dataset_name}.pkl"
    users_dictionary_path = f"../outputs/users_dictionary_{dataset_name}.pkl"

    with open(items_dictionary_path, "rb") as f, open(users_dictionary_path, "rb") as g:
        items_dictionary = pickle.load(f)
        users_dictionary = pickle.load(g)

    ratings_dataset = []

    for (user, item, rating) in actual_ratings:
        reindexed_user = users_dictionary[user]
        reindexed_item = items_dictionary[item]

        ratings_dataset.append([reindexed_user, reindexed_item, rating])

    df = pd.DataFrame(ratings_dataset, columns=["user", "item", "rating"])

    train_data, test_data = sklearn.model_selection.train_test_split(df, test_size=0.2,
                                                                     stratify=df["user"], random_state=0)

    return train_data, test_data


def instantiate_models(strategies, n_users, n_items, items_items_distances, items_genres_matrix, model, device,
                       clip_min, clip_max, evaluate, users_test, dataset_name, jaccard_distance):
    models_dict = {}

    # STRATEGIES
    if "relevance" in strategies:
        relevance = Relevance(n_users, n_items, model, items_items_distances, device, clip_min, clip_max, evaluate,
                              users_test)
        models_dict["relevance"] = relevance
    if "k_means_with_relevance" in strategies or "k_means_without_relevance" in strategies:
        interacted_distance = InteractedDistance(n_users, n_items, model, items_items_distances, device, clip_min,
                                                 clip_max, evaluate,
                                                 users_test)
        models_dict["k_means_with_relevance"] = interacted_distance
        models_dict["k_means_without_relevance"] = interacted_distance
    if "coverage_with_relevance" in strategies or "coverage_without_relevance" in strategies:
        coverage = InteractedCoverage(items_genres_matrix, n_users, n_items, model, items_items_distances, device,
                                      clip_min, clip_max, evaluate,
                                      users_test)
        models_dict["coverage_with_relevance"] = coverage
        models_dict["coverage_without_relevance"] = coverage

    # COMPETITORS
    if "mmr" in strategies:
        lambda_factor = 0.5
        mmr = MMR(lambda_factor, n_users, n_items, model, items_items_distances, device, clip_min, clip_max, evaluate,
                  users_test)
        models_dict["mmr"] = mmr
    if "dpp" in strategies:
        dpp = DPP(n_users, n_items, model, items_items_distances, device, clip_min, clip_max, evaluate,
                  users_test)
        models_dict["dpp"] = dpp
    if "dum" in strategies:
        dum = DUM(items_genres_matrix, n_users, n_items, model, items_items_distances, device, clip_min, clip_max,
                  evaluate,
                  users_test)
        models_dict["dum"] = dum
    if "dgrec" in strategies:
        args = parse_args()
        args.dataset = dataset_name
        args.distance = jaccard_distance
        args.gpu = device
        args.folder = "../outputs"
        dataloader = Dataloader(args, dataset_name, torch.device(device))
        dgrec = DGRec(args, dataloader).to(device)
        early_stop = config(args)
        dgrec.load_state_dict(torch.load("DGRec/" + early_stop.save_path, map_location=device))
        models_dict["dgrec"] = dgrec

    return models_dict
