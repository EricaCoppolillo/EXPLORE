import pandas as pd
from joblib import Parallel, delayed
import time
import os
import sys

from utils import split_users_train_test, instantiate_models
from strategies import *

parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

sys.path.append(parent)

from utility import get_structures
from results import *
from matrix_factorization.model import MatrixFactorization


def main(dataset_name):

    jaccard_distances_dict = {"movielens-1m": "genres", "KuaiRec-2.0_small": "users",
                              "coat": "genres", "yahoo-r2": "genres", "netflix": "genres"}

    jaccard_distance = jaccard_distances_dict[dataset_name]

    print(f"Dataset: {dataset_name}")
    print(f"Adopted jaccard distance: {jaccard_distance}")

    # HERE THE OUTPUTS FOLDER
    folder = f"../outputs/jaccard_{jaccard_distance}"

    if not os.path.exists(folder):
        os.makedirs(folder)

    model_folder = "../models/checkpoints/"

    users_dictionary, items_dictionary, items_items_distances = get_structures(dataset_name,
                                                                               jaccard_distance=jaccard_distance,
                                                                               folder="../outputs")

    items_genres_path = f"../outputs/items_genres_matrix_{dataset_name}.npy"
    items_genres_matrix = get_items_genres_matrix(items_dictionary, dataset_name, None, items_genres_path)

    users, items = list(users_dictionary.values()), list(items_dictionary.values())
    n_users, n_items = len(users), len(items)

    if items_genres_matrix is not None:
        n_genres = items_genres_matrix.shape[1]

    device_id = 1

    if torch.cuda.is_available():
        device = f"cuda:{device_id}"
        torch.cuda.set_device(device_id)
    else:
        device = "cpu"

    print(f"Device: {device}")

    model_path = os.path.join(model_folder, f"matrix_factorization_{dataset_name}.pth")
    if not os.path.exists(model_path):
        print(f"{model_path} does not exist. Run Matrix Factorization main first")
        exit(0)

    n_factors_dict = {"movielens-1m": 10, "KuaiRec-2.0_small": 10, "coat": 1, "yahoo-r2": 5, "netflix": 5}
    model = MatrixFactorization(n_users, n_items, n_factors=n_factors_dict[dataset_name], device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    users = np.array(users)

    clip_min, clip_max = 1, 5

    default_n_jobs = 10
    trials = 20

    coverage_threshold = 0

    strategy = "dgrec"
    strategies = [
        "relevance",
        "k_means_with_relevance",
        # "k_means_without_relevance",
        "coverage_with_relevance",
        # "coverage_without_relevance",
        "mmr",
        "dpp",
        "dum",
        "dgrec"
    ]

    competitors = ["relevance", "mmr", "dpp", "dum", "dgrec"]
    our_strategies_with_relevance = ["k_means_with_relevance", "coverage_with_relevance"]

    # for the ablation study
    our_strategies_without_relevance = ["k_means_without_relevance", "coverage_without_relevance"]

    copula = "clayton"

    USE_WEIBULL = True

    DEBUG = False
    COMPUTE_EXPECTATION = False

    EVALUATE = False
    TAKE_TIME = False
    TUNE_ALPHA = False

    DEFUALT_DETERMINISTIC = True

    LOAD_DICT = True
    OVERWRITE_DICT = True

    UPDATE_ALL_STRATEGIES = True

    UPDATE_ALL_KS = False
    UPDATE_ALL_USERS_BUDGETS = False
    UPDATE_ALL_GS = True

    ks = [10]
    users_budgets = [10, 20, 40]

    if USE_WEIBULL:
        users_budgets = [5, 10]
        default_gammas = [2]

    k = 10
    users_budget = 20
    g = 2

    lambdas_dict = {5: 6.2, 10: 11.85, 20: 23.21}

    if DEBUG:
        LOAD_DICT = False
        OVERWRITE_DICT = False

        UPDATE_ALL_STRATEGIES = True

        UPDATE_ALL_KS = True
        UPDATE_ALL_USERS_BUDGETS = True
        UPDATE_ALL_GS = True

        trials = 20

    if copula == "clayton":
        default_alphas = [0.0001, 0.001, 0.01, 0.5, 0.1, 1, 2, 3, 4]  # current experiments are with alpha=0.5

    if EVALUATE or TAKE_TIME:
        UPDATE_ALL_KS = False
        UPDATE_ALL_USERS_BUDGETS = False
        k = 10
        if EVALUATE:
            users_budget = 1  # does not matter
        else:
            if USE_WEIBULL:
                users_budget = 5
                g = 2
            else:
                users_budget = 10

    if not UPDATE_ALL_STRATEGIES:
        strategies = [strategy]
    if not UPDATE_ALL_KS:
        ks = [k]
    if not UPDATE_ALL_USERS_BUDGETS:
        users_budgets = [users_budget]
    if not UPDATE_ALL_GS:
        default_gammas = [g]

    # EVALUATION
    users_train, users_test = None, None

    if EVALUATE:
        users_train, users_test = split_users_train_test(dataset_name)

    models_dict = instantiate_models(strategies, n_users, n_items, items_items_distances, items_genres_matrix, model,
                                     device, clip_min, clip_max, EVALUATE, users_test, dataset_name, jaccard_distance)

    def process(t):

        r_users = np.zeros(n_users)
        coverage_users = np.zeros(n_users)
        users_steps = np.zeros(n_users)
        user_items_distances = []  # saving the referring user does not matter

        print(f"Trial {t}")

        print("Starting trial...")
        start_trial = time.time()

        users_items_matrix = np.zeros((n_users, n_items))

        users_quitting_vector = np.zeros(len(users))

        while 0 in users_quitting_vector:

            active_users = users[users_quitting_vector[users] == 0]

            actual_model = models_dict[strategy]

            if TAKE_TIME:
                start_strategy = time.time()

            if strategy in our_strategies_with_relevance:
                final_users_items = actual_model.get_recommendations(active_users, k, users_items_matrix, alpha, copula,
                                                                     use_relevance=True)
            elif strategy in our_strategies_without_relevance:
                final_users_items = actual_model.get_recommendations(active_users, k, users_items_matrix, None, None,
                                                                     use_relevance=False)
            else:
                final_users_items = actual_model.get_recommendations(active_users, k, users_items_matrix)

            if TAKE_TIME:
                end_strategy = time.time() - start_strategy
                return end_strategy

            # EVALUATION (HitRate, Precision and Recall)
            if EVALUATE:
                hit_ratios, precisions, recalls = compute_evaluations(users_test, final_users_items)
                return hit_ratios, precisions, recalls

            tensor_final_users_items = torch.tensor(final_users_items, dtype=torch.int, device=device)

            ratings = model(torch.tensor(active_users, device=device).unsqueeze(-1),
                            tensor_final_users_items).detach().cpu().numpy().clip(clip_min, clip_max)

            p = np.interp(ratings, (clip_min, clip_max), (0, 1))

            if USE_WEIBULL:
                l = lambdas_dict[users_budget]
                q = np.exp(-1/(l**g))
                # \eta_k = 1 − 𝑞(𝑘 + 1)^(𝛾 −𝑘^𝛾)
                u_budget = 1 - q**((users_steps[active_users] + 1)**g - users_steps[active_users]**g)
            else:
                u_budget = users_steps[active_users] / users_budget

            theta = np.repeat(1 - u_budget, k).reshape(len(active_users), k)
            # for now theta is constant in the examination of the recommendation list

            quitting_probability = np.full(len(active_users), 1 - theta[:, 0])

            if not COMPUTE_EXPECTATION:
                for i in range(k - 1):
                    quitting_probability += np.prod(theta[:, :i + 1], axis=-1) * np.prod(1 - p[:, :i + 1], axis=-1) * (
                            1 - theta[:, i + 1])

            quitting_bernoulli = np.random.binomial(1, p=quitting_probability)

            p_u = ratings / np.sum(ratings, axis=-1, keepdims=True)

            for i in range(len(active_users)):

                if quitting_bernoulli[i]:  # that user quits
                    users_quitting_vector[active_users[i]] = 1
                else:
                    user_choice = np.random.choice(final_users_items[i], p=p_u[i])
                    users_items_matrix[active_users[i]][user_choice] = 1
                    users_steps[active_users[i]] += 1

        end_trial = time.time() - start_trial
        print(f"Trial ended in {end_trial}")

        users_interactions = np.argwhere(users_items_matrix)
        df = pd.DataFrame(users_interactions, columns=["user", "item"])

        users_interactions = df.groupby("user").agg(list)["item"]

        for u in users_interactions.index:
            user_interactions = users_interactions[u]
            x = np.repeat(user_interactions, len(user_interactions))
            y = np.tile(user_interactions, len(user_interactions))

            # saving all the distances between couples of items
            distances = items_items_distances[x, y]

            norm = users_steps[u] - 1

            if norm > 0:  # at least 2 steps
                r_users[u] = distances.sum() / norm  # users r score of each trial
            else:
                r_users[u] = 0.

            if items_genres_matrix is not None:
                coverage_user_genres = len(
                    np.where(items_genres_matrix[user_interactions].sum(0) > coverage_threshold)[0])
                coverage_users[u] = coverage_user_genres / n_genres
            else:
                coverage_users[u] = 0

        return r_users, coverage_users, users_steps, user_items_distances

    # SAVING by:
    # - outputs
    # -- distance criterion
    # --- k
    # ---- users budget

    for k in ks:

        for users_budget in users_budgets:

            if USE_WEIBULL:
                gammas = default_gammas
            else:
                gammas = [None]

            for g in gammas:

                for strategy in strategies:

                    if strategy == "dgrec" and (dataset_name == "KuaiRec-2.0_small" or dataset_name == "netflix"):
                        n_jobs = 1
                    else:
                        n_jobs = default_n_jobs

                    if strategy in competitors:
                        deterministic = False
                    else:
                        deterministic = DEFUALT_DETERMINISTIC

                    alphas = [None]

                    if strategy in our_strategies_with_relevance:
                        if TUNE_ALPHA:
                            alphas = default_alphas
                        else:
                            alphas = [0.5]

                    for alpha in alphas:

                        print("-----------------------")
                        print(f"Dataset={dataset_name}")
                        print(f"Adopting k={k}")
                        print(f"Use Weibull={USE_WEIBULL}")
                        print(f"Budget={users_budget}")
                        print(f"Gamma={g}")
                        print(f"Deterministic={deterministic}")
                        print(f"Recomendation strategy={strategy}")
                        print(f"Copula={copula}")
                        print(f"Alpha={alpha}")

                        if not EVALUATE:  # launch simulation

                            final_folder = os.path.join(folder, f"k_{k}", f"users_budget_{users_budget}")

                            if deterministic:
                                final_folder += "_deterministic/"

                            if strategy in competitors:
                                s = "competitors"
                                final_folder += "_competitors/"
                            else:
                                s = "strategies"
                                if strategy in our_strategies_with_relevance:
                                    if copula is not None:
                                        final_folder += f"copula_{copula}/alpha_{alpha}/"
                                    else:
                                        final_folder += "no_copula/"
                                elif strategy in our_strategies_without_relevance:
                                    final_folder += "no_relevance/"

                            if g is not None:
                                final_folder += f"gamma_{g}/"

                            if TAKE_TIME:
                                fn = s + f"_time_{dataset_name}.pkl"
                            else:
                                fn = s + f"_scores_{dataset_name}.pkl"

                            if not os.path.exists(final_folder):
                                os.makedirs(final_folder)

                            path = os.path.join(final_folder, fn)

                            print("*******\nStart of the simulation\n*******")
                            start_simulation = time.time()
                            results = Parallel(n_jobs=n_jobs, prefer="processes")(
                                delayed(process)(t) for t in range(trials))
                            print(f"*******\nEnd of the simulation in {time.time() - start_simulation}\n*******")

                            save_results(results, strategy, path, LOAD_DICT, OVERWRITE_DICT, TAKE_TIME)

                        else:  # just evaluate

                            print("*******\nStart of the evaluation\n*******")
                            evaluations = Parallel(n_jobs=n_jobs, prefer="processes")(
                                delayed(process)(t) for t in range(trials))
                            print(f"*******\nEnd of the evaluation\n*******")

                            save_evaluations(evaluations, strategy, competitors, k, users_budget, deterministic,
                                             dataset_name)

                        print("-----------------------")


if __name__ == '__main__':

    datasets = ["coat", "netflix", "movielens-1m", "yahoo-r2", "KuaiRec-2.0_small"]
    datasets = ["KuaiRec-2.0_small"]

    for dataset in datasets:
        main(dataset)
