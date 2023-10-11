import os
import pickle
from collections import defaultdict

import numpy as np
from math import gamma

from utility import get_jaccard_matrix, get_items_genres_matrix


def save_results(results, strategy, strategies_scores_path, load_dict, overwrite_dict, take_time):
    strategies_dict = defaultdict(list)

    if load_dict and os.path.exists(strategies_scores_path):
        with open(strategies_scores_path, "rb") as f:
            strategies_dict = pickle.load(f)

    results = np.array(results, dtype="object")

    if take_time:
        strategies_dict[strategy] = results.mean()
    else:
        r_users = results[:, 0]
        coverage_users = results[:, 1]
        users_steps = results[:, 2]

        final_avg_r_scores = r_users.mean(axis=0)  # same user, different trials
        final_avg_coverage_scores = coverage_users.mean(axis=0)
        final_avg_users_steps = users_steps.mean(axis=0)

        final_avg_r_score = final_avg_r_scores.mean()
        final_avg_coverage_score = final_avg_coverage_scores.mean()
        final_avg_steps = final_avg_users_steps.mean()

        print("Mean Final avg users distance score:", final_avg_r_score)
        print("Mean Final avg users coverage score:", final_avg_coverage_score)
        print("Mean Final avg steps:", final_avg_steps)

        # Update strategies dictionary
        strategies_dict[strategy] = [final_avg_r_score, final_avg_r_scores,
                                     final_avg_coverage_score, final_avg_coverage_scores,
                                     final_avg_steps, final_avg_users_steps]

    if overwrite_dict:
        with open(strategies_scores_path, "wb") as f:
            pickle.dump(strategies_dict, f)


def save_evaluations(evaluations, strategy, competitors, k, users_budget, deterministic,
                     dataset_name):
    final_folder = f"../outputs/evaluation/k_{k}/users_budget_{users_budget}"
    if deterministic:
        final_folder += "_deterministic/"

    if not os.path.exists(final_folder):
        os.makedirs(final_folder)

    evaluations = np.array(evaluations, dtype="object")

    hit_ratios = evaluations[:, 0]
    precisions = evaluations[:, 1]
    recalls = evaluations[:, 2]

    final_avg_hit_ratios = hit_ratios.mean(axis=0)
    final_avg_precisions = precisions.mean(axis=0)
    final_avg_recalls = recalls.mean(axis=0)

    final_avg_hit_ratio = final_avg_hit_ratios.mean()
    final_avg_precision = final_avg_precisions.mean()
    final_avg_recall = final_avg_recalls.mean()

    metrics_dict = {"hit_ratio": final_avg_hit_ratio,
                    "precision": final_avg_precision,
                    "recall": final_avg_recall}

    for metric, results in metrics_dict.items():

        s = "strategies"
        if strategy in competitors:
            s = "competitors"

        fn = f"{s}_{metric}_{dataset_name}.pkl"
        dict_path = os.path.join(final_folder, fn)

        if os.path.exists(dict_path):
            with open(dict_path, "rb") as f:
                results_dict = pickle.load(f)
        else:
            results_dict = {}

        results_dict[strategy] = results

        print(f"{metric}@{k}: {results}")

        with open(dict_path, "wb") as f:
            pickle.dump(results_dict, f)


def compute_evaluations(users_test, final_users_recommendations):
    threshold = 3.5

    hit_ratios, precisions, recalls = [], [], []

    grouped = users_test.groupby("user")
    for user, group in grouped:

        actual_ratings = group["rating"].to_numpy()
        positive = group["item"][actual_ratings >= threshold].to_numpy()

        recommended = list(set(final_users_recommendations[user]))

        positive_recommended = set(positive).intersection(set(recommended))

        if not len(recommended):
            precision = 0
        else:
            precision = len(positive_recommended) / len(recommended)

        if not len(positive):
            recall = 0
        else:
            recall = len(positive_recommended) / len(positive)

        if len(positive_recommended) > 0:
            hit_ratio = 1
        else:
            hit_ratio = 0

        hit_ratios.append(hit_ratio)
        precisions.append(precision)
        recalls.append(recall)

    return hit_ratios, precisions, recalls


def compute_u_budget(i, g, l):
    l += 0.56
    q = np.exp(-1 / (l ** g))

    return 1 - q ** ((i + 1) ** g - i ** g)


def compute_diversity_upper_bound(dataset_name, jaccard_distance, use_weibull=True):
    jaccard_path = f"outputs/jaccard_{jaccard_distance}_distances_{dataset_name}.npy"
    matrix = get_jaccard_matrix(None, jaccard_path)

    flatten = matrix.flatten()
    vector = np.sort(flatten)[::-1]

    saving_path = f"outputs/"
    if use_weibull:
        saving_path += "weibull/"

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    saving_path += f"diversity_upper_bound_{dataset_name}.txt"

    with open(saving_path, "w") as f:
        f.write(f"Budget\tDiversity Upper Bound\n")

    print(f"Computing Diversity Upper Bounds for {dataset_name}")

    if use_weibull:
        Bs = [5, 10]
        g = 2
        epsilon = 1e-3
        s = vector

        for B in Bs:
            l = B / gamma(1 + 1 / g)
            i = 1
            stop = False
            d = 0

            while not stop:
                u_budget = compute_u_budget(i, g, l)
                x = np.prod([1 - compute_u_budget(j, g, l) for j in range(1, i)])

                if i == 1:
                    max_i = 0
                else:
                    max_i = 1 / (i - 1) * sum(s[:i * (i - 1)])

                d_old = d
                d = d_old + (max_i * u_budget * x)

                if i > 1 and abs(d - d_old) < epsilon:
                    stop = True

                i += 1

            with open(saving_path, "a") as f:
                f.write(f"{B}\t{d}\n")
    else:
        Bs = [10, 20, 40]

        for B in Bs:
            s = vector[:B * (B - 1) + 1]
            d = 0
            for i in range(1, B + 1):
                x = np.prod([1 - j / B for j in range(1, i)])
                if i == 1:
                    max_i = 0
                else:
                    max_i = 1 / (i - 1) * sum(s[:i * (i - 1)])
                d += max_i * i / B * x

            with open(saving_path, "a") as f:
                f.write(f"{B}\t{d}\n")


def compute_coverage_upper_bound(dataset_name, use_weibull=True):
    items_genres_matrix_path = f"outputs/items_genres_matrix_{dataset_name}.npy"

    items_genres_matrix = get_items_genres_matrix(None, dataset_name, None, items_genres_matrix_path)

    saving_path = f"outputs/"

    if use_weibull:
        saving_path += "weibull/"

    saving_path += f"coverage_upper_bound_{dataset_name}.txt"

    with open(saving_path, "w") as f:
        f.write(f"Budget\tCoverage Upper Bound\n")

    n_genres = items_genres_matrix.shape[1]

    print(f"Computing Coverage Upper Bounds for {dataset_name}")

    if use_weibull:
        Bs = [5, 10]
        g = 2
        epsilon = 1e-3

        for B in Bs:
            d = 0
            best_coverage_vector = None

            l = B / gamma(1 + 1 / g)
            i = 1
            stop = False

            while not stop:
                u_budget = compute_u_budget(i, g, l)
                x = np.prod([1 - compute_u_budget(j, g, l) for j in range(1, i)])

                if i == 1:
                    max_i = (items_genres_matrix > 0).astype(int).sum(axis=1).max()
                else:
                    max_i, best_coverage_vector = compute_maximum_coverage(items_genres_matrix, best_coverage_vector)

                d_old = d
                d = d_old + (max_i * u_budget * x)

                if i > 1 and abs(d - d_old) < epsilon:
                    stop = True

                i += 1

            d /= n_genres

            with open(saving_path, "a") as f:
                f.write(f"{B}\t{d}\n")
    else:
        Bs = [10, 20, 40]

        for B in Bs:
            d = 0
            best_coverage_vector = None

            for i in range(1, B + 1):
                x = np.prod([1 - j / B for j in range(1, i)])
                if i == 1:
                    max_i = (items_genres_matrix > 0).astype(int).sum(axis=1).max()
                else:
                    max_i, best_coverage_vector = compute_maximum_coverage(items_genres_matrix, best_coverage_vector)
                d += max_i * i / B * x

            d /= n_genres

            with open(saving_path, "a") as f:
                f.write(f"{B}\t{d}\n")


def compute_maximum_coverage(items_genres_matrix, best_coverage_vector):
    binary_items_genres_matrix = (items_genres_matrix > 0).astype(int)

    best_coverage = 0

    if best_coverage_vector is None:
        for row in binary_items_genres_matrix:

            coverage_vector = np.maximum(row, binary_items_genres_matrix)
            genres_covered = coverage_vector.sum(axis=1)

            index_maximum_coverage = np.argsort(genres_covered)[::-1][0]
            maximum_coverage = genres_covered[index_maximum_coverage]

            if maximum_coverage > best_coverage:
                best_coverage = maximum_coverage
                best_coverage_vector = coverage_vector[index_maximum_coverage]

    else:
        coverage_vector = np.maximum(best_coverage_vector, binary_items_genres_matrix)
        genres_covered = coverage_vector.sum(axis=1)

        index_maximum_coverage = np.argsort(genres_covered)[::-1][0]
        best_coverage = genres_covered[index_maximum_coverage]
        best_coverage_vector = coverage_vector[index_maximum_coverage]

    return best_coverage, best_coverage_vector
