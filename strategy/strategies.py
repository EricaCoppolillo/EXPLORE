import numpy as np
import torch


def clayton_copula(u, v, alpha, epsilon=1e-7):
    return (np.maximum((u + epsilon) ** (-alpha) + (v + epsilon) ** (-alpha) - 1, epsilon)) ** -1 / alpha


def get_original_indices(main_vector, recovering_indices_vector):
    main_vector = np.array(main_vector)
    orig_indices = main_vector.argsort()
    ndx = orig_indices[np.searchsorted(main_vector[orig_indices], recovering_indices_vector)]
    return ndx


def combine_metrics(p_relevance, p_distance, alpha=0.5, copula="clayton"):
    temp, temp_2 = p_relevance.min(), p_distance.min()

    if temp != p_relevance.max():
        p_relevance = (p_relevance - temp) / (p_relevance.max() - temp)
    else:
        p_relevance /= p_relevance.sum()

    if temp_2 != p_distance.max():
        p_distance = (p_distance - temp_2) / (p_distance.max() - temp_2)
    else:
        p_distance /= p_distance.sum()

    if copula is not None:
        if copula == "clayton":
            copula = clayton_copula(p_distance, p_relevance, alpha)
        final = copula
    else:
        final = (p_relevance + p_distance)**2
    # p_sum = copula
    #
    # p_sum = p_sum / p_sum.sum()

    return final


def normalize(metric, discrepancy=10):
    # skew probabilities to distinguish high-/low- rated items
    norm_metric = np.exp(metric) / np.sum(np.exp(metric))
    norm_metric_min, norm_metric_max = norm_metric.min(), norm_metric.max()

    epsilon = 1e-7

    if norm_metric_min == 0.0:
        norm_metric_min = epsilon

    stop = False

    count = 0
    ratio_max_min = 1
    while not stop and ratio_max_min >= 1:
        ratio_max_min = max(10 - count, 10 ** discrepancy)
        # a = ln(ratio_max_min)/ln(x.max()/x.min())
        if norm_metric_max / norm_metric_min == 1:  # rare case -> norm_metric_max = norm_metric_min
            ratio = 0
        else:
            ratio = np.log(ratio_max_min) / np.log(norm_metric_max / norm_metric_min)

        a = max(ratio, 0)
        x = norm_metric ** a

        discrepancy -= 1
        if discrepancy <= -1:
            count += 1

        if np.count_nonzero(x) >= 50 and set(x) != {1}:  # x does not contain all ones
            stop = True

    if not stop:
        x = norm_metric

    x = x / x.sum()

    return x


class Strategy:

    def __init__(self, n_users, n_items, model, items_items_distances, device, clip_min, clip_max,
                 evaluate, users_test):
        self.model = model
        self.device = device
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.all_items = list(range(n_items))

        self.items_items_distances = items_items_distances
        self.ratings = model(torch.tensor(list(range(n_users)), device=self.device).unsqueeze(-1),
                             torch.tensor(self.all_items, device=self.device)).detach().cpu().numpy().clip(
            self.clip_min, self.clip_max)

        self.evaluate = evaluate
        self.users_test = users_test

    def get_recommendations(self, active_users, k, users_items_matrix):
        pass


class RecommendedDistance(Strategy):

    def get_recommendations(self, active_users, k, users_items_matrix, alpha=0.5, use_relevance=True):
        final_users_items = [[] for _ in range(len(active_users))]

        for i in range(len(active_users)):
            candidates = np.where(users_items_matrix[active_users[i]] == 0)[0]
            candidate_ratings = self.ratings[active_users[i]][candidates]

            p = normalize(candidate_ratings)

            user_most_relevant_item = np.random.choice(candidates, p=p)

            candidates = np.setdiff1d(candidates, user_most_relevant_item)

            if self.evaluate:
                test_items = self.users_test[self.users_test["user"] == active_users[i]]["item"].to_numpy()
                candidates = np.array(list(set(candidates).intersection(set(test_items))))

            item_candidates_distances = self.items_items_distances[user_most_relevant_item][candidates]

            relevances = self.ratings[active_users[i]][candidates]

            if not use_relevance:
                sampled_distant_items = list(candidates[np.argsort(item_candidates_distances)][::-1][:k])
            else:
                sampled_distant_items = list(
                    candidates[np.argsort(combine_metrics(relevances, item_candidates_distances))[::-1][:k]])

            final_users_items[i].extend(sampled_distant_items)

        return final_users_items


class InteractedDistance(Strategy):

    def get_recommendations(self, active_users, k, users_items_matrix, alpha=0.5, copula="clayton",
                            use_relevance=True):

        final_users_items = [[] for _ in range(len(active_users))]

        for i in range(len(active_users)):
            already_interacted_items = np.where(users_items_matrix[active_users[i]])[0]

            if len(already_interacted_items) == 0:  # first round
                if not use_relevance:
                    mean_items_distances = self.items_items_distances.mean(axis=-1)
                    p = normalize(mean_items_distances)
                    already_interacted_items = [np.random.choice(self.all_items, p=p)]
                else:
                    p = normalize(self.ratings[active_users[i]])
                    already_interacted_items = [np.random.choice(self.all_items, p=p)]  # start with the most relevant

            candidates = np.setdiff1d(self.all_items, already_interacted_items)

            if self.evaluate:
                test_items = self.users_test[self.users_test["user"] == active_users[i]]["item"].to_numpy()
                candidates = np.array(list(set(candidates).intersection(set(test_items))))

            distances = []
            for a in already_interacted_items:
                distances.append(self.items_items_distances[a][candidates])

            candidates_distances_from_already_interacted_items = np.array(distances).sum(axis=0)

            relevances = self.ratings[active_users[i]][candidates]

            if not use_relevance:
                sampled_distant_items = list(candidates[np.argsort(candidates_distances_from_already_interacted_items)]
                                             [::-1][:k])
            else:
                sampled_distant_items = list(candidates[np.argsort(
                    combine_metrics(relevances, candidates_distances_from_already_interacted_items, alpha=alpha,
                                    copula=copula))[::-1][:k]])

            final_users_items[i].extend(sampled_distant_items)

        return final_users_items


class InteractedCoverage(Strategy):

    def __init__(self, items_genres_matrix, n_users, n_items, model, items_items_distances, device, clip_min, clip_max,
                 evaluate,
                 users_test):
        super().__init__(n_users, n_items, model, items_items_distances, device, clip_min, clip_max, evaluate,
                         users_test)

        self.items_genres_matrix = items_genres_matrix

    def get_recommendations(self, active_users, k, users_items_matrix, alpha=0.5, copula="clayton",
                            use_relevance=True):

        binary_items_genres_matrix = (self.items_genres_matrix > 0).astype(int)

        final_users_items = [[] for _ in range(len(active_users))]

        for i in range(len(active_users)):
            already_interacted_items = np.where(users_items_matrix[active_users[i]])[0]
            candidates = np.where(users_items_matrix[active_users[i]] == 0)[0]

            if self.evaluate:
                test_items = self.users_test[self.users_test["user"] == active_users[i]]["item"].to_numpy()
                candidates = np.array(list(set(candidates).intersection(set(test_items))))

            if not len(already_interacted_items):
                coverages = binary_items_genres_matrix[candidates].sum(axis=1)
            else:
                coverage_vector = (binary_items_genres_matrix[already_interacted_items].sum(axis=0) > 0).astype(int)
                candidates_coverage_vectors = np.maximum(coverage_vector, binary_items_genres_matrix[candidates])
                coverages = candidates_coverage_vectors.sum(axis=1)

            relevances = self.ratings[active_users[i]][candidates]

            if not use_relevance:
                sampled_best_coverage_items = list(candidates[np.argsort(coverages.astype(float))][::-1][:k])
            else:
                sampled_best_coverage_items = list(candidates[np.argsort(
                    combine_metrics(relevances, coverages.astype(float), alpha=alpha, copula=copula))[::-1][:k]])

            final_users_items[i].extend(sampled_best_coverage_items)

        return final_users_items
