import numpy as np
import math

from strategies import Strategy


class Competitor(Strategy):

    def __init__(self, n_users, n_items, model, items_items_distances, device, clip_min, clip_max, evaluate,
                 users_test):
        super().__init__(n_users, n_items, model, items_items_distances, device, clip_min, clip_max, evaluate,
                         users_test)

        self.sim_matrix = 1 - items_items_distances

    def get_recommendations(self, active_users, k, users_items_matrix):
        pass


class Random(Competitor):

    def get_recommendations(self, active_users, k, users_items_matrix):
        final_users_items = [[] for _ in range(len(active_users))]

        for i in range(len(active_users)):
            candidates = np.where(users_items_matrix[active_users[i]] == 0)[0]

            if self.evaluate:
                test_items = self.users_test[self.users_test["user"] == active_users[i]]["item"].to_numpy()
                candidates = np.array(list(set(candidates).intersection(set(test_items))))

            random_items = np.random.choice(candidates, size=k, replace=False)

            final_users_items[i].extend(list(random_items))

        return final_users_items


class Relevance(Competitor):

    def get_recommendations(self, active_users, k, users_items_matrix):

        final_users_items = [[] for _ in range(len(active_users))]

        for i in range(len(active_users)):

            candidates = np.where(users_items_matrix[active_users[i]] == 0)[0]

            if self.evaluate:
                test_items = self.users_test[self.users_test["user"] == active_users[i]]["item"].to_numpy()
                candidates = np.array(list(set(candidates).intersection(set(test_items))))

            candidates_relevance = self.ratings[active_users[i]][candidates]
            sampled_relevant_items = list(candidates[np.argsort(candidates_relevance)[::-1][:k]])

            final_users_items[i].extend(sampled_relevant_items)

        return final_users_items


class MMR(Competitor):

    def __init__(self, lambda_factor, n_users, n_items, model, items_items_distances, device, clip_min, clip_max,
                 evaluate,
                 users_test):
        super().__init__(n_users, n_items, model, items_items_distances, device, clip_min, clip_max, evaluate,
                         users_test)

        self.lambda_factor = lambda_factor

    def get_recommendations(self, active_users, k, users_items_matrix):

        # lambda_score * rel - (1 - lambda_score) * sim
        final_recommendation_list = [[] for _ in range(len(active_users))]

        for u in range(len(active_users)):

            candidates = np.where(users_items_matrix[active_users[u]] == 0)[0]

            if self.evaluate:
                test_items = self.users_test[self.users_test["user"] == active_users[u]]["item"].to_numpy()
                candidates = np.array(list(set(candidates).intersection(set(test_items))))

            relevance = self.ratings[active_users[u]][candidates]

            # excluding already interacted items
            similarity = self.sim_matrix[np.ix_(candidates, candidates)]

            for i in range(k):

                if i == 0:
                    final_recommendation_list[u].append(np.argmax(relevance))
                    relevance[final_recommendation_list[u]] = np.NINF
                    continue

                to_maximize = self.lambda_factor * relevance - (1 - self.lambda_factor) * \
                              np.max(similarity[final_recommendation_list[u][:i]], axis=0)

                if self.evaluate and np.max(to_maximize) == np.NINF:
                    break

                final_recommendation_list[u].append(np.argmax(to_maximize))

                relevance[final_recommendation_list[u][i]] = np.NINF

        return final_recommendation_list


class DPP(Competitor):

    def get_kernel_matrix(self, relevance, similarity):  # kernel matrix

        item_size = len(relevance)
        kernel_matrix = relevance.reshape((item_size, 1)) * similarity * relevance.reshape((1, item_size))

        return kernel_matrix

    def dpp(self, L, k, epsilon=1e-10):
        """
        Our proposed fast implementation of the greedy algorithm
        :param kernel_matrix: 2-d array
        :param max_length: positive int
        :param epsilon: small positive scalar
        :return: list
        """
        item_size = L.shape[0]
        cis = np.zeros((k, item_size))
        di2s = np.copy(np.diag(L))
        selected_items = list()
        selected_item = np.argmax(di2s)
        selected_items.append(selected_item)
        while len(selected_items) < k:
            s = len(selected_items) - 1
            ci_optimal = cis[:s, selected_item]
            di_optimal = math.sqrt(di2s[selected_item])
            elements = L[selected_item, :]

            if not di_optimal:
                eis = 0
            else:
                eis = (elements - np.dot(ci_optimal, cis[:s, :])) / di_optimal

            cis[s, :] = eis
            di2s -= np.square(eis)
            di2s[selected_item] = -np.inf
            selected_item = np.argmax(di2s)
            if di2s[selected_item] < epsilon:
                break
            selected_items.append(selected_item)

        return selected_items

    def get_recommendations(self, active_users, k, users_items_matrix):

        final_recommendation_list = [[] for _ in range(len(active_users))]

        for i in range(len(active_users)):
            candidates = np.where(users_items_matrix[active_users[i]] == 0)[0]

            if self.evaluate:
                test_items = self.users_test[self.users_test["user"] == active_users[i]]["item"].to_numpy()
                candidates = np.array(list(set(candidates).intersection(set(test_items))))

            relevance = self.ratings[active_users[i]][candidates]
            similarity = self.sim_matrix[candidates][:, candidates]

            L = self.get_kernel_matrix(relevance, similarity)

            user_recommendation_list = self.dpp(L, k)

            final_recommendation_list[i] = list(candidates[user_recommendation_list])

        return final_recommendation_list


class DUM(Competitor):

    def __init__(self, items_genres_matrix, n_users, n_items, model, items_items_distances, device, clip_min, clip_max,
                 evaluate, users_test):
        super().__init__(n_users, n_items, model, items_items_distances, device, clip_min, clip_max, evaluate,
                         users_test)

        self.items_genres_matrix = items_genres_matrix

    def get_recommendations(self, active_users, k, users_items_matrix):

        final_recommendation_list = [[] for _ in range(len(active_users))]

        for i in range(len(active_users)):

            candidates = np.where(users_items_matrix[active_users[i]] == 0)[0]

            if self.evaluate:
                test_items = self.users_test[self.users_test["user"] == active_users[i]]["item"].to_numpy()
                candidates = np.array(list(set(candidates).intersection(set(test_items))))

            user_weights = self.ratings[active_users[i]][candidates] / self.clip_max

            ordered_by_weights = candidates[np.argsort(user_weights)[::-1]]

            final_recommendation_list[i].append(ordered_by_weights[0])

            for c in ordered_by_weights[1:]:

                coverage_vector_without_c = self.items_genres_matrix[final_recommendation_list[i]].sum(axis=0)
                coverage_without_c = (coverage_vector_without_c > 0).astype(int).sum()

                coverage_vector_with_c = self.items_genres_matrix[(final_recommendation_list[i] + [c])].sum(axis=0)
                coverage_with_c = (coverage_vector_with_c > 0).astype(int).sum()

                if coverage_with_c > coverage_without_c:
                    final_recommendation_list[i].append(c)

                if len(final_recommendation_list[i]) == k:
                    break

            if len(final_recommendation_list[i]) < k:
                n_to_add = k - len(final_recommendation_list[i])
                not_picked_candidates = np.setdiff1d(ordered_by_weights, final_recommendation_list[i])
                not_picked_weights = self.ratings[active_users[i]][not_picked_candidates] / self.clip_max
                not_picked_ordered_by_weights = not_picked_candidates[np.argsort(not_picked_weights)[::-1]]
                final_recommendation_list[i].extend(not_picked_ordered_by_weights[:n_to_add])

        return final_recommendation_list
