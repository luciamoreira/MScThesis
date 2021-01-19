from scipy.stats import poisson
from collections import defaultdict
import random
from data import ImplicitData
import numpy as np
from .Model import Model


class UBISGD(Model):
    def __init__(self, data: ImplicitData, num_factors: int = 10, num_iterations: int = 10, NrNodes: int = 5, learn_rate: float = 0.01, u_regularization: float = 0.1, i_regularization: float = 0.1, random_seed: int = 1, use_numba: bool = False):
        """    Constructor.

        Keyword arguments:
        data -- ImplicitData object
        num_factors -- Number of latent features (int, default 10)
        num_iterations -- Maximum number of iterations (int, default 10)
        learn_rate -- Learn rate, aka step size (float, default 0.01)
        regularization -- Regularization factor (float, default 0.01)
        random_seed -- Random seed (int, default 1)"""

        self.counter=0
        self.data = data
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.learn_rate = learn_rate
        self.user_regularization = u_regularization
        self.item_regularization = i_regularization
        self.random_seed = random_seed
        self.use_numba = use_numba
        self.nrNodes = NrNodes
        np.random.seed(random_seed)
        self._InitModel()

    def _InitModel(self):
        self.user_factors = defaultdict(dict)
        self.item_factors = defaultdict(dict)
        self.kappa_users = defaultdict(dict)

        for node in range(self.nrNodes):
            for u in self.data.userset:
                self.user_factors[node][u] = np.random.normal(0.0, 0.01, self.num_factors)
                self.kappa_users[node][u] = int(poisson.rvs(1, size=1))
            for i in self.data.itemset:
                self.item_factors[node][i] = np.random.normal(0.0, 0.01, self.num_factors)

    def BatchTrain(self):
        """
        Trains a new model with the available data.
        """
        idx = list(range(self.data.size))
        for iter in range(self.num_iterations):
            np.random.shuffle(idx)
            for i in idx:
                user_id, item_id = self.data.GetTuple(i)
                self._UpdateFactors(user_id, item_id)

    def IncrTrain(self, user_id, item_id, update_users: bool = True, update_items: bool = True):
        """
        Incrementally updates the model.

        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """

        if user_id not in self.data.userset:
            for node in range(self.nrNodes):
                self.user_factors[node][user_id] = np.random.normal(0.0, 0.01, self.num_factors)
                self.kappa_users[node][user_id] = int(poisson.rvs(1, size=1))
        if item_id not in self.data.itemset:
            for node in range(self.nrNodes):
                self.item_factors[node][item_id] = np.random.normal(0.0, 0.01, self.num_factors)

        self.data.AddFeedback(user_id, item_id)

        for node in range(self.nrNodes):
            #kappa = int(poisson.rvs(1, size=1))

            if self.kappa_users[node][user_id] > 0:
                for _ in range(self.kappa_users[node][user_id]):
                    self._UpdateFactors(user_id, item_id, node)


    def _UpdateFactors(self, user_id, item_id, node, update_users: bool = True, update_items: bool = True, target: int = 1):

        p_u = self.user_factors[node][user_id]
        q_i = self.item_factors[node][item_id]

        for _ in range(int(self.num_iterations)):
            err = target - np.inner(p_u, q_i)

            if update_users:
                delta = self.learn_rate * (err * q_i - self.user_regularization * p_u)
                p_u += delta

            if update_items:
                delta = self.learn_rate * (err * p_u - self.item_regularization * q_i)
                q_i += delta

        self.user_factors[node][user_id] = p_u
        self.item_factors[node][item_id] = q_i

    def Predict(self, user_id, item_id):
        """
        Return the prediction (float) of the user-item interaction score.

        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """
        #if self.use_numba:
            #return _nb_Predict(self.user_factors[user_id], self.item_factors[item_id])
        return np.inner(self.user_factors[node][user_id], self.item_factors[node][item_id])

    def Recommend(self, user_id: int, n: int = -1, candidates: set = {}, exclude_known_items: bool = True):

        recommendation_list= {}

        for node in range(self.nrNodes):
            recommendation_list[node]=self._Recommend(user_id, node, candidates, exclude_known_items) # resultado é um tuple com lista de itense de respectivos scores

        scores_per_node={}

        for node in recommendation_list.keys():
            scores_per_node[node]={}
            for item, score in recommendation_list[node]:
                scores_per_node[node][item] = score


        avgs_score = defaultdict(list)
        score_item = {}

        for node, item_score_pair in scores_per_node.items():
            for item in item_score_pair:
                avgs_score[item].append(item_score_pair[item])

        for item, score_list in avgs_score.items():
            score_item[item]= sum(score_list)/len(score_list)

        scores_itens = list(score_item.items())
        recommendations_bagged = np.array(scores_itens)
        if len(recommendations_bagged) != 0:
            recommendations_bagged = recommendations_bagged[np.argsort(recommendations_bagged[:, 1], kind = 'heapsort')]

        return recommendations_bagged

    def _Recommend(self, user_id: int, node: int, candidates: set = {}, exclude_known_items: bool = True):
        """
        Returns an list of tuples in the form (item_id, score), ordered by score.

        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """

        recs= []

        if(user_id in self.data.userset):

            if len(candidates) == 0:
                candidates = self.data.itemset

            if exclude_known_items:
                candidates = candidates - set(self.data.GetUserItems(user_id))

            p_u = self.user_factors[node][user_id]
            itemlist = np.array(list(self.item_factors[node].keys()))
            factors = np.array(list(self.item_factors[node].values()))

            scores = np.abs(1 - np.inner(p_u, factors))
            recs = np.column_stack((itemlist, scores))

        n = len(recs)
        return recs[:n]
