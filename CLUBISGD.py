from numba import njit
from collections import defaultdict
import random
from data import ImplicitData
import numpy as np
from .Model import Model

class CLUBISGD(Model):
    def __init__(self, data: ImplicitData, clusters: dict, num_factors: int = 10, num_iterations: int = 10, NrNodes: int = 5, learn_rate: float = 0.01, u_regularization: float = 0.1, i_regularization: float = 0.1, random_seed: int = 1, use_numba: bool = False):
        """    Constructor.

        Keyword arguments:
        data -- ImplicitData object
        num_factors -- Number of latent features (int, default 10)
        num_iterations -- Maximum number of iterations (int, default 10)
        learn_rate -- Learn rate, aka step size (float, default 0.01)
        regularization -- Regularization factor (float, default 0.01)
        random_seed -- Random seed (int, default 1)"""

        self.data = data
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.learn_rate = learn_rate
        self.user_regularization = u_regularization
        self.item_regularization = i_regularization
        self.random_seed = random_seed
        self.use_numba = use_numba
        self.nrNodes = NrNodes #the same as number of clusters
        self.FuzzyClusters = clusters
        np.random.seed(random_seed)
        self._InitModel()

    def _InitModel(self):
        self.user_factors = [[np.random.Generator.normal(0.0, 0.1, self.num_factors) for _ in range(self.data.maxuserid + 1)] for _ in range(self.nrNodes)]
        self.item_factors = [[np.random.Generator.normal(0.0, 0.1, self.num_factors) for _ in range(self.data.maxitemid + 1)] for _ in range(self.nrNodes)]
        #self.kappa_users = [[int(np.random.poisson.rvs(1, size=1)) for _ in range(self.data.maxuserid + 1)] for _ in range(self.nrNodes)]

    def BatchTrain(self):
        """
        Trains a new model with the available data.
        """
        idx = list(range(self.data.size))
        for iter in range(self.num_iterations):
            np.random.shuffle(idx)
            for i in idx:
                user_id, item_id = self.data.GetTuple(i, True)
                self._UpdateFactors(user_id, item_id)

    def IncrTrain(self, user, item, update_users: bool = True, update_items: bool = True):
        """
        Incrementally updates the model.

        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """

        user_id, item_id = self.data.AddFeedback(user, item)

        for node in range(self.nrNodes):
            if len(self.user_factors[node]) == self.data.maxuserid:
                self.user_factors[node].append(np.random.normal(0.0, 0.1, self.num_factors))
                #self.kappa_users[node].apppend(int(np.random.poisson.rvs(1, size=1)))
        for node in range(self.nrNodes):
            if len(self.item_factors[node]) == self.data.maxitemid:
                self.item_factors[node].append(np.random.normal(0.0, 0.1, self.num_factors))


        for node in range(self.nrNodes):
            #kappa = int(np.random.poisson(1, size=1))

            if self.FuzzyClusters[user][str(node)] > 1/(1.05*self.nrNodes): #higher than 95% from a random cluster allocation probability
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

    def Recommend(self, user, n: int = -1, exclude_known_items: bool = True):

        user_id = self.data.GetUserInternalId(user)
        if user_id == -1:
            return []


        recommendation_list = np.empty((self.data.maxitemid + 1, self.nrNodes))

        for node in range(self.nrNodes):
            p_u = self.user_factors[node][user_id]
            recommendation_list[:,node] = np.abs(1 - np.inner(p_u, self.item_factors[node]))

        scores = np.mean(recommendation_list, 1)
        recs = np.column_stack((self.data.itemset, scores))

        if exclude_known_items:
            user_items = self.data.GetUserItems(user_id)
            recs = np.delete(recs, user_items, 0)

        recs = recs[np.argsort(recs[:, 1], kind = 'heapsort')]

        if n == -1 or n > len(recs) :
            n = len(recs)

        return recs[:n]
