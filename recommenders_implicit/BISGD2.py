from collections import defaultdict
import random
from data import ImplicitData
import numpy as np
from .Model3 import Model3


class BISGD2(Model3):
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
        #print(self.user_factors)

    def _InitModel(self):
        self.user_factors = defaultdict(dict)
        self.item_factors = defaultdict(dict)
        self.counter=self.counter + 1
        #print('InitModel', self.counter)
        #print('data.userset no init', self.data.userset)
        for node in range(self.nrNodes):

                self.user_factors[node] = {}
                #self.user_factors[node][u] = np.random.normal(0.0, 0.01, self.num_factors)
                #print(self.user_factors[node][u])

                self.item_factors[node] = {}
                #self.item_factors[node][i] = np.random.normal(0.0, 0.01, self.num_factors)
        #print('user_factors no init', self.user_factors)

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

    def IncrTrain(self, user_id, item_id, node, kappa, update_users: bool = True, update_items: bool = True):
        """
        Incrementally updates the model.

        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """
        self.counter = self.counter + 1

        if user_id not in self.user_factors[node].keys():
            self.user_factors[node][user_id] = np.random.normal(0.0, 0.01, self.num_factors)

        if item_id not in self.item_factors[node].keys():
            self.item_factors[node][item_id] = np.random.normal(0.0, 0.01, self.num_factors)

        self.data.AddFeedback(user_id, item_id)

        for _ in range(kappa):
            self._UpdateFactors(user_id, item_id, node)

        #print('user factors inctrain', self.user_factors)

    def _UpdateFactors(self, user_id, item_id, node, update_users: bool = True, update_items: bool = True, target: int = 1):
        self.counter=self.counter+1
        #print('updatefactors, userid, node',self.counter, user_id, node)
        p_u = self.user_factors[node][user_id]
        #print('pu_antesupdate', len(p_u))
        q_i = self.item_factors[node][item_id]
        #print('qi_antesupdate', len(q_i))
        for _ in range(int(self.num_iterations)):
            err = target - np.inner(p_u, q_i)

            if update_users:
                delta = self.learn_rate * (err * q_i - self.user_regularization * p_u)
                p_u += delta

            if update_items:
                delta = self.learn_rate * (err * p_u - self.item_regularization * q_i)
                q_i += delta

        self.user_factors[node][user_id] = p_u
        #print('pu_depoisupdate', len(p_u))
        #print('user_factors', self.user_factors)
        self.item_factors[node][item_id] = q_i
        #print('qi_depoisupdate', len(q_i))


    def Predict(self, user_id, item_id, node):
        """
        Return the prediction (float) of the user-item interaction score.

        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """
        #if self.use_numba:
            #return _nb_Predict(self.user_factors[user_id], self.item_factors[item_id])
        return np.inner(self.user_factors[node][user_id], self.item_factors[node][item_id])

    def Recommend(self, user_id: int, item_id: int, node: int, kappa: int, n: int = -1, candidates: set = {}, exclude_known_items: bool = True):
        """
        Returns an list of tuples in the form (item_id, score), ordered by score.

        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """
        #print('node no Reccomend', node) #entrou bem
        #print('user_id', user_id) #entrou bem
        #print('self.data.userset', list(self.data.userset)) #dá vazio...
        recs=dict()
        for node_ in range (self.nrNodes):
            recs[node_] = []
        lista=[]

        if user_id not in self.user_factors[node].keys() or item_id not in self.item_factors[node].keys():
            if user_id not in self.user_factors[node].keys():
                self.user_factors[node][user_id] = np.random.normal(0.0, 0.01, self.num_factors)

            if item_id not in self.item_factors[node].keys():
                self.item_factors[node][user_id] = np.random.normal(0.0, 0.01, self.num_factors)

            #self.data.AddFeedback(user_id, item_id)
            #for _ in range(kappa):
                #self._UpdateFactors(user_id, item_id, node)


        if user_id in self.data.userset:
            #print('entrei') # passa aqui...
            self.counter=self.counter+1
            #print('recommend_if1', self.counter)
            #print(self.user_factors[node][user_id])
            if len(candidates) == 0:
                candidates = self.data.itemset

            if exclude_known_items:
                candidates = candidates - set(self.data.GetUserItems(user_id))

            #print('user factors no recomend', self.user_factors)
            #print('list(user factors)', list(self.user_factors.values()))
            p_u = self.user_factors[node][user_id]
            #print('recommend', p_u)
            itemlist = np.array(list(self.item_factors[node].keys()))
            factors = np.array(list(self.item_factors[node].values()))
            """if self.use_numba:
                scores = _nb_get_scores(p_u, factors)
                recs = np.column_stack((itemlist, scores))
                recs = _nb_sort(recs)
            else:"""
            scores = np.abs(1 - np.inner(p_u, factors))
            lista = np.column_stack((itemlist, scores))
            lista = lista[np.argsort(lista[:, 1], kind = 'heapsort')]
            recs[node] = lista

        """
        if n == -1 or n > len(recs[node]):
            self.counter=self.counter+1
            #print(self.user_factors[node][user_id])
            #print('recommend_if2', self.counter)
            #print('recs', recs)
        """
        n = len(recs[node])


        return recs[node][:n]
