from data import ImplicitData
from param_tuning import PatternSearchISGD
from pprint import pprint
import pandas as pd
from operator import itemgetter
import numpy as np
from recommenders_implicit import * #BISGD
from recommenders_implicit.BISGD_ import BISGD
from eval_implicit.EvalPrequential2_ import EvalPrequential
from datetime import datetime

data = pd.read_csv("datasets/playlisted_tracks.tsv","\t")
stream = ImplicitData(data['playlist_id'],data['track_id'])

numeroNodes = 2
model = BISGD(ImplicitData([],[]),200,6, numeroNodes, learn_rate = 0.35, u_regularization = 0.5, i_regularization = 0.5, use_numba = False)

eval = EvalPrequential(model,stream, metrics = ["Recall@20"], NrNodes = numeroNodes)

start_recommend = datetime.now()
print('start time', start_recommend)

resultados=eval.Evaluate(0,stream.size)

for node in range(numeroNodes):
    print('sum(resultados[Recall@20])/stream.size', sum(resultados['Recall@20'])/stream.size)


#recall20 n=2
# 
#run time predicted: 10h for 2 nodes...


end_recommend = datetime.now()
print('end time', end_recommend)

tempo = end_recommend - start_recommend

print('run time', tempo)
