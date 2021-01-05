from data import ImplicitData
from param_tuning import PatternSearchISGD
from pprint import pprint
import pandas as pd
from operator import itemgetter
import numpy as np
from recommenders_implicit import * #BISGD
from recommenders_implicit.BISGD import BISGD
from eval_implicit.EvalPrequential2 import EvalPrequential2
from datetime import datetime

data = pd.read_csv("datasets/playlisted_tracks.tsv","\t")
stream = ImplicitData(data['playlist_id'],data['track_id'])

numeroNodes = 8
model = BISGD(ImplicitData([],[]),200,6, numeroNodes, learn_rate = 0.35, u_regularization = 0.5, i_regularization = 0.5, use_numba = False)

eval = EvalPrequential2(model,stream, metrics = ["Recall@20"], NrNodes = numeroNodes)
"""
res = eval.EvaluateTime(0, 20000)

print("GetTuple: " + str(sum(res['time_get_tuple'])))
print("Recommend: " + str(sum(res['time_recommend'])))
print("EvalPoint: " + str(sum(res['time_eval_point'])))
print("Update: " + str(sum(res['time_update'])))
"""
#meu
start_recommend = datetime.now()
print('start time', start_recommend)

resultados=eval.Evaluate(0,stream.size)

for node in range(numeroNodes):
    print('sum(resultados[Recall@20][node])/stream.size', sum(resultados['Recall@20'][node])/stream.size)


#node1 0.137 #recall10
#node2 0.135 #recall10

#recall20
# 0.1834
# 0.18129
#run time 1:28:22
#recall20 n=6
# 0.1856
# 0.18379
# 0.1852
# 0.1878
# 0.1808
# 0.1825
#run time 4:40:58

#recall20 n=8
# 0.179
#0.1869
#0.1848
#0.1847
#0.1844
#0.1849
#0.1860
#0.1849
#run time 6:15:30

end_recommend = datetime.now()
print('end time', end_recommend)

tempo = end_recommend - start_recommend

print('run time', tempo)


#resultados2=eval.Evaluate(100000,20000)
#print(sum(resultados2['Recall@10'])/20000) #deu 0.0019 para 10000 e para 20000 0.0012 com parâmetros do artigo passou para 0.08415

#resultados3=eval.Evaluate(0,stream.size)
#print(sum(resultados3['Recall@10'])/stream.size) # com hyperparametros otimizados do artigo e o dataset todo deu 0.200 de recall@10
#print(stream.size)

#for i in range(stream.size % 1000):
#    print("Simplex:")
#    pprint(eval.simplex)
#    print("Simplex scores:")
#    pprint(eval.simplex_scores)
#    print("Candidates:")
#    pprint(eval.candidate_points)
#    eval.Iterate()
