from data import ImplicitData
from param_tuning import PatternSearchISGD
from pprint import pprint
import pandas as pd
from operator import itemgetter
import numpy as np
from recommenders_implicit import ISGD
from eval_implicit import EvalPrequential
from datetime import datetime
from scipy.stats import poisson
import random

#np.random.seed(1)
#np.random.seed(2)
np.random.seed(42)

numeroNodes = 12
data = pd.read_csv("datasets/playlisted_tracks.tsv","\t")
#print(data['playlist_id'].head())
data.columns = ['User','Item', 'Time']

kappa={}
for index in data.index:
    kappa[index]={}
    for node in range(numeroNodes):
        kappa[index][node]= int(poisson.rvs(1, size = 1))

df_kappa= pd.DataFrame.from_dict(kappa, orient='index')
data_merged=data.merge(df_kappa, left_on=data.index, right_on=df_kappa.index)

node0=data_merged[['User', 'Item', 0]]
node0.columns = ['User', 'Item', 'N0']
node0_a = node0.loc[np.repeat(node0.index.values, node0.N0)]
indice_0=[x for x in range(len(node0_a))]
node0_a['indice'] = indice_0
node0_= node0_a.set_index('indice')

node1=data_merged[['User', 'Item', 1]]
node1.columns = ['User', 'Item', 'N1']
node1_a = node1.loc[np.repeat(node1.index.values, node1.N1)]
indice_1=[x for x in range(len(node1_a))]
node1_a['indice'] = indice_1
node1_= node1_a.set_index('indice')

node2=data_merged[['User', 'Item', 2]]
node2.columns = ['User', 'Item', 'N2']
node2_a = node2.loc[np.repeat(node2.index.values, node2.N2)]
indice_2=[x for x in range(len(node2_a))]
node2_a['indice'] = indice_2
node2_= node2_a.set_index('indice')

node3=data_merged[['User', 'Item', 3]]
node3.columns = ['User', 'Item', 'N3']
node3_a = node3.loc[np.repeat(node3.index.values, node3.N3)]
indice_3=[x for x in range(len(node3_a))]
node3_a['indice'] = indice_3
node3_= node3_a.set_index('indice')

node4=data_merged[['User', 'Item', 4]]
node4.columns = ['User', 'Item', 'N4']
node4_a = node4.loc[np.repeat(node4.index.values, node4.N4)]
indice_4=[x for x in range(len(node4_a))]
node4_a['indice'] = indice_4
node4_= node4_a.set_index('indice')

node5=data_merged[['User', 'Item', 5]]
node5.columns = ['User', 'Item', 'N5']
node5_a = node5.loc[np.repeat(node5.index.values, node5.N5)]
indice_5=[x for x in range(len(node5_a))]
node5_a['indice'] = indice_5
node5_= node5_a.set_index('indice')

node6=data_merged[['User', 'Item', 6]]
node6.columns = ['User', 'Item', 'N6']
node6_a = node6.loc[np.repeat(node6.index.values, node6.N6)]
indice_6=[x for x in range(len(node6_a))]
node6_a['indice'] = indice_6
node6_= node6_a.set_index('indice')

node7=data_merged[['User', 'Item', 7]]
node7.columns = ['User', 'Item', 'N7']
node7_a = node7.loc[np.repeat(node7.index.values, node7.N7)]
indice_7=[x for x in range(len(node7_a))]
node7_a['indice'] = indice_7
node7_= node7_a.set_index('indice')

node8=data_merged[['User', 'Item', 8]]
node8.columns = ['User', 'Item', 'N8']
node8_a = node8.loc[np.repeat(node8.index.values, node8.N8)]
indice_8=[x for x in range(len(node8_a))]
node8_a['indice'] = indice_8
node8_= node8_a.set_index('indice')

node9=data_merged[['User', 'Item', 9]]
node9.columns = ['User', 'Item', 'N9']
node9_a = node9.loc[np.repeat(node9.index.values, node9.N9)]
indice_9=[x for x in range(len(node9_a))]
node9_a['indice'] = indice_9
node9_= node9_a.set_index('indice')

node10=data_merged[['User', 'Item', 10]]
node10.columns = ['User', 'Item', 'N10']
node10_a = node10.loc[np.repeat(node10.index.values, node10.N10)]
indice_10=[x for x in range(len(node10_a))]
node10_a['indice'] = indice_10
node10_= node10_a.set_index('indice')

node11=data_merged[['User', 'Item', 11]]
node11.columns = ['User', 'Item', 'N11']
node11_a = node11.loc[np.repeat(node11.index.values, node11.N11)]
indice_11=[x for x in range(len(node11_a))]
node11_a['indice'] = indice_11
node11_= node11_a.set_index('indice')

stream_0 = ImplicitData(node0_['User'], node0_['Item'])
stream_1 = ImplicitData(node1_['User'], node1_['Item'])
stream_2 = ImplicitData(node2_['User'], node2_['Item'])
stream_3 = ImplicitData(node3_['User'], node3_['Item'])
stream_4 = ImplicitData(node4_['User'], node4_['Item'])
stream_5 = ImplicitData(node5_['User'], node5_['Item'])
stream_6 = ImplicitData(node6_['User'], node6_['Item'])
stream_7 = ImplicitData(node7_['User'], node7_['Item'])
stream_8 = ImplicitData(node8_['User'], node8_['Item'])
stream_9 = ImplicitData(node9_['User'], node9_['Item'])
stream_10 = ImplicitData(node10_['User'], node10_['Item'])
stream_11 = ImplicitData(node11_['User'], node11_['Item'])

model = ISGD(ImplicitData([],[]),200,6, learn_rate = 0.35, u_regularization = 0.5, i_regularization = 0.5, use_numba = False)

eval_0 = EvalPrequential(model,stream_0)
eval_1 = EvalPrequential(model,stream_1)
eval_2 = EvalPrequential(model,stream_2)
eval_3 = EvalPrequential(model,stream_3)
eval_4 = EvalPrequential(model,stream_4)
eval_5 = EvalPrequential(model,stream_5)
eval_6 = EvalPrequential(model,stream_6)
eval_7 = EvalPrequential(model,stream_7)
eval_8 = EvalPrequential(model,stream_8)
eval_9 = EvalPrequential(model,stream_9)
eval_10 = EvalPrequential(model,stream_10)
eval_11 = EvalPrequential(model,stream_11)

"""
res = eval.EvaluateTime(0, 1000)

print("GetTuple: " + str(sum(res['time_get_tuple'])))
print("Recommend: " + str(sum(res['time_recommend'])))
print("EvalPoint: " + str(sum(res['time_eval_point'])))
print("Update: " + str(sum(res['time_update'])))

"""
#meu
#resultados=eval.Evaluate(0,1000)
#print(sum(resultados['Recall@10'])/20000) #deu 0.0043 de recall@10

#resultados2=eval.Evaluate(100000,20000)
#print(sum(resultados2['Recall@10'])/20000) #deu 0.0019 para 10000 e para 20000 0.0012 com parâmetros do artigo passou para 0.08415
start_recommend = datetime.now()
print('start time', start_recommend)

resultados_0=eval_0.Evaluate(0,stream_0.size)
print(sum(resultados_0['Recall@20'])/stream_0.size) # com hyperparametros otimizados do artigo e o dataset todo deu 0.200 de recall@10

resultados_1=eval_1.Evaluate(0,stream_1.size)
print(sum(resultados_1['Recall@20'])/stream_1.size)

resultados_2=eval_2.Evaluate(0,stream_2.size)
print(sum(resultados_2['Recall@20'])/stream_2.size)

resultados_3=eval_3.Evaluate(0,stream_3.size)
print(sum(resultados_3['Recall@20'])/stream_3.size)

resultados_4=eval_4.Evaluate(0,stream_4.size)
print(sum(resultados_4['Recall@20'])/stream_4.size)

resultados_5=eval_5.Evaluate(0,stream_5.size)
print(sum(resultados_5['Recall@20'])/stream_5.size)

resultados_6=eval_6.Evaluate(0,stream_6.size)
print(sum(resultados_6['Recall@20'])/stream_6.size)

resultados_7=eval_7.Evaluate(0,stream_7.size)
print(sum(resultados_7['Recall@20'])/stream_7.size)

resultados_8=eval_8.Evaluate(0,stream_8.size)
print(sum(resultados_8['Recall@20'])/stream_8.size)

resultados_9=eval_9.Evaluate(0,stream_9.size)
print(sum(resultados_9['Recall@20'])/stream_9.size)

resultados_10=eval_10.Evaluate(0,stream_10.size)
print(sum(resultados_10['Recall@20'])/stream_10.size)

resultados_11=eval_11.Evaluate(0,stream_11.size)
print(sum(resultados_11['Recall@20'])/stream_11.size)


end_recommend = datetime.now()
print('end time', end_recommend)

tempo = end_recommend - start_recommend
print('run time', tempo)


# tempo de execução 42 min para single node
#Recall@20=0.256 (maior que o recall10) no artigo é 0.302
#two nodes
#Recall@20_1= 0.4719
#Recall@20_2= 0.5838
#media= (0.4719+0.5838)/2=0.52785
#run time 1:48:39

#6 nodes, #Recall@20
#0.47564
#0.58404
#0.60663
#0.60952
#0.60758
#0.60588
#media=0.58155
#run time 7:15:17

#8 nodes Recall@20
#0.472768
#0.583885
#0.604196
#0.607264
#0.609735
#0.608921
#0.604086
#0.598165
#media=0.58613
#run time 10:02:32


#8 node recall20 random seed(2)
#0.472019
#0.580753
#0.603330
#0.606535
#0.607539
#0.604957
#0.602052
#0.598922
#media=0.58451

#12 nodes recall20 random seed(1)
#0.476633
#0.583596
#0.604145
#0.606532
#0.607150
#0.604721
#0.601108
#0.596256
#0.593388
#0.592384
#0.586645
#0.583197
#media=0.58631
#run time 15:44:51

#12 nodes recall20 random seed(42)
#0.473312
#0.585545
#0.605927
#0.610216
#0.608571
#0.606329
#0.603653
#0.601122
#0.599189
#0.592734
#0.591729
#0.586447
#media=0.58873
#run time 16:24:26
