from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, Topology, THPSimulation
from castle.algorithms import TTPM

import numpy as np
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(description='argparse testing')
parser.add_argument('--dataset', '-d', type=str, default="sample", required=True, help="dataset folder name")
parser.add_argument('--iter', '-i', type=int, default=20, required=True, help="max iterations")
args = parser.parse_args()

dataset = args.dataset
iter = args.iter

# dataset = "sample"
# iter = 5

print(dataset)

tp = np.load("data/datasets/{}/topology.npy".format(dataset))
ttpm = TTPM(topology_matrix=tp,max_iter=iter, max_hop=2)


X = pd.read_csv('data/datasets/{}/alarm.csv'.format(dataset))
X.drop("end_timestamp",axis=1,inplace=True)
X.rename({"alarm_id":"event","device_id":"node","start_timestamp":"timestamp"},axis=1,inplace=True)


ttpm.learn(X)
# print(ttpm.causal_matrix)

# true_causal_matrix = np.load('data/datasets/{}/causal_prior.npy'.format(dataset))
# GraphDAG(ttpm.causal_matrix, true_causal_matrix)
# ret_metrix = MetricsDAG(ttpm.causal_matrix, true_causal_matrix)
# print(ret_metrix.metrics)
os.makedirs("./submission/ttpm_{}_iter".format(str(iter)), exist_ok=True)
try:
    np.save("./submission/ttpm_{}_iter/{}_graph_matrix.npy".format(str(iter),dataset), ttpm.causal_matrix)
    print("1")
except:
    np.save("./submission/ttpm_{}_iter/{}_graph_matrix.npy".format(str(iter),dataset), ttpm.causal_matrix.to_numpy())
    print("2")