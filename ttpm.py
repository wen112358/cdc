from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, Topology, THPSimulation
from castle.algorithms import TTPM

import numpy as np
import pandas as pd
import argparse
import os

# parser = argparse.ArgumentParser(description='argparse testing')
# parser.add_argument('--dataset', '-d', type=str, default="sample", required=True, help="Dataset folder name")
# args = parser.parse_args()

# dataset = args.dataset
dataset = "sample"
iter = 1

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
np.save("./submission/ttpm_{}_iter/{}_graph_matrix.npy".format(str(iter),dataset), ttpm.causal_matrix.to_numpy())
