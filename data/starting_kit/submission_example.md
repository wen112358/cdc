# 1. Basic Format and Structure of the Submission File

In the 1st phase, participants must submit a `.zip` file containing four `.npy` files. Each `.npy` file should store a binary causal matrix that reflects the causal graph specific to its dataset. It's important to note that within a causal matrix, the element at the `i-th` row and `j-th` column represents the presence (or absence) of a directed edge from alarm type (i.e. alarm_id) `i` to alarm type (i.e. alarm_id) `j`.

# 2. Reference to Standard Submission File

For a comprehensive example of the standard submission file, please refer to the following link: [NeurIPS CSL Competition Submission File](https://github.com/huawei-noah/trustworthyAI/blob/master/competition/NeurIPS2023/submission/submission.zip).

# 3. Naming Rule of `.npy` Files

The `.npy` file naming convention should follow this pattern: `{dataset name}_graph_matrix.npy`.

For example:
- The `.npy` file for dataset_1: `dataset_1_graph_matrix.npy`

The subsequent script demonstrates a basic example of generating an `.npy` file for `dataset_1`. Similar steps can be followed for other datasets.


```python
import numpy as np
import pandas as pd
```

#### 3.1 Load Relevant Files for Training in `dataset_1`


```python
# alarm data
alarms = pd.read_csv(r'./datasets/dataset_1/alarm.csv')
# causal_prior
causal_prior= np.load(r'./datasets/dataset_1/causal_prior.npy')


print(f"shape of alarm data: {alarms.shape}")
print(f"shape of causal prior matrix: {causal_prior.shape}")
# Notes: topology.npy and rca_prior.csv are not used in this script.
```

    shape of alarm data: (141853, 4)
    shape of causal prior matrix: (39, 39)


#### 3.2  Select `gCastle` as the Base Algorithm Library


```python
from castle.algorithms import PC
from castle.common.priori_knowledge import PrioriKnowledge
```

    2023-08-15 16:17:36,518 - /home/zhangkeli/anaconda3/lib/python3.9/site-packages/castle/backend/__init__.py[line:36] - INFO: You can use `os.environ['CASTLE_BACKEND'] = backend` to set the backend(`pytorch` or `mindspore`).
    2023-08-15 16:17:36,557 - /home/zhangkeli/anaconda3/lib/python3.9/site-packages/castle/algorithms/__init__.py[line:36] - INFO: You are using ``pytorch`` as the backend.


#### Illustrated by the `PC algorithm` with the ability to incorporate prior knowledge.


```python
TIME_WIN_SIZE  = 300
```


```python
#Use a time sliding window(300 seconds) to generate IID samples.
alarms = alarms.sort_values(by='start_timestamp')
alarms['win_id'] = alarms['start_timestamp'].map(lambda elem:int(elem/TIME_WIN_SIZE))

samples=alarms.groupby(['alarm_id','win_id'])['start_timestamp'].count().unstack('alarm_id')
samples = samples.dropna(how='all').fillna(0)
samples = samples.sort_index(axis=1)
```


```python
samples.shape
```




    (2017, 39)




```python
# create the prior knowledge object for the PC algorithm 
prior_knowledge = PrioriKnowledge(causal_prior.shape[0])
for i, j in zip(*np.where(causal_prior == 1)):
    prior_knowledge.add_required_edge(i, j)

for i, j in zip(*np.where(causal_prior == 0)):
    prior_knowledge.add_forbidden_edge(i, j)
```

#### 3.3 Obtain the Causal Graph Matrix and Save it as a Numpy Array


```python
pc = PC(priori_knowledge=prior_knowledge)
pc.learn(samples)
```


```python
graph_matrix = np.array(pc.causal_matrix)
```


```python
graph_matrix.shape
```




    (39, 39)




```python
np.save(r'./submission/dataset_1_graph_matrix.npy',graph_matrix)
```
