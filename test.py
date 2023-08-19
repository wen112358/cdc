import matplotlib.pyplot as plt
import numpy

matrix = numpy.load("submission/baseline/dataset_1_graph_matrix.npy")

plt.imshow(matrix)
plt.show()