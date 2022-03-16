# Code based on the tutorial at https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html

# Helper function for visualization.
#%matplotlib inline
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from torch.utils.cpp_extension import CUDA_HOME
print(CUDA_HOME)

from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from torch_geometric.datasets import *
from torch_geometric.profile import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_dataset_stats(dataset):
  data = dataset[0]  # Get the first graph object.

  print("Dataset STATS", data)
  print('==============================================================')

  # Gather some statistics about the graph.
  print(f'Number of nodes: {data.num_nodes}')
  print(f'Number of edges: {data.num_edges}')
  print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
  print(f'Number of training nodes: {data.train_mask.sum()}')
  print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
  print(f'Has isolated nodes: {data.has_isolated_nodes()}')
  print(f'Has self-loops: {data.has_self_loops()}')
  print(f'Is undirected: {data.is_undirected()}')
  
def get_info(model, data):
  # https://pytorch-geometric.readthedocs.io/en/latest/modules/profile.html?highlight=profileit#torch_geometric.profile.get_cpu_memory_from_gc
  print("MODEL INFO", model, data)
  print("trainable parameters count", count_parameters(model))
  print("model size in bytes", get_model_size(model))
  print("Data theoretical memory usage in bytes", get_data_size(data))
  print("CPU memory", get_cpu_memory_from_gc())
  print("GPU memory", get_gpu_memory_from_gc(0))
  print("nvidia-smi free and used GPU memory", get_gpu_memory_from_nvidia_smi())
  
"""
class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True)

    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        x = F.relu(x)
        x = self.conv2(x, adj_t)
        return F.log_softmax(x, dim=1)
"""

def run_original(dataset):
  class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True)

    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        x = F.relu(x)
        x = self.conv2(x, adj_t)
        return F.log_softmax(x, dim=1)

  #dataset = Planetoid("Planetoid", name="Cora")#, transform=T.ToSparseTensor())
  data = dataset[0]

  model = GNN()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

  print("RUNNING ORIGINAL")
  #get_dataset_stats(dataset)
  get_info(model, data)

  @profileit()
  def train(model, data):
      model.train()
      optimizer.zero_grad()
      out = model(data.x, data.edge_index)#adj_t)
      loss = F.nll_loss(out, data.y)
      loss.backward()
      optimizer.step()
      return float(loss)

  all_stats = []
  for epoch in range(1, 201):
      loss, stats = train(model, data)
      #print(stats)
      all_stats.append(stats)
  return all_stats
  
"""
class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True)

    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        x = F.relu(x)
        x = self.conv2(x, adj_t)
        return F.log_softmax(x, dim=1)
"""

def run_sparsetensor(dataset2):
  class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset2.num_features, 16, cached=True)
        self.conv2 = GCNConv(16, dataset2.num_classes, cached=True)

    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        x = F.relu(x)
        x = self.conv2(x, adj_t)
        return F.log_softmax(x, dim=1)

  #dataset2 = Planetoid("Planetoid", name="Cora", transform=T.ToSparseTensor())
  data2 = dataset2[0]

  model2 = GNN()
  optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)

  print("RUNNING SPARSETENSOR")
  #get_dataset_stats(dataset2)
  get_info(model2, data2)

  @profileit()
  def train(model, data):
      model.train()
      optimizer.zero_grad()
      out = model(data.x, data.adj_t)
      loss = F.nll_loss(out, data.y)
      loss.backward()
      optimizer.step()
      return float(loss)

  all_stats = []
  for epoch in range(1, 201):
      loss, stats = train(model2, data2)
      #print(stats)
      all_stats.append(stats)
  return (all_stats)
  
def run_all(name, dataset1, dataset2):

  print("DATASET STATS OF ORIGINAL")
  get_dataset_stats(dataset1)
  print()
  print("DATASET STATS OF SPARSETENSOR")
  get_dataset_stats(dataset2)

  all_stats1 = run_original(dataset1)
  stats1_sum = get_stats_summary(all_stats1)
  print("STATS_SUM1", stats1_sum)
  times1 = [all_stats1[i].time for i in range(len(all_stats1))]

  all_stats2 = run_sparsetensor(dataset2)
  stats2_sum = get_stats_summary(all_stats2)
  print("STATS_SUM2", stats2_sum)
  times2 = [all_stats2[i].time for i in range(len(all_stats2))]
  print("SPEEDUP of", stats1_sum.time_mean/stats2_sum.time_mean)
  plt.figure()
  plt.title(name)
  plt.plot(range(1, len(times1)), times1[1:])
  plt.plot(range(1, len(times2)), times2[1:])
  plt.axhline(y=stats1_sum.time_mean, color='g', linestyle='-')
  plt.axhline(y=stats2_sum.time_mean, color='r', linestyle='-')
  plt.legend(["Original", "with SparseTensor"])
  plt.xlabel("Training iteration")
  plt.ylabel("Time (s)")
  plt.savefig(f'{name}.png', bbox_inches='tight')
  plt.plot()
  plt.show()
  
run_all("KarateClub", KarateClub(), KarateClub(transform=T.ToSparseTensor()))

run_all("Planetoid-Cora", Planetoid("Planetoid", name="Cora"), Planetoid("Planetoid", name="Cora", transform=T.ToSparseTensor()))

run_all("Planetoid-CiteSeer", Planetoid("Planetoid", name="CiteSeer"), Planetoid("Planetoid", name="CiteSeer", transform=T.ToSparseTensor()))

run_all("Planetoid-PubMed", Planetoid("Planetoid", name="PubMed"), Planetoid("Planetoid", name="PubMed", transform=T.ToSparseTensor()))