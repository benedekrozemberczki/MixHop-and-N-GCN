import random
import torch
import glob
import json
import sparse
from scipy import sparse
from utils import enumerate_graphs
from tqdm import tqdm, trange
from layers import CharacteristicFunctionNetwork
from sklearn.model_selection import train_test_split
import numpy as np

class CharacteristicFunctionNetworkTrainer(object):

    def __init__(self, args):
        self.args = args
        self._enumerate_labels()
        self._create_train_test_split()
        self._setup_model()

    def _enumerate_labels(self):
        print("\nEnumerating labels.\n")
        self.graphs = glob.glob(self.args.graphs+"*.json")
        self.label_map = enumerate_graphs(self.graphs)
        self.number_of_labels = len(self.label_map)
        print(self.number_of_labels)

    def _create_train_test_split(self):
        self.train_graphs, self.test_graphs = train_test_split(self.graphs, test_size=self.args.test_size, random_state=self.args.seed)

    def _setup_model(self):
        self.model = CharacteristicFunctionNetwork(self.args, self.number_of_labels)

    def _create_batches(self):
        random.shuffle(self.graphs)
        self.batches = [self.train_graphs[i:i + self.args.batch_size] for i in range(0,len(self.train_graphs), self.args.batch_size)]


    def _read_input_data(self, path):
        raw_data = json.load(open(path))         
        index_1 = [edge[0] for edge in raw_data["edges"]]
        index_2 = [edge[1] for edge in raw_data["edges"]]
        index_1, index_2 = index_1+index_2, index_2+index_1
        values = [1.0 for index, edge in enumerate(index_1)]
        node_count = max(max(index_1),max(index_1))+1
        nodes = sparse.csr_matrix(sparse.coo_matrix((values, (index_1, index_2)), shape = (node_count,node_count), dtype = np.float32))
        nodes = np.array(nodes.todense())

        target = np.array([raw_data["label"]])
        return nodes, target

    def calculate_transition(self, raw_data):
        adjacency = torch.FloatTensor(raw_data)
        adjacency_hat = adjacency/adjacency.sum(dim=0)
        adjacency_hats = [adjacency_hat]
        for power in range(self.args.order-1):
            adjacency_hats.append(torch.mm(adjacency_hats[power],adjacency_hat))
        adjacency_hats = torch.stack(adjacency_hats)

        return adjacency_hats,2
        

    def fit(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        for epoch in tqdm(range(self.args.epochs), desc = "Epochs: ", leave = True):
            self._create_batches()
            losses = 0       
            self.steps = trange(len(self.batches), desc="Loss")
            for step in self.steps:
                accumulated_losses = 0
                optimizer.zero_grad()
                batch = self.batches[step]
                for path in batch:
                    nodes, target = self._read_input_data(path)
                    target = torch.LongTensor(target)
                    feature_matrix, _ = self.calculate_transition(nodes)
                    prediction = self.model(feature_matrix)
                    loss = torch.nn.functional.nll_loss(prediction, target)
                    accumulated_losses = accumulated_losses + loss
                accumulated_losses = accumulated_losses/len(batch)
                accumulated_losses.backward()
                optimizer.step()
                losses = losses + accumulated_losses.item()
                average_loss = losses/(step + 1)
                self.steps.set_description("CapsGNN (Loss=%g)" % round(average_loss,4))    

    def score(self):
        self.model.eval()
        predictions = []
        for graph in self.test_graphs:
            nodes, target = self._read_input_data(graph)
            target = torch.LongTensor(target)
            feature_matrix, _ = self.calculate_transition(nodes)
            prediction,x = self.model(feature_matrix).max(1)
            predictions.append(x == target)
        predictions = sum([p.item() for p in predictions])/len(self.test_graphs)
        print("\n")
        print(predictions)
    
