import torch
import random
from tqdm import trange
from utils import create_propagator_matrix
from ngcn_layer import NGCNLayer, ListModule

class NGCNNetwork(torch.nn.Module):
    def __init__(self, args, feature_number, class_number):

        super(NGCNNetwork, self).__init__()
        self.args = args
        self.feature_number = feature_number
        self.class_number = class_number
        self.order = len(self.args.layers)
        self.setup_layer_structure()

    def setup_layer_structure(self):
        """
        Creating the layer structure (3 convolutional layers).
        """
        self.main_layers = [NGCNLayer(self.feature_number, self.args.layers[i-1], i, self.args.dropout) for i in range(1, self.order+1)]
        self.main_layers = ListModule(*self.main_layers)
        self.fully_connected = torch.nn.Linear(sum(self.args.layers), self.class_number)


    def forward(self, normalized_adjacency_matrix, features):
        abstract_features = torch.cat([self.main_layers[i](normalized_adjacency_matrix, features) for i in range(self.order)],dim=1)
        predictions =  torch.nn.functional.log_softmax(self.fully_connected(abstract_features),dim=1)
        return predictions



class NGCNTrainer(object):
    """
    Class for training the neural network.
    :param args: Arguments object.
    :param graph: NetworkX graph.
    :param features: Feature sparse matrix.
    :param target: Target vector.
    """
    def __init__(self, args, graph, features, target):
        self.args = args
        self.graph = graph
        self.features = features
        self.target = target
        self.setup_features()
        self.train_test_split()
        self.setup_model()

    def train_test_split(self):
        """
        Creating a train/test split.
        """
        random.seed(self.args.seed)
        nodes = [node for node in range(self.ncount)]
        random.shuffle(nodes)
        self.train_nodes = torch.LongTensor(nodes[0:self.args.training_size])
        self.validation_nodes = torch.LongTensor(nodes[self.args.training_size:self.args.training_size+self.args.validation_size])
        self.test_nodes = torch.LongTensor(nodes[self.args.training_size+self.args.validation_size:])

    def setup_features(self):
        """
        Creating a feature matrix, target vector and propagation matrix.
        """
        self.ncount = self.features["dimensions"][0]
        self.feature_number = self.features["dimensions"][1]
        self.class_number = torch.max(self.target).item()+1
        self.propagation_matrix = create_propagator_matrix(self.graph)

    def setup_model(self):
        """
        Defining a PageRankNetwork.
        """
        self.model = NGCNNetwork(self.args, self.feature_number, self.class_number)

    def fit(self):
        """
        Fitting a neural network with early stopping.
        """
        accuracy = 0
        no_improvement = 0
        epochs = trange(self.args.epochs, desc="Accuracy")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model.train()
        for epoch in epochs:
            self.optimizer.zero_grad()
            prediction = self.model(self.propagation_matrix, self.features)
            loss = torch.nn.functional.nll_loss(prediction[self.train_nodes], self.target[self.train_nodes])
            loss.backward()
            self.optimizer.step()
            new_accuracy = self.score(self.validation_nodes)
            epochs.set_description("Validation Accuracy: %g" % round(new_accuracy,4))
            if new_accuracy < accuracy:
                no_improvement = no_improvement + 1
                if no_improvement == self.args.early_stopping:
                    epochs.close()
                    break
            else:
                no_improvement = 0
                accuracy = new_accuracy               
        acc = self.score(self.test_nodes)
        print("\nTest accuracy: " + str(round(acc,4)) )

    def score(self, indices):
        """
        Scoring a neural network.
        :param indices: Indices of nodes involved in accuracy calculation.
        :return acc: Accuracy score.
        """
        self.model.eval()
        _, prediction = self.model(self.propagation_matrix, self.features).max(dim=1)
        correct = prediction[indices].eq(self.target[indices]).sum().item()
        acc = correct / indices.shape[0]
        return acc
