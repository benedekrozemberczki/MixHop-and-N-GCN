import torch
import random
from tqdm import trange
from utils import create_propagator_matrix
from layers import SparseNGCNLayer, DenseNGCNLayer, ListModule

class NGCNNetwork(torch.nn.Module):
    """
    Higher Order Graph Convolutional Model.
    :param args: Arguments object.
    :param feature_number: Feature input number.
    :param class_number: Target class number.
    """
    def __init__(self, args, feature_number, class_number):
        super(NGCNNetwork, self).__init__()
        self.args = args
        self.feature_number = feature_number
        self.class_number = class_number
        self.order = len(self.args.layers_1)
        self.setup_layer_structure()

    def setup_layer_structure(self):
        """
        Creating the layer structure (3 convolutional layers) and dense final.
        """
        self.main_layers = [SparseNGCNLayer(self.feature_number, self.args.layers_1[i-1], i, self.args.dropout) for i in range(1, self.order+1)]
        self.main_layers = ListModule(*self.main_layers)
        self.fully_connected = torch.nn.Linear(sum(self.args.layers_1), self.class_number)

    def forward(self, normalized_adjacency_matrix, features):
        """
        Forward pass.
        :param normalized adjacency_matrix: Target matrix as a dict with indices and values.
        :param features: Feature matrix.
        :return predictions: Label predictions.
        """
        abstract_features = torch.cat([self.main_layers[i](normalized_adjacency_matrix, features) for i in range(self.order)],dim=1)
        predictions =  torch.nn.functional.log_softmax(self.fully_connected(abstract_features),dim=1)
        return predictions

class MixHopNetwork(torch.nn.Module):
    """
    MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing.
    :param args: Arguments object.
    :param feature_number: Feature input number.
    :param class_number: Target class number.
    """
    def __init__(self, args, feature_number, class_number):
        super(MixHopNetwork, self).__init__()
        self.args = args
        self.feature_number = feature_number
        self.class_number = class_number
        self.calculate_layer_sizes()
        self.setup_layer_structure()

    def calculate_layer_sizes(self):
        self.abstract_feature_number_1 = sum(self.args.layers_1)
        self.abstract_feature_number_2 = sum(self.args.layers_2)
        self.order_1 = len(self.args.layers_1)
        self.order_2 = len(self.args.layers_2)

    def setup_layer_structure(self):
        """
        Creating the layer structure (3 convolutional upper layers, 3 bottom layers) and dense final.
        """
        self.upper_layers = [SparseNGCNLayer(self.feature_number, self.args.layers_1[i-1], i, self.args.dropout) for i in range(1, self.order_1+1)]
        self.upper_layers = ListModule(*self.upper_layers)
        self.bottom_layers = [DenseNGCNLayer(self.abstract_feature_number_1, self.args.layers_2[i-1], i, self.args.dropout) for i in range(1, self.order_2+1)]
        self.bottom_layers = ListModule(*self.bottom_layers)
        self.fully_connected = torch.nn.Linear(self.abstract_feature_number_2, self.class_number)

    def calculate_group_loss(self):
        """
        Calculating the column losses.
        """
        weight_loss = 0
        for i in range(self.order_1):
            upper_column_loss = torch.norm(self.upper_layers[i].weight_matrix, dim=0)
            loss_upper = torch.sum(upper_column_loss)
            weight_loss = weight_loss + self.args.lambd*loss_upper
        for i in range(self.order_2):
            bottom_column_loss = torch.norm(self.bottom_layers[i].weight_matrix, dim=0)
            loss_bottom = torch.sum(bottom_column_loss)
            weight_loss = weight_loss + self.args.lambd*loss_bottom
        return weight_loss

    def calculate_loss(self):
        """
        Calculating the losses.
        """
        weight_loss = 0
        for i in range(self.order_1):
            loss_upper = torch.norm(self.upper_layers[i].weight_matrix)
            weight_loss = weight_loss + self.args.lambd*loss_upper
        for i in range(self.order_2):
            loss_bottom = torch.norm(self.bottom_layers[i].weight_matrix)
            weight_loss = weight_loss + self.args.lambd*loss_bottom
        return weight_loss
            

    def forward(self, normalized_adjacency_matrix, features):
        """
        Forward pass.
        :param normalized adjacency_matrix: Target matrix as a dict with indices and values.
        :param features: Feature matrix.
        :return predictions: Label predictions.
        """
        abstract_features_1 = torch.cat([self.upper_layers[i](normalized_adjacency_matrix, features) for i in range(self.order_1)],dim=1)
        abstract_features_2 = torch.cat([self.bottom_layers[i](normalized_adjacency_matrix, abstract_features_1) for i in range(self.order_2)],dim=1)
        predictions =  torch.nn.functional.log_softmax(self.fully_connected(abstract_features_2),dim=1)
        return predictions

class Trainer(object):
    """
    Class for training the neural network.
    :param args: Arguments object.
    :param graph: NetworkX graph.
    :param features: Feature sparse matrix.
    :param target: Target vector.
    """
    def __init__(self, args, graph, features, target, base_run):
        self.args = args
        self.graph = graph
        self.features = features
        self.target = target
        self.base_run = base_run
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
        if self.args.model == "mixhop":
            self.model = MixHopNetwork(self.args, self.feature_number, self.class_number)
        else:
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
            if self.args.model == "mixhop" and self.base_run == True:
                loss = loss + self.model.calculate_group_loss()
            elif self.args.model == "mixhop" and self.base_run == False:
                loss = loss + self.model.calculate_loss()
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
        print("\nTest accuracy: " + str(round(acc,4)) +"\n")
       
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

    def evaluate_architecture(self):
        """
        Making a choice about the optimal layer sizes.
        """
        print("The best architecture is:\n")
        self.layer_sizes = dict()

        self.layer_sizes["upper"] = []

        for layer in self.model.upper_layers:
            norms = torch.norm(layer.weight_matrix**2, dim=0)
            norms = norms[norms<self.args.cut_off]
            self.layer_sizes["upper"].append(norms.shape[0])

        self.layer_sizes["bottom"] = []

        for layer in self.model.bottom_layers:
            norms = torch.norm(layer.weight_matrix**2, dim=0)
            norms = norms[norms<self.args.cut_off]
            self.layer_sizes["bottom"].append(norms.shape[0])

        self.layer_sizes["upper"] = [int(self.args.budget*layer_size/sum(self.layer_sizes["upper"]))  for layer_size in self.layer_sizes["upper"]]
        self.layer_sizes["bottom"] = [int(self.args.budget*layer_size/sum(self.layer_sizes["bottom"]))  for layer_size in self.layer_sizes["bottom"]]
        print("Layer 1.: "+str(tuple(self.layer_sizes["upper"])))
        print("Layer 2.: "+str(tuple(self.layer_sizes["bottom"])))

    def reset_architecture(self):
        """
        Changing the layer sizes.
        """
        print("\nResetting the architecture.\n")
        self.args.layers_1 = self.layer_sizes["upper"]
        self.args.layers_2 = self.layer_sizes["bottom"]
        return self.args
 
        

