import torch
import random
from tqdm import tqdm

class LearningToEvaluateLayer(torch.nn.Module):

    def __init__(self, number_of_evaluation_points, order):
        super(LearningToEvaluateLayer, self).__init__()
        self.number_of_evaluation_points = number_of_evaluation_points
        self.order = order
        self._setup_weights()
        

    def _setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.order,self.number_of_evaluation_points))
        torch.nn.init.uniform_(self.weight_matrix,-1000,1000) 


    def forward(self, normalzed_adjacency_tensor):
        number_of_nodes = normalzed_adjacency_tensor.shape[1]
        multi_scale_features = []
        for i in range(self.order):
            square_tensor = normalzed_adjacency_tensor[i,:,:].squeeze()
            thetas = self.weight_matrix[i,:].view(-1)
            scores = torch.ger(thetas, square_tensor.view(-1)).view(self.number_of_evaluation_points, number_of_nodes, number_of_nodes)
            non_linear_scores = torch.cos(scores)
            features = torch.t(torch.mean(non_linear_scores, dim=2))
            multi_scale_features.append(features)
        multi_scale_features = torch.cat(tuple(multi_scale_features),dim=1)
        return multi_scale_features

class SagePoolingLayer(torch.nn.Module):

    def __init__(self, args):
        super(SagePoolingLayer,self).__init__()
        self.args = args
        self._setup()

    def _setup(self):
        self.fully_connected_1 = torch.nn.Linear(self.args.dense_2_dimensions, self.args.pooling_1_dimensions)
        self.fully_connected_2 = torch.nn.Linear(self.args.pooling_1_dimensions, self.args.pooling_2_dimensions)

    def forward(self, features):

        abstract_features = torch.tanh(self.fully_connected_1(features))
        attention = torch.nn.functional.softmax(self.fully_connected_2(abstract_features),dim=0)
        graph_embedding = torch.mm(torch.t(attention), features)
        graph_embedding = graph_embedding.view(1,-1)
        return graph_embedding



class CharacteristicFunctionNetwork(torch.nn.Module):

    def __init__(self, args, number_of_labels):
        super(CharacteristicFunctionNetwork, self).__init__()
        self.args = args
        self.number_of_labels = number_of_labels
        self._setup_model()

    def _setup_model(self):
        self.characteristic_function_evaluation_layer = LearningToEvaluateLayer(self.args.number_of_evaluation_points, self.args.order)
        self.dense_layer_1 = torch.nn.Linear(self.args.number_of_evaluation_points*self.args.order,self.args.dense_1_dimensions)
        self.dense_layer_2 = torch.nn.Linear(self.args.dense_1_dimensions, self.args.dense_2_dimensions)
        self.pooling_layer_1 = SagePoolingLayer(self.args)
        self.classification_layer = torch.nn.Linear(self.args.pooling_2_dimensions*self.args.dense_2_dimensions, self.number_of_labels)

    def forward(self, normalized_adjacency_tensor):

        multi_scale_features = self.characteristic_function_evaluation_layer(normalized_adjacency_tensor)
        node_level_features_1 = torch.relu(self.dense_layer_1(multi_scale_features))
        node_level_features_2 = torch.relu(self.dense_layer_2(node_level_features_1))
        graph_level_features = self.pooling_layer_1(node_level_features_2)
        predictions = self.classification_layer(graph_level_features)
        predictions = torch.nn.functional.log_softmax(predictions,dim=1)
        return predictions
