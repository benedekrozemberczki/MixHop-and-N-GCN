import torch
from trainer_and_networks import Trainer
from parser import parameter_parser
from utils import tab_printer, graph_reader, feature_reader, target_reader

def main():
    """
    Parsing command line parameters, reading data, fitting an NGCN and scoring the model.
    """
    args = parameter_parser()
    torch.manual_seed(args.seed)
    tab_printer(args)
    graph = graph_reader(args.edge_path)
    features = feature_reader(args.features_path)
    target = target_reader(args.target_path)
    trainer = Trainer(args, graph, features, target)
    trainer.fit()

if __name__ == "__main__":
    main()
