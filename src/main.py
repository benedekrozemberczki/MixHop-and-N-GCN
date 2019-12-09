"""Running MixHop or N-GCN."""

import torch
from param_parser import parameter_parser
from trainer_and_networks import Trainer
from utils import tab_printer, graph_reader, feature_reader, target_reader

def main():
    """
    Parsing command line parameters, reading data.
    Fitting an NGCN and scoring the model.
    """
    args = parameter_parser()
    torch.manual_seed(args.seed)
    tab_printer(args)
    graph = graph_reader(args.edge_path)
    features = feature_reader(args.features_path)
    target = target_reader(args.target_path)
    trainer = Trainer(args, graph, features, target, True)
    trainer.fit()
    if args.model == "mixhop":
        trainer.evaluate_architecture()
        args = trainer.reset_architecture()
        trainer = Trainer(args, graph, features, target, False)
        trainer.fit()

if __name__ == "__main__":
    main()
