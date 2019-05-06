import argparse

def parameter_parser():
    """
    A method to parse up command line parameters. By default it trains on the Cora dataset.
    The default hyperparameters give a good quality representation without grid search.
    """

    parser = argparse.ArgumentParser(description = "Run MixHop/N-GCN.")

    parser.add_argument("--edge-path",
                        nargs = "?",
                        default = "./input/cora_edges.csv",
	                help = "Edge list csv.")

    parser.add_argument("--features-path",
                        nargs = "?",
                        default = "./input/cora_features.json",
	                help = "Features json.")

    parser.add_argument("--target-path",
                        nargs = "?",
                        default = "./input/cora_target.csv",
	                help = "Target classes csv.")

    parser.add_argument("--model",
                        nargs = "?",
                        default = "mixhop",
	                help = "Target classes csv.")

    parser.add_argument("--epochs",
                        type = int,
                        default = 2000,
	                help = "Number of training epochs. Default is 2000.")

    parser.add_argument("--seed",
                        type = int,
                        default = 42,
	                help = "Random seed for train-test split. Default is 42.")

    parser.add_argument("--early-stopping",
                        type = int,
                        default = 10,
	                help = "Number of early stopping rounds. Default is 10.")

    parser.add_argument("--training-size",
                        type = int,
                        default = 1500,
	                help = "Training set size. Default is 1500.")

    parser.add_argument("--validation-size",
                        type = int,
                        default = 500,
	                help = "Validation set size. Default is 500.")

    parser.add_argument("--dropout",
                        type = float,
                        default = 0.5,
	                help = "Dropout parameter. Default is 0.5.")

    parser.add_argument("--learning-rate",
                        type = float,
                        default = 0.01,
	                help = "Learning rate. Default is 0.01.")

    parser.add_argument("--lambd",
                        type = float,
                        default = 0.0001,
	                help = "L2 regularization coefficient. Default is 0.0001.")

    parser.add_argument("--layers",
                        nargs="+",
                        type=int,
                        help = "Layer dimensions separated by space. E.g. 64 64.")

    parser.set_defaults(layers = [32, 32, 32])
    
    return parser.parse_args()

