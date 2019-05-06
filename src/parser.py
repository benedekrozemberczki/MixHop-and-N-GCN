
import argparse

def parameter_parser():
    """
    A method to parse up command line parameters. By default it learns on the Watts-Strogatz dataset.
    The default hyperparameters give good results without cross-validation.
    """
    parser = argparse.ArgumentParser(description = "Run CapsGNN.")
	
    parser.add_argument("--graphs",
                        nargs = "?",
                        default = "./input/watts/",
	                help = "Training graphs folder.")

    parser.add_argument("--seed",
                        type = int,
                        default = 42,
	                help = "Number of graphs processed per batch. Default is 32.")

    parser.add_argument("--epochs",
                        type = int,
                        default = 10,
	                help = "Number of training epochs. Default is 100.")

    parser.add_argument("--batch-size",
                        type = int,
                        default = 32,
	                help = "Number of graphs processed per batch. Default is 32.")

    parser.add_argument("--number_of_evaluation_points",
                        type = int,
                        default = 30,
	                help = "Number of Graph Convolutional filters. Default is 20.")

    parser.add_argument("--order",
                        type = int,
                        default = 5,
	                help = "Number of Graph Convolutional Layers. Default is 2.")

    parser.add_argument("--dense-1-dimensions",
                        type = int,
                        default = 20,
	                help = "Number of Attention Neurons. Default is 20.")

    parser.add_argument("--dense-2-dimensions",
                        type = int,
                        default = 10,
	                help = "Capsule dimensions. Default is 8.")

    parser.add_argument("--pooling-1-dimensions",
                        type = int,
                        default = 10,
	                help = "Number of capsules per layer. Default is 8.")

    parser.add_argument("--pooling-2-dimensions",
                        type = int,
                        default = 10,
	                help = "Number of capsules per layer. Default is 8.")

    parser.add_argument("--weight-decay",
                        type = float,
                        default = 10**-4,
	                help = "Weight decay. Default is 10^-6.")

    parser.add_argument("--learning-rate",
                        type = float,
                        default = 0.01,
	                help = "Learning rate. Default is 0.01.")


    parser.add_argument("--test-size",
                        type = float,
                        default = 0.1,
	                help = "Reconstruction loss weight. Default is 0.1.")
    
    return parser.parse_args()
