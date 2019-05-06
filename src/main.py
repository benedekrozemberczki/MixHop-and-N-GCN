from utils import tab_printer
from parser import parameter_parser
from learning_to_evaluate import CharacteristicFunctionNetworkTrainer

def main():
    """
    Parsing command line parameters, reading data, doing sparsification, fitting a GWNN and saving the logs.
    """
    args = parameter_parser()
    tab_printer(args)
    trainer = CharacteristicFunctionNetworkTrainer(args)
    trainer.fit()
    trainer.score()

if __name__ == "__main__":
    main()
