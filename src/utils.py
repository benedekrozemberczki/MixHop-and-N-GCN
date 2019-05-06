from texttable import Texttable
import json
from tqdm import tqdm

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())


def enumerate_graphs(graphs):
    labels = set()
    for graph in tqdm(graphs):
        data = json.load(open(graph))
        labels = labels.union(set([data["label"]]))
    labels = {label: i for i, label in enumerate(labels)}
    return labels
