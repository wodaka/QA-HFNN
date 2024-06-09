import argparse
from pprint import pprint

def parse_cli_parameters():
    parser = argparse.ArgumentParser(description="FuzzyDNN on CIFAR-10")
    parser.add_argument('--learning-rate', dest='learning_rate', default=10 ** -3, type=float,
                        help='Learning Rate of your classifier. Default 0.001')
    parser.add_argument('--epoch', dest='epochs', default=100, type=int,
                        help='Number of times you want to train your data. Default 100')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=16,
                        help='Batch size for prediction. Default=16.')
    parser.add_argument('--colour-image', dest='is_colour_image', action="store_true", default=False,
                        help='Passing this argument will keep the coloured image (RGB) during training. Default=False.')
    parser.add_argument('--membership-layer-units', dest='membership_layer_units', type=int, default=100,
                        help='Defines the number of units/nodes in the Membership Function Layer')
    parser.add_argument('--first-dr-layer-units', dest='dr_layer_1_units', type=int, default=100,
                        help='Defines the number of units in the first DR Layer')
    parser.add_argument('--second-dr-layer-units', dest='dr_layer_2_units', type=int, default=100,
                        help='Defines the number of units in the second DR Layer')
    parser.add_argument('--fusion-dr-layer-units', dest='fusion_dr_layer_units', type=int, default=100,
                        help='Defines the number of units in the Fusion DR Layer')
    parser.add_argument('--hide-graph', dest='should_hide_graph', action="store_true", default=False,
                        help='Hides the graph of results displayed via matplotlib')

    options = parser.parse_args()

    print("Starting with the following options:")
    pprint(vars(options))
    return options

class cli_parameters():
    def __init__(self):
        learning_rate = 10 ** -3
        epochs = 100
        batch_size = 16
        is_colour_image = False
        membership_layer_units = 100
        dr_layer_1_units = 100
        dr_layer_2_units = 100
        fusion_dr_layer_units = 100
        should_hide_graph = False