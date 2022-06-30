import argparse
import datetime
import json
import pickle

import numpy as np

from .mc import MarkovChainLobModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_levels', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_events', type=int, default=100000)
    parser.add_argument('--num_initial_events', type=int, default=100000)
    parser.add_argument('--data_path', type=str, default="./data")
    parser.add_argument('--filename', type=str, default="mc_data")

    return parser.parse_args()


def main(args):
    # set seed
    np.random.seed(args.seed)

    # model info
    args.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

    # set up a start LOB
    x0 = np.zeros((2, args.num_levels + 1), dtype=int)
    x0[0, 1:] = 1
    x0[1, :] = -1

    mc_model = MarkovChainLobModel(num_levels=args.num_levels, ob_start=x0)

    # add rates as lists to info file as well
    args.rates = {}
    for k, v in mc_model.rates.items():
        if isinstance(v, np.ndarray):
            args.rates[k] = list(v)
        else:
            args.rates[k] = v
    args.event_types = mc_model.event_types

    # save info file
    with open(args.data_path + "//" + args.filename + "_info.txt", 'w') as file:
        json.dump(args.__dict__, file, indent=2)

    # simulate away from initial
    _ = mc_model.simulate(num_events=args.num_initial_events)

    # simulate events and save
    data_dict = mc_model.simulate(num_events=args.num_events)
    data_dict["event_types"] = mc_model.event_types

    pickle.dump(data_dict, open("{}//{}.pkl".format(args.data_path, args.filename), "wb"))


if __name__ == '__main__':
    main(parse_args())
