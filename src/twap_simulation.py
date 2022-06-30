import argparse
import datetime
import json
import os
import pickle
import time

import numpy as np
import tensorflow as tf

from .mc import MarkovChainLobModel
from .mc_functions import load_data_mc, setup_model_spec_mc
from .rnn import LobRNN

tf.compat.v1.disable_eager_execution()


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume', type=int, default=5)
    parser.add_argument('--timedelta', type=int, default=10)
    parser.add_argument('--timefactor', type=int, default=10)
    parser.add_argument('--num_runs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default="-1", choices=["0", "1", "-1"])
    parser.add_argument('--checkpoint_timestamp', type=str, required=True)

    return parser.parse_args()


def simulate_twaps(twap_function, num_runs, volume, end_time, timefactor, start_dict, padding=(10, 10), **kwargs):
    num_times = int(end_time * timefactor) + 2
    twap_dict = {}
    twap_dict["num_tries"] = 0
    twap_dict["price"] = np.nan * np.zeros((num_runs, 1))
    twap_dict["num_events"] = np.zeros((num_runs, 1))
    twap_dict["num_mo"] = np.zeros((num_runs, 1))
    twap_dict["num_mo_bid"] = np.zeros((num_runs, 1))
    twap_dict["num_mo_ask"] = np.zeros((num_runs, 1))
    twap_dict["num_lo_filled"] = np.zeros((num_runs, 1))
    twap_dict["mid_vals"] = np.zeros((num_runs, num_times))
    twap_dict["ask_vals"] = np.zeros((num_runs, num_times))
    twap_dict["bid_vals"] = np.zeros((num_runs, num_times))
    twap_dict["time_vals"] = np.zeros((num_runs, num_times + 1))

    initial_dict = {}
    for i in range(num_runs):
        print('iteration: ', i)
        finished = False
        while not finished:
            twap_dict["num_tries"] += 1
            for k, v in start_dict.items():
                if k == "ob":
                    initial_dict[k] = v[i].copy().squeeze()
                else:
                    initial_dict[k] = v[i].copy()
            t0 = time.time()
            twap_tmp = twap_function(volume, end_time, timefactor, initial_dict, padding)
            t1 = time.time()
            print('time: ', t1 - t0)
            finished = twap_tmp["finished"]
            if finished:
                for k, v in twap_tmp.items():
                    if k != "finished":
                        twap_dict[k][i, :] = v
    if not twap_dict["num_mo"].any():
        twap_dict["num_mo"] = twap_dict["num_mo_bid"] + twap_dict["num_mo_ask"]

    return twap_dict


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    args.padding = (args.timedelta, args.timedelta)
    args.end_time = args.timedelta * (args.volume + 1)
    args.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    p = os.path.abspath(__file__ + "/../../")
    args.data_path = os.path.join(p, "data/twap_simulations/" + "twap_" + args.timestamp)
    rnn_path = os.path.join(p, 'models/checkpoints/mc_lob_rnn_{}'.format(args.checkpoint_timestamp))
    args.rnn_info_path = os.path.join(rnn_path, 'model_info.txt')
    args.rnn_checkpoint_path = os.path.join(rnn_path, 'lob_{}_lstm_predict_model.hdf5')

    with open(args.rnn_info_path) as file:
        rnn_info = json.load(file)

    # get start LOB data from training data
    data, event_types, rnn_info = load_data_mc(rnn_info)
    args.start_lob = os.path.join(p, 'data/mc_train.pkl')
    ob_data = data["ob"][0]
    start_len = rnn_info["seq_length"]
    start_ind = np.random.choice(ob_data.shape[0] - start_len, args.num_runs, replace=False)
    args.start_dict = {"ob": [ob_data[i:i + start_len, ...].astype(int) for i in start_ind]}
    for k, v in data.items():
        if k != "ob":
            args.start_dict[k] = [v[0][i:i + start_len] for i in start_ind]

    mc_model = MarkovChainLobModel(num_levels=args.start_dict["ob"][0].shape[-1] - 1,
                                   ob_start=args.start_dict["ob"][0][-1, ...].copy())


    # simulate TWAPs using MC LOB model

    def mc_twap(v, e, t, d, p):
        d = d.copy()
        for k in d:
            d[k] = d[k][-1, ...]
        return mc_model.run_twap(v, e, t, d, p)


    mc_dict = simulate_twaps(mc_twap, **vars(args))
    mc_start_mid = mc_dict["mid_vals"][:, 0, np.newaxis]
    mc_dict["mid_vals"] = mc_dict["mid_vals"] - mc_start_mid
    mc_dict["bid_vals"] = mc_dict["bid_vals"] - mc_start_mid
    mc_dict["ask_vals"] = mc_dict["ask_vals"] - mc_start_mid
    mc_dict["price"] = mc_dict["price"] - args.volume * mc_start_mid

    print('mc model number of tries: ', mc_dict["num_tries"])

    # simulate TWAPs using RNN trained on MC model data
    rnn_args = Namespace(**rnn_info)
    data, event_types, rnn_args = load_data_mc(rnn_args)
    model_spec = setup_model_spec_mc(rnn_args)
    rnn = LobRNN(data, model_spec, event_types, integer_size=True, outside_volume=1)
    model = rnn.setup_model(rnn_args, predict=True, device='/cpu:0')
    for k, v in model.items():
        v.load_weights(args.rnn_checkpoint_path.format(k))

    rnn_twap_function = lambda v, et, tf, x0, pad: rnn.run_twap(model, v, et, tf, x0, pad)
    rnn_dict = simulate_twaps(rnn_twap_function, **vars(args))

    start_mid_rnn = rnn_dict["mid_vals"][:, 0, np.newaxis]
    rnn_dict["mid_vals"] = rnn_dict["mid_vals"] - start_mid_rnn
    rnn_dict["bid_vals"] = rnn_dict["bid_vals"] - start_mid_rnn
    rnn_dict["ask_vals"] = rnn_dict["ask_vals"] - start_mid_rnn
    rnn_dict["price"] = rnn_dict["price"] - args.volume * start_mid_rnn

    print('rnn model number of tries: ', rnn_dict["num_tries"])

    # save results
    del args.start_dict
    dict_to_save = {"info": args.__dict__, "rnn_info": rnn_info, "mc_dict": mc_dict, "rnn_dict": rnn_dict}
    pickle.dump(dict_to_save, open(args.data_path + ".pkl", "wb"))

    with open(args.data_path + '_info.txt', 'w') as file:
        file.write(json.dumps(args.__dict__))
