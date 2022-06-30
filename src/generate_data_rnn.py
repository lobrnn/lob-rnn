import argparse
import datetime
import json
import os
import pickle
import time

import numpy as np
import tensorflow as tf

from .mc_functions import load_data_mc, setup_model_spec_mc
from .nasdaq_functions import load_data_nasdaq, setup_model_spec_nasdaq
from .rnn import LobRNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_length', type=int, default=20)
    parser.add_argument('--seq_length_time', type=int, default=60)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_units', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument('--type', type=str, default="mc", choices=["mc", "nasdaq"])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default="-1", choices=["0", "1", "-1"])
    parser.add_argument('--checkpoint_path', type=str, required=True)

    return parser.parse_args()


def generate_data_time(lob_rnn, predict_model, data_dict, num_gen_samples=-1, end_time=120, start_len=50, offset=1000):
    start_idx = np.zeros(shape=(0, 3), dtype=int)
    for s in range(len(data_dict["event"])):
        next_index = 0
        done = False
        while not done:
            first_index = next_index
            next_index += np.searchsorted(np.cumsum(
                data_dict["time" if "abs_time" not in data_dict else "abs_time"][s][start_len + next_index + 1:]),
                end_time) + start_len + 1
            if next_index >= data_dict["event"][s].shape[0]:
                done = True
            else:
                start_idx = np.append(start_idx, np.array([[s, first_index, next_index]]), axis=0)
                next_index += offset

    if num_gen_samples == -1:
        num_gen_samples = start_idx.shape[0]

    if num_gen_samples > 0:
        gen_data_dict = {"event_types": lob_rnn.event_types}
        print(num_gen_samples)
        print(start_idx.shape[0])
        num_gen_samples = np.min([num_gen_samples, start_idx.shape[0]])
        idx = np.random.choice(start_idx.shape[0], size=num_gen_samples, replace=False)

        for k, v in data_dict.items():
            gen_data_dict[k] = [v[start_idx[i, 0]][start_idx[i, 1]:start_idx[i, 2], ...] for i in idx]
            gen_data_dict["generated_{}".format(k)] = []

        for ind in range(num_gen_samples):
            print(str(ind + 1) + '/' + str(num_gen_samples))
            t0 = time.time()

            seed_dict = {"ob": gen_data_dict["ob"][ind][:1 + start_len, ...]}
            for s in ["size", "abs_size", "part_size", "level", "event", "index", "time", "abs_level", "abs_time"]:
                if s in lob_rnn.training_data:
                    seed_dict[s] = gen_data_dict[s][ind][:1 + start_len].reshape(1 + start_len)

            gen_dict, finished = lob_rnn.generate_sequence(seed_dict, predict_model, end_time, max_time=True)
            for i in data_dict:
                gen_data_dict["generated_{}".format(i)].append(gen_dict[i])
                gen_data_dict[i][ind] = gen_data_dict[i][ind][start_len:]

            t1 = time.time()
            print(t1 - t0)
            print('total generated time')
            print(np.sum(gen_dict["time"][1:]))
            print(np.sum(gen_data_dict["time"][ind][1:]))
            if "abs_time" in gen_dict:
                print('total generated absolute time')
                print(np.sum(gen_dict["abs_time"][1:]))
                print(np.sum(gen_data_dict["abs_time"][ind][1:]))
                print('first events')
                print(gen_dict["event"][:10])
                print(gen_data_dict["event"][ind][:10])
    else:
        gen_data_dict = {}
    return gen_data_dict


def main(args):
    # set seed
    tf.compat.v1.disable_eager_execution()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # set GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # add timestamp and paths to new files
    args.timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    args.data_path = os.path.abspath(
        __file__ + "/../../") + "/data/rnn_generated_data/" + args.type + "_lob_" + args.timestamp

    # load training data
    data, event_types, args = load_data_mc(args) if args.type == "mc" else load_data_nasdaq(args)

    # save info files in data directory
    with open(args.data_path + '_info.txt', 'w') as file:
        json.dump(args.__dict__, file, indent=2)

    # set up model
    model_spec = setup_model_spec_mc(args) if args.type == "mc" else setup_model_spec_nasdaq(args, event_types)
    if args.type == "mc":
        lob_rnn = LobRNN(data, model_spec, event_types, integer_size=True, outside_volume=1)
    else:
        lob_rnn = LobRNN(data, model_spec, event_types, integer_size=False, outside_volume=0. / args.std_size,
                         max_constant=1.)

    model = lob_rnn.setup_model(args, predict=True, device='/cpu:0')
    for k, v in model.items():
        print(k, " weights: ", v.count_params())
        v.load_weights(args.checkpoint_path.format(k))

    # generate data
    if (args.num_samples > 0) or (args.num_samples == -1):
        data["ob"] = [d[1:, ...] for d in data["ob"]]
        gen_data_dict = generate_data_time(lob_rnn, model, data, args.num_samples, args.seq_length_time,
                                           args.seq_length)

        # save generated data
        pickle.dump(gen_data_dict, open(args.data_path + ".pkl", "wb"))


if __name__ == '__main__':
    main(parse_args())
