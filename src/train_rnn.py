import argparse
import datetime
import json
import os

import numpy as np
import tensorflow as tf

from .mc_functions import load_data_mc, setup_model_spec_mc
from .nasdaq_functions import load_data_nasdaq, setup_model_spec_nasdaq
from .rnn import LobRNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_length', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_units', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--type', type=str, default="mc", choices=["mc", "nasdaq"])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default="-1", choices=["0", "1", "-1"])

    return parser.parse_args()


def main(args):
    # set seed
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
    args.model_path = os.path.abspath(__file__ + "/../../") + '/models/checkpoints/{}_lob_rnn_{}/'.format(
        args.type,
        args.timestamp)

    # load training data
    data, event_types, args = load_data_mc(args) if args.type == "mc" else load_data_nasdaq(args)

    # save info files in data and model directories
    try:
        os.makedirs(args.model_path)
    except FileExistsError:
        print("Directory already exists")
    with open(args.model_path + 'model_info.txt', 'w') as file:
        json.dump(args.__dict__, file, indent=2)

    # set up model
    model_spec = setup_model_spec_mc(args) if args.type == "mc" else setup_model_spec_nasdaq(args, event_types)
    if args.type == "mc":
        lob_rnn = LobRNN(data, model_spec, event_types, integer_size=True, outside_volume=1)
    else:
        lob_rnn = LobRNN(data, model_spec, event_types, integer_size=False, outside_volume=0. / args.std_size,
                         max_constant=1.)

    model = lob_rnn.setup_model(args)
    for k, v in model.items():
        print(k, " weights: ", v.count_params())

    # train model
    checkpoint_path = args.model_path + 'lob_lstm_model_epoch_{epoch:02d}.hdf5'
    train_history = lob_rnn.train_model(model["nll"], checkpoint_path, args.batch_size, epochs=args.epochs)
    loss_keys = list(model["nll"].loss.keys())
    training_info = {}

    for i in range(len(loss_keys)):
        tmp_key = \
            [k for k in train_history.history.keys() if k.startswith(model["nll"].outputs[i].name.split("/")[0])][0]
        train_history.history["{}_loss".format(loss_keys[i])] = train_history.history[tmp_key]
        train_history.history["val_{}_loss".format(loss_keys[i])] = train_history.history["val_{}".format(tmp_key)]
        training_info[loss_keys[i]] = {}
        training_info[loss_keys[i]]["train_loss"] = [np.mean(l).astype(float) for l in
                                                     train_history.history["{}_loss".format(loss_keys[i])]]
        training_info[loss_keys[i]]["val_loss"] = [np.mean(l).astype(float) for l in
                                                   train_history.history["val_{}_loss".format(loss_keys[i])]]
        best_epoch = np.nanargmin(training_info[loss_keys[i]]["val_loss"]) + 1
        print('Best epoch for {}: '.format(loss_keys[i]), best_epoch)
        model["nll"].load_weights(checkpoint_path.format(epoch=best_epoch))
        model[loss_keys[i]].save(args.model_path + 'lob_{}_lstm_predict_model.hdf5'.format(loss_keys[i]))

    with open(args.model_path + 'training_info.txt', 'w') as file:
        file.write(json.dumps(training_info))


if __name__ == '__main__':
    main(parse_args())
