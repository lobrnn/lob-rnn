import argparse
import copy
import datetime
import json
import os
import pickle
import time

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.utils as utils
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Dense, Input, Reshape, Conv2D, MaxPooling2D, LSTM
from tensorflow.keras.models import Model

from .lob import LOB
from .mc_functions import load_data_mc, setup_model_spec_mc
from .nasdaq_functions import load_data_nasdaq, setup_model_spec_nasdaq
from .rnn import LobRNN


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def data_classification(X, Y, T):
    # code from https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX.reshape(dataX.shape + (1,)), dataY


def create_deeplob(T, NF, number_of_lstm):
    # code from https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/
    input_lmd = Input(shape=(T, NF, 1))

    # build the convolutional block
    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(input_lmd)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 10))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    # build the inception module
    convsecond_1 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = Conv2D(64, (3, 1), padding='same')(convsecond_1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = Conv2D(64, (5, 1), padding='same')(convsecond_2)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
    convsecond_3 = Conv2D(64, (1, 1), padding='same')(convsecond_3)
    convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)

    convsecond_output = keras.layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)

    # use the MC dropout here
    conv_reshape = Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)

    # build the last LSTM layer
    conv_lstm = LSTM(number_of_lstm)(conv_reshape)

    # build the output layer
    out = Dense(3, activation='softmax')(conv_lstm)
    model = Model(inputs=input_lmd, outputs=out)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def data_format(X, T=100):
    # code from https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/
    N = X.shape[0]
    # [N, D] = X.shape
    df = np.array(X)

    dataX = np.zeros((N - T + 1, T) + X.shape[1:])
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, ...]

    return dataX


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--num_trajectories', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default="-1", choices=["0", "1", "-1"])
    parser.add_argument('--checkpoint_timestamp', type=str, required=True)
    parser.add_argument('--type', type=str, default='mc')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    tf.compat.v1.disable_eager_execution()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
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

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    p = os.path.abspath(__file__ + "/../../")
    rnn_path = os.path.join(p, 'models/checkpoints/{}_lob_rnn_{}'.format(args.type, args.checkpoint_timestamp))
    args.rnn_info_path = os.path.join(rnn_path, 'model_info.txt')
    args.rnn_checkpoint_path = os.path.join(rnn_path, 'lob_{}_lstm_predict_model.hdf5')

    with open(args.rnn_info_path) as file:
        rnn_info = json.load(file)

    rnn_args = Namespace(**rnn_info)
    data, event_types, rnn_args = load_data_mc(rnn_args) if args.type == "mc" else load_data_nasdaq(rnn_args)
    test_data, _, _ = load_data_mc(copy.deepcopy(rnn_args), "mc_test") if args.type == "mc" else load_data_nasdaq(
        copy.deepcopy(rnn_args), "nasdaq_test")
    print(data["abs_level"][0])

    get_features = lambda lob: np.stack(
        [np.arange(10) + lob.ask, lob.q_ask()[:10], -np.arange(10) + lob.bid,
         lob.q_bid()[:10]]).T.flatten()
    ob_to_features = lambda ob: np.apply_along_axis(lambda d: get_features(LOB(d)), -1, ob.reshape((ob.shape[0], -1)))
    ob_to_mid = lambda ob: np.apply_along_axis(lambda d: LOB(d).mid, -1, ob.reshape((ob.shape[0], -1)))
    apply_type = (lambda f, x: f(x)) if args.type == "mc" else (lambda f, x: [f(d) for d in x])

    features = apply_type(ob_to_features, (data["ob"][0] if args.type == "mc" else data["ob"]))
    mids = apply_type(ob_to_mid, data["ob"][0] if args.type == "mc" else data["ob"])

    # take out train/val/test splits of index
    val_part = .1
    train_part = 1 - val_part
    all_idx = np.arange(features.shape[0]) if args.type == "mc" else np.arange(len(features))
    train_idx = all_idx[:int(all_idx.shape[0] * train_part)]
    val_idx = all_idx[int(all_idx.shape[0] * train_part):]

    # normalize features by training data
    if args.type == "mc":
        all_train_features = features[train_idx, ...]
    else:
        all_train_features = np.concatenate([features[i] for i in train_idx])
    train_features_mean = all_train_features.mean(axis=0)
    train_features_std = all_train_features.std(axis=0)
    features = apply_type(lambda f: (f - train_features_mean) / train_features_std, features)
    features = apply_type(lambda f: f[:-args.k, :], features)


    def assign_labels(x):
        ls = 100 * np.ones(x.shape[0] + args.k - 1)
        ls[(args.k - 1):] = np.where(x > args.alpha, 1, 0)
        ls[(args.k - 1):] = np.where(x < -args.alpha, -1, ls[(args.k - 1):])
        ls[(args.k - 1):] = ls[(args.k - 1):] + 1
        return ls


    m_func = lambda m: np.array([np.sum(m[i:i + args.k]) / args.k for i in range(m.shape[-1] - args.k + 1)])
    m_values = apply_type(lambda m: np.apply_along_axis(m_func, -1, m), mids)
    lt = apply_type(lambda m: m[args.k:] - m[:-args.k], m_values)
    labels = apply_type(assign_labels, lt)
    _, vv = np.unique(labels if args.type == "mc" else np.concatenate(labels), return_counts=True)
    print('class proportions: ', list(vv / np.sum(vv)))

    test_features = ob_to_features(test_data["ob"][0])
    test_mids = ob_to_mid(test_data["ob"][0])
    test_features = (test_features - train_features_mean) / train_features_std
    test_features = test_features[:-args.k, :]
    test_m_values = np.apply_along_axis(m_func, -1, test_mids)
    test_m_minus = test_m_values[:-args.k]
    test_m_plus = test_m_values[args.k:]
    test_lt = (test_m_plus - test_m_minus)  # /test_m_minus
    test_labels = 100 * np.ones(test_features.shape[:-1])
    test_labels[(args.k - 1):] = np.where(test_lt > args.alpha, 1, 0)
    test_labels[(args.k - 1):] = np.where(test_lt < -args.alpha, -1, test_labels[(args.k - 1):])
    test_labels[(args.k - 1):] = test_labels[(args.k - 1):] + 1
    _, vv = np.unique(test_labels, return_counts=True)
    print('class proportions test: ', list(vv / np.sum(vv)))

    cnn_data = data_classification(features, labels, T=args.T) if args.type == "mc" else [
        data_classification(features[i], labels[i], T=100) for i in range(len(features))]
    train_x = cnn_data[0] if args.type == "mc" else np.concatenate([cnn_data[i][0] for i in train_idx])
    train_y = utils.to_categorical(
        cnn_data[1] if args.type == "mc" else np.concatenate([cnn_data[i][1] for i in train_idx]), 3)

    test_cnn_data = data_classification(test_features, test_labels, T=args.T)
    test_x = test_cnn_data[0]
    test_y = utils.to_categorical(test_cnn_data[1], 3)

    deeplob = create_deeplob(args.T, train_x.shape[2], 64)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_history = deeplob.fit(train_x, train_y, epochs=100, batch_size=64, verbose=1, validation_split=.1,
                                callbacks=[callback])

    print(classification_report(test_y.argmax(axis=-1), deeplob.predict(test_x).argmax(axis=-1),
                                target_names=["down", "stationary", "up"], digits=3))

    # set up model
    model_spec = setup_model_spec_mc(rnn_args) if args.type == "mc" else setup_model_spec_nasdaq(rnn_args, event_types)
    integer_size = True if args.type == "mc" else False
    outside_volume = 1 if args.type == "mc" else 0
    lob_rnn = LobRNN(data, model_spec, event_types, integer_size=integer_size, outside_volume=outside_volume,
                     max_constant=1.2 if args.type == "mc" else 1.0)

    model = lob_rnn.setup_model(rnn_args, predict=True, device='/cpu:0')

    for key, v in model.items():
        v.load_weights(args.rnn_checkpoint_path.format(key))

    test_data_input = {}
    for key, v in test_data.items():
        test_data_input[key] = v[0]
        if len(test_data_input[key].shape) == 2:
            test_data_input[key] = test_data_input[key].reshape(test_data_input[key].shape + (1,))
        if key == "ob":
            test_data_input[key] = data_format(test_data_input[key][1:-args.k, ...])
        else:
            test_data_input[key] = data_format(test_data_input[key][:-args.k, ...])

    max_sequence_len = args.T + args.k
    lt_rnn = np.zeros((args.num_samples, args.num_trajectories))
    l_true = np.zeros(args.num_samples)
    predict_time = np.zeros((args.num_samples, args.num_trajectories + 1))
    rand_test_idx = np.random.choice(test_data_input["event"].shape[0], size=args.num_samples, replace=False)

    for i in range(args.num_samples):
        ind = rand_test_idx[i]
        t0 = time.time()
        l_vals = np.zeros(args.num_trajectories)
        if not i % 1e2:
            print(str(i + 1) + '/' + str(args.num_samples))
            print('index: ', ind)

        seed_dict = {"ob": test_data_input["ob"][ind, :args.T, ...]}
        for s in ["size", "abs_size", "part_size", "level", "event", "index", "time", "abs_time", "abs_level"]:
            if s in lob_rnn.training_data:
                seed_dict[s] = test_data_input[s][ind, :args.T, ...]

        mid_minus = np.mean(ob_to_mid(seed_dict["ob"][-args.k:, ...]))
        finished = False
        while not finished:
            gen_dict, finished = lob_rnn.generate_sequence(seed_dict, model, args.T, reset=True, reset_weights=True)
        for s in ["size", "abs_size", "part_size", "level", "event", "index", "time", "abs_time", "abs_level", "ob"]:
            if s in lob_rnn.training_data:
                seed_dict[s] = test_data_input[s][ind, args.T - 1, ...]
        t1 = time.time()
        predict_time[i, -1] = t1 - t0
        for n in range(args.num_trajectories):
            t0 = time.time()
            finished = False
            while not finished:
                gen_dict, finished = lob_rnn.generate_sequence(seed_dict, model, args.k + 1, reset=False)
            mid_plus = np.mean(ob_to_mid(gen_dict["ob"][1:, ...]))
            lt_rnn[i, n] = (mid_plus - mid_minus)  # /mid_minus
            t1 = time.time()
            predict_time[i, n] = t1 - t0

    rnn_labels = np.where(np.mean(lt_rnn, axis=-1) > args.alpha, 1, 0)
    rnn_labels = np.where(np.mean(lt_rnn, axis=-1) < -args.alpha, -1, rnn_labels)
    rnn_labels = rnn_labels + 1

    t0 = time.time()
    deeplob.predict(test_x[rand_test_idx, ...]).argmax(axis=-1)
    t1 = time.time()
    predict_time_deeplob = t1 - t0

    print('Average time (ms) for DeepLOB prediction: ',
          1e3 * predict_time_deeplob / test_x[rand_test_idx, ...].shape[0])
    print('Average time (ms) for Simulation RNNLOB prediction: ', 1e3 * np.mean(np.mean(predict_time, axis=-1)))

    print('RNN RESULTS')
    print(classification_report(test_y[rand_test_idx, :].argmax(axis=-1), rnn_labels,
                                target_names=["down", "stationary", "up"], digits=3))

    print('DEEPLOB RESULTS')
    print(classification_report(test_y[rand_test_idx, :].argmax(axis=-1),
                                deeplob.predict(test_x[rand_test_idx, ...]).argmax(axis=-1),
                                target_names=["down", "stationary", "up"], digits=3))

    save_dict = {}
    # prediction horizon
    save_dict["seed"] = args.seed
    save_dict["k"] = args.k
    # label threshold
    save_dict["alpha"] = args.alpha
    save_dict["T"] = args.T
    save_dict["lt"] = lt
    save_dict["test_labels"] = test_y[rand_test_idx, :]
    save_dict["deeplob_test_prediction"] = deeplob.predict(test_x[rand_test_idx, ...])
    save_dict["simulation_prediction"] = lt_rnn
    save_dict["deeplob_test_time"] = predict_time_deeplob
    save_dict["simulation_test_time"] = predict_time
    pickle.dump(save_dict,
                open(os.path.join(p, "data/midprice_prediction_{}_{}.pkl".format(args.type, timestamp)), "wb"))
