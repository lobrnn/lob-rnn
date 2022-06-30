import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .lob import LOB


def generate_size_nasdaq(size_params, event):
    """
    Generate a size sample for RNN trained on Markov Chain model data

    Parameters
    ----------
    size_params : numpy array of logits
    event : integer not actually needed in this computation but needed to fit into pattern

    Returns
    -------
    size
        integer of sampled size
    boolean
        indicating success (True) or weird logits (False)

    """
    size_probs = np.exp(size_params - np.where(np.max(size_params) > -np.inf, np.max(size_params), 0))
    if np.sum(size_probs) != 0:
        size = np.random.choice(size_probs.size, p=size_probs / np.sum(size_probs))
        return size, True
    else:
        return 0, False


def get_bins(data, num_bins=100, max_width=10):
    vals = []
    cont_data = data.copy()
    loop = True
    while loop:
        v, c = np.unique(cont_data, return_counts=True)
        new_vals = v[c / np.sum(c) >= 1 / (num_bins - len(vals))]
        vals = np.concatenate([vals, new_vals])
        cont_data = cont_data[np.isin(cont_data, vals) == False]
        loop = (len(new_vals) > 0)

    cont_bins = np.quantile(cont_data, np.arange(num_bins - len(vals) + 1) / (num_bins - len(vals)))
    if max_width is not None:
        idx = np.argwhere(np.diff(cont_bins) > max_width).flatten()
        while idx.size > 0:
            new_vals = (cont_bins[idx] + cont_bins[idx + 1]) / 2
            cont_bins = np.insert(cont_bins, idx + 1, new_vals)
            num_bins += new_vals.size
            idx = np.argwhere(np.diff(cont_bins) > max_width).flatten()
    bins = [list(vals), list(cont_bins)]
    return bins, num_bins


def size2bin(s, bins):
    if s in bins[0]:
        b = np.argwhere(np.isin(bins[0], s)).flat[0]
    else:
        if s == bins[1][0]:
            b = len(bins[0])
        else:
            b = len(bins[0]) + np.digitize(s, bins[1], right=True) - 1
    return b


def bin2size(b, bins):
    if b < len(bins[0]):
        value = bins[0][b]
    else:
        i = b - len(bins[0])
        value = (bins[1][i + 1] - bins[1][i]) * np.random.random() + bins[1][i]
    return value


def generate_time_nasdaq(time_params):
    """
    Generate a time sample for RNN trained on Markov Chain model data

    Parameters
    ----------
    time_params : numpy array of logits

    Returns
    -------
    time
        integer of sampled size
    boolean
        indicating success (True) or weird logits (False)

    """
    time_probs = np.exp(time_params - np.where(np.max(time_params) > -np.inf, np.max(time_params), 0))
    return np.random.choice(time_probs.size, p=time_probs / np.sum(time_probs))


def setup_model_spec_nasdaq(args, event_types):
    """
    Setups which functions to use for modelling Nasdaq data

    Parameters
    ----------
    args : dict containing info about the model, such as number of bins and the boundaries of all bins
    event_types : dict of mapping between event types and integers

    Returns
    -------
    model_spec
        dict of all functions needed for RNN event model

    """
    inverse_event_types = {v: k for k, v in event_types.items()}
    model_spec = {"size_num_outputs": args.num_bins_size, "time_num_outputs": args.num_bins_time,
                  "categorical": {"event": True, "level": True, "size": True, "time": True},
                  "generate_size": generate_size_nasdaq,
                  "size_log_likelihood": lambda s, inputs: tfp.distributions.Categorical(s).log_prob(
                      tf.squeeze(inputs["size_new"], -1)), "mask_size": lambda x, *args: x}

    def transform_size(b, event, vol):
        event_type = inverse_event_types[event].split()[0]
        if event_type == "lo":
            bins = args.size_lo_bins
        elif event_type == "mo":
            bins = args.size_mo_bins
        else:
            bins = args.size_cancellation_bins
        if event_type.startswith("lo"):
            return bin2size(b, bins)
        else:
            return vol * bin2size(b, bins)

    def transform_size_bins(s, event, vol):
        event_type = inverse_event_types[event].split()[0]
        if event_type == "lo":
            bins = args.size_lo_bins
        elif event_type == "mo":
            bins = args.size_mo_bins
        else:
            bins = args.size_cancellation_bins
        if event_type == "lo":
            return size2bin(s, bins)
        else:
            return size2bin(s / vol, bins)

    model_spec["transform_size"] = transform_size
    model_spec["transform_size_bins"] = transform_size_bins

    model_spec["generate_time"] = generate_time_nasdaq
    model_spec["time_log_likelihood"] = lambda t, inputs: tfp.distributions.Categorical(t).log_prob(
        tf.squeeze(inputs["time_new"], -1))
    model_spec["transform_time"] = lambda b: bin2size(b, args.time_bins)
    model_spec["transform_time_bins"] = lambda t: size2bin(t, args.time_bins)

    return model_spec


def load_data_nasdaq(args, filename="nasdaq_train", directory=None, num_bins=100):
    """
    Loads Nasdaq data into dictionary to use as training data

    Parameters
    ----------
    args : arguments to use
    filename : string with name of nasdaq data file
    directory : string with path to nasdaq data folder
    num_bins : int of number of bins to use

    Returns
    -------
    data : dict of Nasdaq data
    event_types : dict of mapping between event types and integers
    args : updated args

    """
    if directory is None:
        directory = os.path.abspath(__file__ + "/../../") + "\\data\\nasdaq_data"

    data = pickle.load(open(os.path.join(directory, "{}.pkl".format(filename)), "rb"))
    event_types = data.pop("event_types")
    data["time"] = [d / 1e9 for d in data["time"]]
    args.num_bins = num_bins
    if not hasattr(args, "std_size"):
        print('setting std')
        args.std_size = np.std(np.concatenate(data["size"], axis=0))
    data["size"] = [d / args.std_size for d in data["size"]]
    for i in range(len(data["ob"])):
        data["ob"][i][:, :, 1:] = data["ob"][i][:, :, 1:] / args.std_size

    data["abs_size"] = [d.copy() for d in data["size"]]
    data["part_size"] = [d.copy() for d in data["size"]]
    not_lo = [np.isin(d, [event_types['lo buy'], event_types['lo sell']]) == False for d in data["event"]]
    ob_tmp = [np.array([LOB(data["ob"][d][dd, ...]).get_volume(data["abs_level"][d][dd], absolute_level=True) for dd in
                        range(data["ob"][d].shape[0] - 1)]) for d in range(len(data["ob"]))]
    for i in range(len(not_lo)):
        data["part_size"][i][not_lo[i]] = data["abs_size"][i][not_lo[i]] / np.abs(ob_tmp[i][not_lo[i]])
    events = np.concatenate(data["event"])
    sizes = np.concatenate(data["part_size"])
    num_bins_size = num_bins
    if not hasattr(args, "time_bins"):
        for order_type in ["lo", "mo", "cancellation"]:
            keys = [v for k, v in event_types.items() if k.startswith(order_type)]
            data_index = np.isin(events, keys)
            if order_type == "lo":
                args.size_lo_bins, num_bins_size = get_bins(sizes[data_index], num_bins_size)
            elif order_type == "mo":
                args.size_mo_bins, num_bins_size = get_bins(sizes[data_index], num_bins_size)
            else:
                args.size_cancellation_bins, num_bins_size = get_bins(sizes[data_index], num_bins_size)
        args.num_bins_size = num_bins_size

        times = np.concatenate(data["time"])
        args.time_bins, args.num_bins_time = get_bins(times, num_bins)
    data["abs_time"] = [d.copy() for d in data["time"]]
    for i in range(len(data["size"])):
        for j in range(data["size"][i].size):
            data["time"][i][j] = size2bin(data["abs_time"][i][j], args.time_bins)
            if data["event"][i][j] in [event_types["lo buy"], event_types["lo sell"]]:
                data["size"][i][j] = size2bin(data["part_size"][i][j], args.size_lo_bins)
            else:
                size_bins = args.size_mo_bins if data["event"][i][j] in [event_types["mo bid"], event_types[
                    "mo ask"]] else args.size_cancellation_bins
                data["size"][i][j] = size2bin(data["part_size"][i][j], size_bins)

    for i in range(len(data["size"])):
        data["size"][i] = data["size"][i].astype(int)
        data["time"][i] = data["time"][i].astype(int)

    del data["date"]
    return data, event_types, args
