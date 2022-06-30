import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def generate_size_mc(size_params, event):
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
        size = np.random.choice(size_probs.size, p=size_probs / np.sum(size_probs)) + 1
        return size, True
    else:
        return 0, False


def generate_time_mc(time_params):
    """
    Generate a time sample for RNN trained on Markov Chain model data

    Parameters
    ----------
    time_params : numpy array of one parameter for exponential distributions

    Returns
    -------
    float sample of exponential distribution

    """
    return np.random.exponential(scale=1 / (1e-5 + np.square(1 + time_params[0])))


def mask_size_mc(x, inputs, event_types):
    arange = tf.reshape(tf.tile(tf.range(x.shape[-1]), tf.math.reduce_prod(x.shape[:-1], keepdims=True)), x.shape)
    # if MO bid: max size given by best bid volume
    mo_bid_mask = tf.reshape(tf.repeat((tf.equal(inputs['event_new'], event_types["mo bid"])), x.shape[-1]), x.shape)
    bid_vol = tf.cast(tf.reshape(tf.repeat(tf.abs(inputs["bid_last"]), x.shape[-1]), x.shape), tf.int32)
    x = tf.where(tf.logical_or(tf.equal(mo_bid_mask, False), tf.greater(bid_vol, arange)), x, -np.inf)
    # if MO ask: max size given by best ask volume
    mo_ask_mask = tf.reshape(tf.repeat((tf.equal(inputs['event_new'], event_types["mo ask"])), x.shape[-1]), x.shape)
    ask_vol = tf.cast(tf.reshape(tf.repeat(inputs["ask_last"], x.shape[-1]), x.shape), tf.int32)
    x = tf.where(tf.logical_or(tf.equal(mo_ask_mask, False), tf.greater(ask_vol, arange)), x, -np.inf)

    # if cancellation buy: max size given by volume at event level distance from best ask
    canc_buy_mask = tf.reshape(tf.repeat((tf.equal(inputs['event_new'], event_types["cancellation buy"])), x.shape[-1]),
                               x.shape)
    buy_gather_index = tf.where(tf.equal(inputs['event_new'], event_types["cancellation buy"]),
                                tf.cast(inputs["level_new"], tf.int32),
                                tf.zeros(inputs["level_new"].shape, dtype=tf.int32))
    canc_buy_sizes = tf.cast(tf.reshape(
        tf.repeat(tf.abs(tf.gather_nd(-inputs["ob_last"][..., 1, 1:], buy_gather_index, batch_dims=len(x.shape) - 1)),
                  x.shape[-1]),
        x.shape), tf.int32)
    x = tf.where(tf.logical_or(tf.equal(canc_buy_mask, False), tf.greater(canc_buy_sizes, arange)), x, -np.inf)

    # if cancellation sell: max size given by volume at event level distance from best bid
    canc_sell_mask = tf.reshape(
        tf.repeat((tf.equal(inputs['event_new'], event_types["cancellation sell"])), x.shape[-1]), x.shape)
    sell_gather_index = tf.where(tf.equal(inputs['event_new'], event_types["cancellation sell"]),
                                 tf.cast(inputs["level_new"], tf.int32),
                                 tf.zeros(inputs["level_new"].shape, dtype=tf.int32))
    canc_sell_sizes = tf.cast(tf.reshape(
        tf.repeat(tf.gather_nd(inputs["ob_last"][..., 0, 1:], sell_gather_index, batch_dims=len(x.shape) - 1),
                  x.shape[-1]),
        x.shape), tf.int32)
    x = tf.where(tf.logical_or(tf.equal(canc_sell_mask, False), tf.greater(canc_sell_sizes, arange)), x, -np.inf)
    return x


def setup_model_spec_mc(args):
    """
    Setups which functions to use for modelling Markov chain model data

    Returns
    -------
    model_spec
        dict of all functions needed for RNN event model

    """
    model_spec = {"generate_size": generate_size_mc, "generate_time": generate_time_mc}
    model_spec["categorical"] = {"event": True, "level": True, "size": True, "time": False}
    model_spec["size_log_likelihood"] = lambda s, inputs: tfp.distributions.Categorical(s).log_prob(
        tf.squeeze(inputs["size_new"] - 1, -1))
    model_spec["time_log_likelihood"] = lambda t, inputs: tfp.distributions.Exponential(
        rate=tf.square(1 + tf.squeeze(t, -1)) + 1e-5).log_prob(tf.squeeze(inputs["time_new"], -1))
    model_spec["time_num_outputs"] = 1
    model_spec["mask_size"] = mask_size_mc
    model_spec["transform_size"] = lambda b, e, vol: b
    model_spec["transform_size_bins"] = lambda s, e, vol: s
    return model_spec


def load_data_mc(args, filename="mc_train", directory=None):
    """
    Loads Markov chain model data into dictionary to use as training data

    Parameters
    ----------
    args : arguments not used in this function but needed to fit into pattern
    filename : string of the name of the data file
    directory : string of directory of the data file

    Returns
    -------
    data : dict of Markov chain model data
    event_types : dict of mapping between event types and integers
    args : dict being the same as the input dict

    """
    if directory is None:
        directory = os.path.abspath(__file__ + "/../../") + "\\data"

    data_dict = pickle.load(open(os.path.join(directory, "{}.pkl".format(filename)), "rb"))
    event_types = data_dict.pop("event_types")
    data_keys = [k for k in data_dict.keys() if k not in ["total_time", "num_events", "index"]]
    data = {}
    for k in data_keys:
        data[k] = [data_dict[k]]

    start_ask = int(data["ob"][0][0, 0, 0])
    data["ob"][0][:, 0, 0] -= start_ask
    data["abs_level"][0] -= start_ask

    return data, event_types, args
