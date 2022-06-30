import argparse
import os
import pickle

import numpy as np

from .lob import LOB
from .mc_functions import load_data_mc
from .nasdaq_functions import load_data_nasdaq, setup_model_spec_nasdaq


def sample_type(lob, events_freq, events_v, event_types, levels):
    freq = events_freq.copy()
    redo_norm = False
    if lob.ask_volume == 0:
        freq[event_types["mo ask"]] = 0
        freq[event_types["cancellation sell"]] = 0
        redo_norm = True
    if lob.bid_volume == 0:
        freq[event_types["mo bid"]] = 0
        freq[event_types["cancellation buy"]] = 0
        redo_norm = True
    if not np.any(lob.data[0, 1 + levels["cancellation sell"]["v"]]):
        freq[event_types["cancellation sell"]] = 0
        redo_norm = True
    if not np.any(lob.data[1, 1 + levels["cancellation buy"]["v"]]):
        freq[event_types["cancellation buy"]] = 0
        redo_norm = True
    if redo_norm:
        freq = freq / np.sum(freq)
    return np.random.choice(events_v, p=freq)


def sample_level(lob, event_type, levels, inverse_event_types):
    p = levels[inverse_event_types[event_type]]["freq"].copy()
    v = levels[inverse_event_types[event_type]]["v"]
    redo_norm = False
    if inverse_event_types[event_type] == "cancellation sell":
        zero_index = lob.data[0, 1 + levels["cancellation sell"]["v"]] == 0
        p[zero_index] = 0
        redo_norm = True
    elif inverse_event_types[event_type] == "cancellation buy":
        zero_index = lob.data[1, 1 + levels["cancellation buy"]["v"]] == 0
        p[zero_index] = 0
        redo_norm = True
    if redo_norm:
        p = p / np.sum(p)
    cat_level = np.random.choice(v, p=p)
    if inverse_event_types[event_type] in ['lo buy', 'cancellation buy']:
        abs_level = lob.ask - cat_level - 1
    elif inverse_event_types[event_type] == 'mo ask':
        abs_level = lob.ask
    elif inverse_event_types[event_type] == 'mo bid':
        abs_level = lob.bid
    else:
        abs_level = lob.bid + cat_level + 1
    return cat_level, abs_level


def sample_size(lob, event_type, level, sizes, inverse_event_types, data_type="mc", transform_size=None):
    p = sizes[inverse_event_types[event_type]]["freq"].copy()
    v = sizes[inverse_event_types[event_type]]["v"]
    if data_type == "mc":
        redo_norm = False
        if inverse_event_types[event_type] == "mo ask":
            zero_index = (v > lob.ask_volume)
            p[zero_index] = 0
            redo_norm = True
        elif inverse_event_types[event_type] == "mo bid":
            zero_index = (v > lob.bid_volume)
            p[zero_index] = 0
            redo_norm = True
        elif inverse_event_types[event_type] == "cancellation sell":
            zero_index = (v > lob.data[0, 1 + level])
            p[zero_index] = 0
            redo_norm = True
        elif inverse_event_types[event_type] == "cancellation buy":
            zero_index = (v > np.abs(lob.data[1, 1 + level]))
            p[zero_index] = 0
            redo_norm = True
        if redo_norm:
            p = p / np.sum(p)
        return np.random.choice(v, p=p)
    else:
        cat_size = np.random.choice(v, p=p)
        vol = np.abs(lob.get_volume(level, absolute_level=True))
        abs_size = transform_size(cat_size, event_type, vol)
        return cat_size, abs_size


def sample_time(time_params, data_type="mc", transform_time=None):
    if data_type == "mc":
        return np.random.exponential(time_params)
    else:
        cat_t = np.random.choice(time_params["v"], p=time_params["freq"])
        abs_t = transform_time(cat_t)
        return cat_t, abs_t


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_length_time', type=int, default=60)
    parser.add_argument('--type', type=str, default="mc", choices=["mc", "nasdaq"])
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--start_len', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()


def main(args):
    np.random.seed(args.seed)
    data, event_types, args = load_data_mc(args) if args.type == "mc" else load_data_nasdaq(args)
    if args.type == "nasdaq":
        model_spec = setup_model_spec_nasdaq(args, event_types)
    inverse_event_types = {v: k for k, v in event_types.items()}

    flat_data = {}
    for k, v in data.items():
        flat_data[k] = np.concatenate(v)

    events = {}
    for s in event_types.keys():
        events[s] = (flat_data["event"].squeeze() == event_types[s])
    events_v, events_c = np.unique(flat_data["event"], return_counts=True)
    events_freq = events_c / np.sum(events_c)

    levels = {}
    sizes = {}
    for s in event_types.keys():
        levels[s] = {}
        levels[s]["data"] = flat_data["level"][events[s]]
        levels[s]["v"], levels[s]["c"] = np.unique(levels[s]["data"], return_counts=True)
        levels[s]["freq"] = levels[s]["c"] / np.sum(levels[s]["c"])
        sizes[s] = {}
        sizes[s]["data"] = flat_data["size"][events[s]]
        sizes[s]["v"], sizes[s]["c"] = np.unique(sizes[s]["data"], return_counts=True)
        sizes[s]["freq"] = sizes[s]["c"] / np.sum(sizes[s]["c"])
    if args.type == "mc":
        time_params = np.mean(flat_data["time"])
    else:
        time_params = {}
        time_params["v"], time_params["c"] = np.unique(flat_data["time"], return_counts=True)
        time_params["freq"] = time_params["c"] / np.sum(time_params["c"])

    start_idx = np.zeros(shape=(0, 3), dtype=int)
    for s in range(len(data["event"])):
        next_index = 0
        done = False
        while not done:
            first_index = next_index
            next_index += np.searchsorted(np.cumsum(
                data["time" if args.type == "mc" else "abs_time"][s][args.start_len + next_index + 1:]),
                args.seq_length_time) + args.start_len + 1
            if next_index >= data["event"][s].shape[0]:
                done = True
            else:
                start_idx = np.append(start_idx, np.array([[s, first_index, next_index]]), axis=0)

    print(args.num_samples)
    print(start_idx.shape[0])
    num_samples = np.min([args.num_samples, start_idx.shape[0]])
    idx = np.random.choice(start_idx.shape[0], size=num_samples, replace=False)
    gen_data_dict = {}
    for k, v in data.items():
        gen_data_dict[k] = [v[start_idx[i, 0]][start_idx[i, 1]:start_idx[i, 2], ...] for i in idx]
        gen_data_dict["generated_{}".format(k)] = []

    for j in range(num_samples):
        print('iteration: ', j)
        if args.type == "mc":
            times = [gen_data_dict["time"][j][0]]
            total_time = 0
            while total_time < args.seq_length_time:
                t = sample_time(time_params)
                times.append(t)
                total_time += times[-1]
            gen_data_dict["generated_time"].append(np.array(times[:-1]))
        else:
            times = [gen_data_dict["abs_time"][j][0]]
            cat_times = [gen_data_dict["time"][j][0]]
            total_time = 0
            while total_time < args.seq_length_time:
                cat_t, abs_t = sample_time(time_params, data_type=args.type,
                                           transform_time=model_spec["transform_time"])
                times.append(abs_t)
                cat_times.append(cat_t)
                total_time += times[-1]
            gen_data_dict["generated_abs_time"].append(np.array(times[:-1]))
            gen_data_dict["generated_time"].append(np.array(cat_times[:-1]))
        num_events = len(times[:-1])
        for k, v in data.items():
            if k == "ob":
                gen_data_dict["generated_{}".format(k)].append(np.zeros((num_events,) + v[0].shape[1:]))
                lob = LOB(gen_data_dict["ob"][j][0, ...].copy())
                gen_data_dict["generated_{}".format(k)][j][0, ...] = lob.data.copy()
            elif k not in ["time", "abs_time"]:
                gen_data_dict["generated_{}".format(k)].append(np.zeros(num_events, dtype=v[0].dtype))
                gen_data_dict["generated_{}".format(k)][j][0] = gen_data_dict[k][j][0]

        for i in range(num_events):
            gen_data_dict["generated_event"][j][i] = sample_type(lob, events_freq, events_v, event_types, levels)
            gen_data_dict["generated_level"][j][i], gen_data_dict["generated_abs_level"][j][i] = \
                sample_level(lob, gen_data_dict["generated_event"][j][i], levels, inverse_event_types)
            if args.type == "mc":
                gen_data_dict["generated_size"][j][i] = sample_size(lob, gen_data_dict["generated_event"][j][i],
                                                                    gen_data_dict["generated_level"][j][i], sizes,
                                                                    inverse_event_types)
            else:
                gen_data_dict["generated_size"][j][i], gen_data_dict["generated_abs_size"][j][i] = \
                    sample_size(lob, gen_data_dict["generated_event"][j][i], gen_data_dict["generated_abs_level"][j][i],
                                sizes, inverse_event_types, args.type, model_spec["transform_size"])
                if gen_data_dict["generated_abs_size"][j][i] == 0:
                    print('size 0')
                    break
            size_key = "generated_size" if args.type == "mc" else "generated_abs_size"
            if inverse_event_types[gen_data_dict["generated_event"][j][i]] in ["mo bid", "lo sell", "cancellation buy"]:
                change_ok = lob.change_volume(gen_data_dict["generated_abs_level"][j][i], gen_data_dict[size_key][j][i],
                                              absolute_level=True)
            else:
                change_ok = lob.change_volume(gen_data_dict["generated_abs_level"][j][i],
                                              -gen_data_dict[size_key][j][i], absolute_level=True)
            if not change_ok:
                print("BAD EVENT")
                print("event type: ", inverse_event_types[gen_data_dict["generated_event"][j][i]])
                print(lob.data)
                print("level: ", gen_data_dict["generated_level"][j][i])
                print("abs level: ", gen_data_dict["generated_abs_level"][j][i])
                print("size: ", gen_data_dict[size_key][j][i])
                print(np.abs(lob.get_volume(gen_data_dict["generated_level"][j][i], absolute_level=False)))
                break
            elif i + 1 < num_events:
                gen_data_dict["generated_ob"][j][i + 1, ...] = lob.data.copy()

    gen_data_dict["event_types"] = event_types
    data_path = os.path.abspath(
        __file__ + "/../../") + "/data/" + args.type + "_naive_model_data_" + str(args.seed) + ".pkl"
    pickle.dump(gen_data_dict, open(data_path, "wb"))


if __name__ == '__main__':
    main(parse_args())
