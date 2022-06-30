import argparse
import os
import pickle

import numpy as np
import pandas as pd

from .lob import LOB
from .utils import ask, bid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_levels', type=int, default=30)
    parser.add_argument('--month', type=int, default=-1)
    parser.add_argument('--max_num_days', type=int, default=1)
    parser.add_argument('--directory', type=str, default="data//nasdaq_data")
    parser.add_argument('--save_filename', type=str, required=True)

    return parser.parse_args()


def get_event_types(df, event_types):
    """
    Creates array of event types

    Parameters
    ----------
    df : pandas dataframe
        having columns type and direction
    event_types : dict of mapping between event types and integers

    Returns
    -------
    e : numpy array of event types for each event in data frame

    """
    e = np.zeros(df.shape[0], dtype=int) - 1
    e[(df.type == "A") & (df.direction == "B")] = event_types["lo buy"]
    e[(df.type == "A") & (df.direction == "S")] = event_types["lo sell"]
    e[(df.type == "E") & (df.direction == "B")] = event_types["mo bid"]
    e[(df.type == "E") & (df.direction == "S")] = event_types["mo ask"]
    e[(e == -1) & (df.direction == "B")] = event_types["cancellation buy"]
    e[(e == -1) & (df.direction == "S")] = event_types["cancellation sell"]
    return e


def relative_levels(data_dict, event_types, index=-1):
    """
    Creates array of relative levels of events in a sequence

    Parameters
    ----------
    data_dict : dict containing lists of "ob", "abs_level", "event"
    event_types : dict of event types
    index : integer of which index to consider for data_dict

    Returns
    -------
    levels : numpy array of relative levels for each event in specified sequence
    """
    ob = data_dict["ob"][index][:-1, ...]
    levels = np.zeros(data_dict["abs_level"][index].shape, dtype=int)
    buy = (data_dict["event"][index] == event_types["lo buy"]) | (
            data_dict["event"][index] == event_types["cancellation buy"])
    sell = (data_dict["event"][index] == event_types["lo sell"]) | (
            data_dict["event"][index] == event_types["cancellation sell"])
    levels[buy] = np.apply_along_axis(lambda x: LOB(x).ask, -1, ob[buy].reshape((ob[buy].shape[0], -1))) - \
                  data_dict["abs_level"][index][buy] - 1
    levels[sell] = - np.apply_along_axis(lambda x: LOB(x).bid, -1, ob[sell].reshape((ob[sell].shape[0], -1))) + \
                   data_dict["abs_level"][index][sell] - 1

    return levels


def get_lob(data, num_levels):
    """
    Get another representation of the LOB.
    Original representation: full 1D array with negative volumes on bid side and positive on ask side
    New representation: 2D array consisting of ((ask, ask vol 1, ... ask vol n), (bid-ask, bid vol 1, ...., bid vol n))

    Parameters
    ----------
    data : array of full LOB
    num_levels : number of top ask/bid levels to keep

    Returns
    -------
    x : numpy array of shape (2, num_levels +1) with new representation of LOB

    """
    ask_level = ask(data)
    bid_level = bid(data)
    x = np.zeros((2, num_levels + 1))
    x[:, 0] = (ask_level, bid_level - ask_level)
    x[0, 1:] = data[bid_level + 1:bid_level + 1 + num_levels]
    x[1, 1:] = np.flip(data[ask_level - num_levels:ask_level])
    return x


def main(args):
    directory = args.directory
    os_directory = os.fsencode(directory)

    order_book_view = {}
    order_book_view_2 = {}
    ob = {}
    num_levels = args.num_levels
    cleaning_numbers = {}
    dates = []
    good_index = {}

    data_dict = {"event_types": {'lo buy': 0,
                                 'lo sell': 1,
                                 'mo bid': 2,
                                 'mo ask': 3,
                                 'cancellation buy': 4,
                                 'cancellation sell': 5}}

    for k in ["ob", "size", "level", "event", "time", "abs_level", "date"]:
        data_dict[k] = []

    # loop over all csv files in folder
    for file in os.listdir(os_directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv") and filename.startswith(
                "NASDAQ_order_book_view_SE0000148884_2019-{}".format(args.month)) and (
                filename != "NASDAQ_order_book_view_SE0000148884_2019-10-23.csv") and (len(dates) < args.max_num_days):
            print(filename)
            dates.append(filename[-14:-4])
            cleaning_numbers[dates[-1]] = {}
            order_book_view[dates[-1]] = pd.read_csv(directory + "\\" + filename, delimiter=";")
            cleaning_numbers[dates[-1]]["initial"] = order_book_view[dates[-1]].shape[0]
            print('Total number of rows: ', cleaning_numbers[dates[-1]]["initial"])

            order_book_view[dates[-1]]["timestamp_hours"] = order_book_view[dates[-1]]["timestamp"] / 1e9 / 3600
            start_hour = order_book_view[dates[-1]].iloc[0]["timestamp_hours"]
            end_hour = order_book_view[dates[-1]].iloc[-1]["timestamp_hours"]
            end_events = order_book_view[dates[-1]].timestamp_hours > (end_hour - (1 / 6))
            cleaning_numbers[dates[-1]]["end"] = np.sum(end_events)
            order_book_view[dates[-1]] = order_book_view[dates[-1]][end_events == False]

            # find tick size
            prices = order_book_view[dates[-1]]["price"].unique()
            tick_size = np.min(np.diff(np.sort(prices)))

            # remove orders without add information
            references_added = order_book_view[dates[-1]][order_book_view[dates[-1]].type == "A"].orderReference
            cleaning_numbers[dates[-1]]["no add order"] = order_book_view[dates[-1]][
                order_book_view[dates[-1]].orderReference.isin(references_added) == False].shape[0]
            order_book_view[dates[-1]] = order_book_view[dates[-1]][
                order_book_view[dates[-1]].orderReference.isin(references_added)]

            # add quantity to delete events (assuming only A/D/X/E types)
            print(order_book_view[dates[-1]].type.unique())
            quantity_added = order_book_view[dates[-1]][order_book_view[dates[-1]].type == "A"][
                ["orderReference", "quantity"]]
            quantity_cancelled = order_book_view[dates[-1]][order_book_view[dates[-1]].type == "X"].groupby(
                "orderReference").agg({'quantity': 'sum'})
            quantity_executed = order_book_view[dates[-1]][order_book_view[dates[-1]].type == "E"].groupby(
                "orderReference").agg({'quantity': 'sum'})
            quantity_deleted = order_book_view[dates[-1]][order_book_view[dates[-1]].type == "D"].groupby(
                "orderReference").agg({'quantity': 'sum'})

            for index, row in quantity_deleted.iterrows():
                row["quantity"] = quantity_added[quantity_added.orderReference == index].quantity
                for tmp in [quantity_cancelled[quantity_cancelled.index == index].quantity,
                            quantity_executed[quantity_executed.index == index].quantity]:
                    if tmp.size > 0:
                        row["quantity"] = row["quantity"] - tmp

            quantity_deleted["type"] = "D"
            order_book_view[dates[-1]] = order_book_view[dates[-1]].merge(quantity_deleted,
                                                                          on=["orderReference", "type"],
                                                                          how="left",
                                                                          suffixes=("", "_d"))
            order_book_view[dates[-1]].loc[order_book_view[dates[-1]].type == "D", "quantity"] = \
                order_book_view[dates[-1]].loc[
                    order_book_view[dates[-1]].type == "D", "quantity_d"]

            # remove orders which are not fully removed but with later orders executed at the same price
            last_exec_order = order_book_view[dates[-1]][order_book_view[dates[-1]].type == "E"].groupby("price").agg(
                {'orderReference': 'last'})

            quantity_removed = order_book_view[dates[-1]][order_book_view[dates[-1]].type != "A"].groupby(
                "orderReference").agg({'quantity': 'sum'})
            order_book_view_2[dates[-1]] = order_book_view[dates[-1]].merge(quantity_removed, on=["orderReference"],
                                                                            how="left", suffixes=("", "_r"))
            not_all_removed = order_book_view_2[dates[-1]][
                (order_book_view_2[dates[-1]].type == "A") & (
                        order_book_view_2[dates[-1]].quantity != order_book_view_2[dates[-1]].quantity_r)]
            not_all_removed = not_all_removed.merge(last_exec_order, on=["price"], how="left", suffixes=("", "_last"))

            weird_orders = not_all_removed[
                not_all_removed.orderReference_last > not_all_removed.orderReference].orderReference.unique()
            cleaning_numbers[dates[-1]]["later E order"] = order_book_view[dates[-1]][
                order_book_view[dates[-1]].orderReference.isin(weird_orders)].shape[0]
            order_book_view[dates[-1]] = order_book_view[dates[-1]][
                order_book_view[dates[-1]].orderReference.isin(weird_orders) == False]
            tmp = not_all_removed[(not_all_removed.orderReference_last > not_all_removed.orderReference) == False]
            tmp_orders = []
            for index, row in tmp.iterrows():
                num_other_orders = order_book_view[dates[-1]][
                    (order_book_view[dates[-1]].direction != row["direction"]) & (
                            order_book_view[dates[-1]].price == row["price"]) & (
                            order_book_view[dates[-1]].timestamp > row["timestamp"])].orderReference.nunique()
                if num_other_orders > 0:
                    tmp_orders.append(row["orderReference"])

            cleaning_numbers[dates[-1]]["later opposite direction order"] = order_book_view[dates[-1]][
                order_book_view[dates[-1]].orderReference.isin(tmp_orders)].shape[0]
            order_book_view[dates[-1]] = order_book_view[dates[-1]][
                order_book_view[dates[-1]].orderReference.isin(tmp_orders) == False]

            # add cumsum column of quantity left
            order_book_view[dates[-1]]['signed_quantity'] = order_book_view[dates[-1]].quantity
            order_book_view[dates[-1]].loc[order_book_view[dates[-1]].type != "A", 'signed_quantity'] = - \
                order_book_view[dates[-1]].loc[order_book_view[dates[-1]].type != "A", 'signed_quantity']
            order_book_view[dates[-1]]['cumsum'] = order_book_view[dates[-1]].groupby('orderReference')[
                'signed_quantity'].transform(pd.Series.cumsum)

            # add integer price levels connected to prices
            time_ok = (order_book_view[dates[-1]].timestamp_hours >= (start_hour + (1 / 6))) & (
                    order_book_view[dates[-1]].timestamp_hours <= (end_hour - (1 / 6)))
            time_begin = order_book_view[dates[-1]].timestamp_hours < (start_hour + 1 / 6)
            price_min = np.min(
                order_book_view[dates[-1]][order_book_view[dates[-1]].timestamp_hours <= (end_hour - (1 / 6))]["price"])
            price_max = np.max(
                order_book_view[dates[-1]][order_book_view[dates[-1]].timestamp_hours <= (end_hour - (1 / 6))]["price"])
            total_num_levels = int((price_max - price_min) / tick_size) + 1
            order_book_view[dates[-1]]["price_level"] = (
                    (order_book_view[dates[-1]]["price"] - price_min) / tick_size).astype(int)

            # build order book after 10 minutes to use as start order book
            ob_start = np.zeros(total_num_levels)
            active_orders = {}
            bad_orders_begin = []
            for index, row in order_book_view[dates[-1]][time_begin].iterrows():
                if 0 <= row["price_level"] < total_num_levels:
                    # check if order tries to add buy LO where sell LO already or vice versa
                    if row["type"] == "A":
                        if row["direction"] == "S":
                            if (ob_start[row["price_level"]] < 0) or (row["price_level"] < bid(ob_start)):
                                bad_orders_begin.append(row["orderReference"])
                        elif row["direction"] == "B":
                            if (ob_start[row["price_level"]] > 0) or (row["price_level"] > ask(ob_start)):
                                bad_orders_begin.append(row["orderReference"])

                    if row["orderReference"] not in bad_orders_begin:
                        s = row["quantity"] if row["type"] == "A" else -row["quantity"]
                        ob_start[row["price_level"]] += s if row["direction"] == "S" else -s
                        if row["type"] == "A":
                            if row["price_level"] not in active_orders:
                                active_orders[row["price_level"]] = [row["orderReference"]]
                            else:
                                active_orders[row["price_level"]].append(row["orderReference"])
                        elif row["cumsum"] == 0:
                            active_orders[row["price_level"]] = [a for a in active_orders[row["price_level"]] if
                                                                 a != row["orderReference"]]
                            if len(active_orders[row["price_level"]]) == 0:
                                active_orders.pop(row["price_level"])
                    if (-1 if np.argwhere(ob_start < 0).shape[0] == 0 else np.argwhere(ob_start < 0)[-1]) > (
                            total_num_levels if np.argwhere(ob_start > 0).shape[0] == 0 else np.argwhere(ob_start > 0)[
                                0]):
                        print(index)
                        print(ob_start)
                        break
                else:
                    print(row["price_level"])

            cleaning_numbers[dates[-1]]["wrong side begin"] = order_book_view[dates[-1]][
                order_book_view[dates[-1]].orderReference.isin(bad_orders_begin)].shape[0]
            cleaning_numbers[dates[-1]]["begin"] = np.sum(
                time_begin & (order_book_view[dates[-1]].orderReference.isin(bad_orders_begin) == False))
            # compute order book for each event
            ob_view = order_book_view[dates[-1]][time_ok]
            ob_view = ob_view[ob_view.orderReference.isin(bad_orders_begin) == False].reset_index(drop=True)
            print('Number of events after basic cleaning: ', ob_view.shape[0])
            ob[dates[-1]] = np.zeros(
                (ob_view.shape[0] + 1, 2, num_levels + 1))  # keep only current range instead of this whole LOB
            ob[dates[-1]][0, ...] = get_lob(ob_start, num_levels)
            lob = LOB(ob[dates[-1]][0, ...].copy(), outside_volume=0)
            remove_orders = []
            good_index[dates[-1]] = [0]

            # current levels tracked in the order book representation (not tracking orders outside this range)
            current_levels = [lob.ask - num_levels, lob.bid + num_levels]
            for k in list(active_orders.keys()):
                if (k < current_levels[0]) or (k > current_levels[1]):
                    remove_orders = remove_orders + active_orders.pop(k)

            i = 0
            bad_index = []
            bad_orders = []
            for index, row in ob_view.iterrows():
                ob[dates[-1]][i + 1, ...] = ob[dates[-1]][i, ...].copy()
                # check if order tries to add buy LO where sell LO already or vice versa
                if 0 <= row["price_level"] < total_num_levels:
                    if row["type"] == "A":
                        if row["direction"] == "S":
                            if (lob.get_volume(row["price_level"], absolute_level=True) < 0) or (
                                    row["price_level"] < lob.bid):
                                bad_orders.append(row["orderReference"])
                                bad_index.append(i)
                        elif row["direction"] == "B":
                            if (lob.get_volume(row["price_level"], absolute_level=True) > 0) or (
                                    row["price_level"] > lob.ask):
                                bad_orders.append(row["orderReference"])
                                bad_index.append(i)

                    if row["orderReference"] not in bad_orders:
                        add_order = False
                        # check if order outside current range
                        if (row["price_level"] < current_levels[0]) or (row["price_level"] > current_levels[1]):
                            add_order = False

                        # add new lo inside current range
                        elif row["type"] == "A":
                            add_order = True
                            if row["price_level"] not in active_orders:
                                active_orders[row["price_level"]] = [row["orderReference"]]
                            else:
                                active_orders[row["price_level"]].append(row["orderReference"])

                        # cancel/mo then keep if in active orders and remove from active orders if fully done (cumsum=0)
                        elif row["price_level"] in active_orders:
                            if row["orderReference"] in active_orders[row["price_level"]]:
                                add_order = True
                                if row["cumsum"] == 0:
                                    active_orders[row["price_level"]] = [a for a in active_orders[row["price_level"]] if
                                                                         a != row["orderReference"]]
                                    if len(active_orders[row["price_level"]]) == 0:
                                        active_orders.pop(row["price_level"])

                        # add order to new OB
                        if add_order:
                            good_index[dates[-1]].append(i + 1)
                            s = row["quantity"] if row["type"] == "A" else -row["quantity"]
                            change_ok = lob.change_volume(row["price_level"], s if row["direction"] == "S" else -s,
                                                          absolute_level=True)
                            if not change_ok:
                                print(lob.data)
                                print(row)
                            ob[dates[-1]][i + 1, ...] = lob.data.copy()
                            current_levels = [lob.ask - num_levels, lob.bid + num_levels]

                            for k in list(active_orders.keys()):
                                if (k < current_levels[0]) or (k > current_levels[1]):
                                    remove_orders = remove_orders + active_orders.pop(k)
                        else:
                            if row["orderReference"] not in remove_orders:
                                remove_orders.append(row["orderReference"])

                    elif row["type"] != "A":
                        bad_index.append(i)

                    if (lob.bid >= lob.ask) or (np.min(lob.q_bid()) < 0) or (np.min(lob.q_ask()) < 0):
                        print(index)
                        print(ob[dates[-1]][i + 1, :])
                        break
                else:
                    print(row["price_level"])
                i += 1
            cleaning_numbers[dates[-1]]["wrong side"] = order_book_view[dates[-1]][
                order_book_view[dates[-1]].orderReference.isin(bad_orders)].shape[0]
            cleaning_numbers[dates[-1]]["outside range"] = ob_view.shape[0] - len(bad_index) - (
                    len(good_index[dates[-1]]) - 1)
            cleaning_numbers[dates[-1]]["clean"] = len(good_index[dates[-1]]) - 1

            print('Total number of events after full cleaning: ', len(good_index[dates[-1]]) - 1)
            print('Part of events kept: ', (len(good_index[dates[-1]]) - 1) / cleaning_numbers[dates[-1]]["initial"])

            # only keep events in good index
            ob[dates[-1]] = ob[dates[-1]][good_index[dates[-1]], ...]
            ob_ok = ob_view.iloc[np.array(good_index[dates[-1]][1:]) - 1]

            # add time diff between events
            ob_ok["timediff"] = ob_ok["timestamp"].diff()
            ob_begin = order_book_view[dates[-1]][time_begin]
            ob_begin = ob_begin[ob_begin.orderReference.isin(bad_orders + bad_orders_begin) == False]
            ob_ok["timediff"].iloc[0] = ob_ok.timestamp.iloc[0] - ob_begin.timestamp.iloc[-1]

            # add data to dict
            data_dict["date"].append(dates[-1])
            data_dict["ob"].append(ob[dates[-1]])
            data_dict["event"].append(get_event_types(ob_ok, data_dict["event_types"]))
            data_dict["size"].append(ob_ok.quantity.values)
            data_dict["abs_level"].append(ob_ok.price_level.values)
            data_dict["level"].append(relative_levels(data_dict, data_dict["event_types"]))
            data_dict["time"].append(ob_ok.timediff.values)

            # set start ask price to be the price level 0
            start_ask = int(ob[dates[-1]][0, 0, 0])
            data_dict["ob"][-1][:, 0, 0] -= start_ask
            data_dict["abs_level"][-1] -= start_ask

            for k, v in cleaning_numbers[dates[-1]].items():
                print(k, ": ", v)
    return data_dict, cleaning_numbers


if __name__ == '__main__':
    args = parse_args()
    data_dict, cleaning_numbers = main(args)
    pickle.dump(data_dict, open("{}//{}.pkl".format(args.directory, args.save_filename), "wb"))
    pickle.dump(cleaning_numbers, open("{}//cleaning_{}.pkl".format(args.directory, args.save_filename), "wb"))
