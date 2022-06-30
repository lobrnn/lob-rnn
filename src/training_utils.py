import logging

import numpy as np
import tensorflow.keras as keras


class LobDataGenerator(keras.utils.Sequence):
    def __init__(self, data, model=None, seq_length=100, batch_size=32, shuffle=True, return_y=False):
        self.diff_size = True if type(data["event"]) == list else False
        self.batch_size = batch_size
        self.data = data
        self.shuffle = shuffle
        self.seq_length = seq_length
        self.num_parts = len(data["event"]) if self.diff_size else 1
        self.num_seq = np.array([d.shape[0] for d in data["event"]]) if self.diff_size else data["event"].shape[0]
        self.seq_per_part = self.num_seq - seq_length + 1
        if self.diff_size:
            self.cumsum = np.cumsum(self.seq_per_part)
        self.model = model
        self.return_y = return_y
        if self.return_y:
            self.y = {}
            for k, v in model.output.items():
                self.y[k] = np.zeros(v.shape, dtype=v.dtype.as_numpy_dtype)
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(np.sum(self.seq_per_part) / self.batch_size))

    def __getitem__(self, index):
        # Sample indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Get data
        batch_dict = {}
        if self.diff_size:
            indexes_2d = self.get_2d_index(indexes)
            for k, v in self.data.items():
                if k in ["ob", "features", "ask", "bid"]:
                    if k == "ob":
                        v_idx = [
                            v[indexes_2d[0][i]][np.newaxis, indexes_2d[1][i] + 1:indexes_2d[1][i] + 1 + self.seq_length,
                            ...] for i in range(self.batch_size)]
                    else:
                        v_idx = [
                            v[indexes_2d[0][i]][np.newaxis, indexes_2d[1][i] + 1:indexes_2d[1][i] + 1 + self.seq_length,
                            np.newaxis] for i in range(self.batch_size)]
                else:
                    v_idx = [
                        v[indexes_2d[0][i]][np.newaxis, indexes_2d[1][i]:indexes_2d[1][i] + self.seq_length, np.newaxis]
                        for i in range(self.batch_size)]
                batch_dict["{}_input".format(k)] = np.concatenate(v_idx, axis=0)

        else:
            for k, v in self.data.items():
                if k in ["ob", "features", "ask", "bid"]:
                    index_1 = [range(i + 1, i + 1 + self.seq_length) for i in indexes]
                    if k == "ob":
                        all_index = (index_1, slice(None), slice(None))
                    else:
                        all_index = (index_1, np.newaxis)
                else:
                    index_1 = [range(i, i + self.seq_length) for i in indexes]
                    all_index = (index_1, np.newaxis)
                batch_dict["{}_input".format(k)] = v[all_index]

        if self.model is not None:
            if type(self.model) == dict:
                for k, v in self.model.items():
                    v.reset_states()
            else:
                self.model.reset_states()
        if self.return_y:
            return batch_dict, self.y
        else:
            return batch_dict

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(np.sum(self.seq_per_part))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_2d_index(self, indexes):
        if self.diff_size:
            seq_id = []
            id_in_seq = []
            for i in indexes:
                seq_id.append(np.argmin(i >= self.cumsum))
                id_in_seq.append(i if seq_id[-1] == 0 else i - self.cumsum[seq_id[-1] - 1])
            return seq_id, id_in_seq
        else:
            return indexes


class EarlyStopByLosses(keras.callbacks.Callback):
    def __init__(self, patience=10):
        super().__init__()
        self.patience = patience
        self.loss_keys = None
        self.wait = 0
        self.best = {}

    def on_epoch_end(self, epoch, logs=None):
        if self.loss_keys is None:
            self.loss_keys = []
            for key in logs.keys():
                if key.startswith("val_"):
                    self.loss_keys.append(key)
            self.best = {k: np.inf for k in self.loss_keys}
            print(self.loss_keys)
        add_wait = True
        for k in self.loss_keys:
            loss_changes = logs[k] - self.best[k]
            if loss_changes < 0:
                add_wait = False
                self.best[k] = logs[k]
                self.wait = 0
        if add_wait:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True


class EarlyStopByValLoss(keras.callbacks.EarlyStopping):
    def __init__(self, patience=10, restore_best_weights=True):
        super().__init__(patience=patience, restore_best_weights=restore_best_weights)

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = np.mean(logs.get('val_loss'))
        if monitor_value is None:
            logging.warning('Early stopping conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))
        return monitor_value
