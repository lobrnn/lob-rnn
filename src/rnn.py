import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_addons.rnn import LayerNormLSTMCell

from .lob import LOB
from .training_utils import LobDataGenerator, EarlyStopByLosses


class LobRNN:

    def __init__(self, data, model_spec, event_types, seq_length=int(1e2), max_constant=1.2, integer_size=True,
                 outside_volume=0):
        """
        Sets up training data by splitting the data into shorter sequences and takes out the changes as levels and sizes

        Parameters
        ----------
        data : dictionary containing data about the simulation including
            ob: numpy array of shape (num_parts,N,num_levels) containing the LOB in all timesteps
            event: numpy array of event types in all timesteps
            level: numpy array of levels in all timesteps
            abs_level: numpy array of absolute levels in all timesteps
            size: numpy array of size in all timesteps
            time: numpy array of time between all events
        model_spec : dict containing functions for generating and log likelihood computation for size and time
        event_types : dict
            mapping between event types and integers
        seq_length : int
            wanted length of the shorter sequences
        max_constant : float
            multiplier of maximum in data to use as max for generated data
        integer_size : bool
            whether sizes are given as integers
        """
        self.event_types = event_types
        self.inverse_event_types = {v: k for k, v in event_types.items()}
        self.num_event_types = len(event_types)
        self.seq_length = seq_length
        self.num_levels = int(data["ob"][0].shape[-1] - 1)
        self.outside_volume = outside_volume

        self.apply_ask = lambda x: np.apply_along_axis(lambda d: LOB(d, self.outside_volume).ask_volume, -1,
                                                       x.reshape((x.shape[:-2] + (-1,))))
        self.apply_bid = lambda x: np.apply_along_axis(lambda d: LOB(d, self.outside_volume).bid_volume, -1,
                                                       x.reshape((x.shape[:-2] + (-1,))))
        self.num_features = 2
        all_features = [[self.apply_ask(d) for d in data["ob"]], [self.apply_bid(d) for d in data["ob"]]]
        feature_names = ["ask", "bid"]

        self.training_data = data
        for i in range(len(feature_names)):
            self.training_data[feature_names[i]] = all_features[i]
        for k, v in self.training_data.items():
            if k in ["size"]:
                self.training_data[k] = [np.abs(d) for d in self.training_data[k]]

        self.max_volume = max_constant * np.max([np.max(np.abs(d[..., 1:])) for d in self.training_data["ob"]])
        self.max_size = np.max([np.max(d) for d in self.training_data["size"]])
        self.max_time = max_constant * np.max([np.max(d) for d in self.training_data["time"]])
        if integer_size:
            self.max_volume = int(self.max_volume)
            self.max_size = int(self.max_size)
            for i in range(len(self.training_data["ob"])):
                self.training_data["ob"][i] = self.training_data["ob"][i].astype(int)
                if "abs_size" in data:
                    self.training_data["abs_size"][i] = self.training_data["abs_size"][i].astype(int)
                else:
                    self.training_data["size"][i] = self.training_data["size"][i].astype(int)
        if "abs_time" in data:
            self.training_data["time"][i] = self.training_data["time"][i].astype(int)

        self.models = None
        self.submodel_names = ["event", "level", "size", "time"]
        self.new_keys = {s: "abs_{}".format(s) if ("abs_{}".format(s) in self.training_data and s != "level") else s for
                         s in self.submodel_names}
        self.categorical = model_spec["categorical"]
        self.size_log_likelihood = model_spec["size_log_likelihood"]
        self.time_log_likelihood = model_spec["time_log_likelihood"]
        self.generate_size = model_spec["generate_size"]
        self.generate_time = model_spec["generate_time"]
        self.mask_size = model_spec["mask_size"]
        if "transform_size" in model_spec:
            self.transform_size = model_spec["transform_size"]
        if "transform_time" in model_spec:
            self.transform_time = model_spec["transform_time"]
        if "transform_size_bins" in model_spec:
            self.transform_size_bins = model_spec["transform_size_bins"]
        if "transform_time_bins" in model_spec:
            self.transform_time_bins = model_spec["transform_time_bins"]

        self.num_outputs = {"event": self.num_event_types, "level": self.num_levels,
                            "size": self.max_size if "size_num_outputs" not in model_spec else model_spec[
                                "size_num_outputs"], "time": model_spec["time_num_outputs"]}

    def compute_log_likelihood(self, out, inputs, submodel="event"):
        """
        Computes log likelihoods for each submodel

        Parameters
        ----------
        out : dict of output from all submodels
        inputs : dict of all tensorflow inputs
        submodel : string indicating which submodel to consider (event/level/size/time)

        Returns
        -------
        loglik : dict of all computed log likelihoods
        """
        loglik = None
        if submodel == "event":
            es = tf.squeeze(inputs["event_new"], -1)
            loglik = tfp.distributions.Categorical(out).log_prob(es)
        elif submodel == "level":
            loglik = tfp.distributions.Categorical(out).log_prob(tf.squeeze(inputs["level_new"], -1))
        elif submodel == "size":
            loglik = self.size_log_likelihood(out, inputs)
        elif submodel == "time":
            loglik = self.time_log_likelihood(out, inputs)
        return loglik

    def setup_model(self, model_args, predict=False, device='/gpu:0'):
        """
        Sets up keras model.

        Parameters
        ----------
        model_args :
            should contain
            num_units : int
                the number of units in each specific LSTM layer
            num_layers : int
                number of specific layers
            dropout : float
                between 0 and 1 of how much dropout to use for LSTM layers
            learning_rate : float
                learning rate to use with Adam optimizer
            batch_size: int
                batch size to use in the model
        predict : boolean
            whether to use model for predict (batch_size=1) or not
        device : string
            which device to use for model

        Returns
        -------
        model
            compiled keras model

        """
        batch_size = model_args.batch_size if not predict else 1
        return_sequences = False if predict else True

        with tf.device(device):
            def TimeDistWrap(layer):
                if not predict:
                    return tf.keras.layers.TimeDistributed(layer)
                else:
                    return layer

            def one_hot_encoding(in_tensor, num_values=self.num_event_types):
                out_shape = in_tensor.get_shape().as_list()
                out_shape[-1] = num_values
                out_tensor = tf.reshape(tf.one_hot(tf.cast(in_tensor, dtype=tf.int32), depth=num_values,
                                                   axis=len(in_tensor.shape) - 1), out_shape)
                return out_tensor

            def lstm_layers(inputs, num_layers=None, num_units=None, dropout=0):
                x = tf.keras.layers.Reshape((-1, inputs.get_shape().as_list()[-1]))(inputs)

                for layer in range(num_layers):
                    # x = tf.keras.layers.LSTM(num_units,
                    #                         return_sequences=return_sequences if layer == num_layers - 1 else True,
                    #                         stateful=False, unroll=True if predict else False)(x)

                    x = tf.keras.layers.RNN(LayerNormLSTMCell(num_units),
                                            return_sequences=return_sequences if layer == num_layers - 1 else True,
                                            stateful=False, unroll=True if predict else False)(x)

                    x = tf.keras.layers.Dropout(dropout)(x)

                return x

            # set up all input layers to the model
            if predict:
                inputs = {"ob_input": tf.keras.layers.Input(batch_shape=(1, 1, 2, self.num_levels + 1),
                                                            name='ob_input')}
                inputs["ob_last"] = tf.keras.layers.Lambda(lambda x: x[:, -1, :, :])(inputs["ob_input"])
                for a in ["ask", "bid"]:
                    inputs["{}_input".format(a)] = tf.keras.layers.Input(batch_shape=(1, 1, 1),
                                                                         name='{}_input'.format(a))
                    inputs["{}_last".format(a)] = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(
                        inputs["{}_input".format(a)])
                input_layer = lambda y: tf.keras.layers.Input(batch_shape=(batch_size, 1, 1), name="{}_input".format(y))
                last_layer = lambda y: tf.keras.layers.Lambda(lambda x: x[:, -1, :])(inputs["{}_input".format(y)])
                new_layer = lambda y: tf.keras.layers.Input(batch_shape=(batch_size, 1), name="{}_new".format(y))

            else:
                inputs = {"ob_input": tf.keras.layers.Input(
                    batch_shape=(model_args.batch_size, model_args.seq_length, 2, self.num_levels + 1),
                    name='ob_input')}
                inputs["ob_last"] = tf.keras.layers.Lambda(lambda x: x[:, :-1, :, :])(inputs["ob_input"])
                for a in ["ask", "bid"]:
                    inputs["{}_input".format(a)] = tf.keras.layers.Input(
                        batch_shape=(model_args.batch_size, model_args.seq_length, 1),
                        name='{}_input'.format(a))
                    inputs["{}_last".format(a)] = tf.keras.layers.Lambda(lambda x: x[:, :-1, :])(
                        inputs["{}_input".format(a)])
                input_layer = lambda y: tf.keras.layers.Input(batch_shape=(batch_size, model_args.seq_length, 1),
                                                              name="{}_input".format(y))
                last_layer = lambda y: tf.keras.layers.Lambda(lambda x: x[:, :-1, :])(inputs["{}_input".format(y)])
                new_layer = lambda y: tf.keras.layers.Lambda(lambda x: x[:, 1:, :])(inputs["{}_input".format(y)])

            last = []
            for s in self.submodel_names:
                if s in self.training_data:
                    inputs["{}_input".format(s)] = input_layer(s)
                    inputs["{}_last".format(s)] = last_layer(s)
                    inputs["{}_new".format(s)] = new_layer(s)
                    if self.categorical[s]:
                        inputs["{}_last".format(s)] = one_hot_encoding(inputs["{}_last".format(s)],
                                                                       num_values=self.num_outputs[s])
                    last.append(inputs["{}_last".format(s)])

            all_inputs = [v for k, v in inputs.items() if ("input" in k)]

            # layers for each submodel
            tf_inputs = tf.keras.layers.Concatenate()([tf.reshape(inputs["ob_last"], inputs["ob_last"].shape[:-2] + (
                inputs["ob_last"].shape[-2] * inputs["ob_last"].shape[-1],))] + last)
            model_inputs = all_inputs
            self.models = {}
            out = {}
            for s in self.submodel_names:
                if s in self.training_data:
                    print(s)
                    print('model inputs: ')
                    print(model_inputs)
                    print(tf_inputs)
                    x = lstm_layers(tf_inputs, model_args.num_layers,
                                    model_args.num_units,
                                    model_args.dropout)
                    x = TimeDistWrap(tf.keras.layers.Dense(self.num_outputs[s]))(x)

                    if s == "event":
                        # set cancellation sell/mo ask to zero when no volume
                        event_types_ask = [1 if self.inverse_event_types[i] in ["cancellation sell", "mo ask"] else 0
                                           for i in range(self.num_event_types)]
                        ask_0 = tf.reshape(
                            tf.tile(tf.constant(event_types_ask), tf.math.reduce_prod(x.shape[:-1], keepdims=True)),
                            x.shape)
                        ask_1 = tf.reshape(
                            tf.repeat(tf.cast(tf.equal(inputs["ask_last"], 0), ask_0.dtype), self.num_event_types),
                            ask_0.shape)
                        ask_mask = tf.equal(ask_0 + ask_1, 2)
                        x = tf.where(ask_mask, -np.inf * tf.ones_like(x), x)

                        # set cancellation buy/mo bid to zero when bid == -1
                        event_types_bid = [1 if self.inverse_event_types[i] in ["cancellation buy", "mo bid"] else 0 for
                                           i in range(self.num_event_types)]
                        bid_0 = tf.reshape(
                            tf.tile(tf.constant(event_types_bid), tf.math.reduce_prod(x.shape[:-1], keepdims=True)),
                            x.shape)
                        bid_1 = tf.reshape(
                            tf.repeat(tf.cast(tf.equal(inputs["bid_last"], 0), bid_0.dtype), self.num_event_types),
                            bid_0.shape)
                        bid_mask = tf.equal(bid_0 + bid_1, 2)
                        x = tf.where(bid_mask, -np.inf * tf.ones_like(x), x)

                    elif s == "level":
                        # MO: best bid/ask (level = 0) with prob 1 (logit inf)
                        mo_mask = tf.logical_or(tf.equal(inputs['event_new'], self.event_types["mo ask"]),
                                                tf.equal(inputs['event_new'], self.event_types["mo bid"]))
                        mo_mask = tf.reshape(tf.repeat(mo_mask, x.shape[-1]), x.shape)
                        x = tf.where(mo_mask, tf.one_hot(tf.zeros(inputs["ob_last"][..., 0, 0].shape, dtype=tf.int32),
                                                         depth=x.shape[-1], on_value=0, off_value=-np.inf,
                                                         dtype=x.dtype), x)

                        # cancellations: only levels with volume on right side possible
                        canc_buy_mask = tf.reshape(
                            tf.repeat((tf.equal(inputs['event_new'], self.event_types["cancellation buy"])),
                                      x.shape[-1]), x.shape)
                        canc_sell_mask = tf.reshape(
                            tf.repeat((tf.equal(inputs['event_new'], self.event_types["cancellation sell"])),
                                      x.shape[-1]), x.shape)
                        canc_buy_cond = tf.not_equal(inputs["ob_last"][..., 1, 1:], 0)
                        x = tf.where(tf.logical_or(tf.equal(canc_buy_mask, False), canc_buy_cond), x, -np.inf)
                        canc_sell_cond = tf.not_equal(inputs["ob_last"][..., 0, 1:], 0)
                        x = tf.where(tf.logical_or(tf.equal(canc_sell_mask, False), canc_sell_cond), x, -np.inf)
                    elif s == "size":
                        x = self.mask_size(x, inputs, self.event_types)

                    self.models[s] = tf.keras.models.Model(inputs=model_inputs, outputs=x, name="{}_model".format(s))
                    print('model output: ', x)
                    out[s] = self.models[s](model_inputs)
                    if s != "time":
                        if self.categorical[s]:
                            tf_inputs = tf.keras.layers.Concatenate()(
                                [tf_inputs,
                                 one_hot_encoding(inputs["{}_new".format(s)], num_values=self.num_outputs[s])])
                        else:
                            tf_inputs = tf.keras.layers.Concatenate()([tf_inputs, inputs["{}_new".format(s)]])
                        if predict:
                            model_inputs = model_inputs + [inputs["{}_new".format(s)]]

            # set up optimizer to use negative log likelihood as loss
            if not predict:
                nll = {}
                loss = {}
                for submodel in self.submodel_names:
                    nll[submodel] = tf.math.negative(self.compute_log_likelihood(out[submodel], inputs, submodel),
                                                     name=submodel)
                    loss[submodel] = lambda y_true, y_pred: y_pred

                self.models["nll"] = tf.keras.Model(model_inputs, nll, name="model")
                adam = tf.keras.optimizers.Adam(learning_rate=model_args.learning_rate, clipnorm=1.)
                self.models["nll"].compile(optimizer=adam, loss=loss)

        return self.models

    def train_model(self, model, checkpoint_path, batch_size=32, epochs=100, early_stopping=True,
                    initial_epoch=0, val_split=.1):
        """
        Train tf.keras model.

        Parameters
        ----------
        model : compiled keras model
        checkpoint_path : string
            path to where to save checkpoints
        batch_size : int
            batch size to use
        epochs : int
            number of epochs to run
        early_stopping : bool
            whether to use stop the training once the val loss has not decreased for a number of epochs
        initial_epoch : int
            which epoch to start from, used if training is resumed
        val_split : float between 0 and 1
            how big part of data to use for validation
        """

        train_dict = {}
        val_dict = {}
        num_parts = len(self.training_data["event"])
        seq_length = model.input_shape[0][1]
        if num_parts > 1:
            val_size = int(np.ceil(num_parts * val_split))

            for k, v in self.training_data.items():
                train_dict[k] = v[:-val_size]
                val_dict[k] = v[-val_size:]
        else:
            val_size = int(np.ceil(self.training_data["event"][0].shape[0] * val_split))
            for k, v in self.training_data.items():
                train_dict[k] = v[0][:-val_size, ...]
                val_dict[k] = v[0][-val_size - 1:, ...] if k in ["ob", "ask", "bid"] else v[0][-val_size:, ...]

        train_generator = LobDataGenerator(train_dict, model, seq_length=seq_length, batch_size=batch_size,
                                           shuffle=True, return_y=True)
        val_generator = LobDataGenerator(val_dict, model, seq_length=seq_length, batch_size=batch_size,
                                         shuffle=False, return_y=True)
        callbacks = [EarlyStopByLosses(patience=10)] if early_stopping else []
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path))
        train_history = model.fit(train_generator, epochs=epochs, verbose=1,
                                  batch_size=batch_size, initial_epoch=initial_epoch, validation_data=val_generator,
                                  callbacks=callbacks)
        return train_history

    def generate_event(self, model, pred_dict, gen_dict, j):
        """
        Generates one event

        Parameters
        ----------
        model : dict of all submodels to use to predict parameters for samples
        pred_dict : dict of arrays to use as input to model
        gen_dict : dict of arrays of generated data
        j : index in sequence of event to generate

        Returns
        -------
        model : updated model
        gen_dict : updated dict of generated data
        integer of 1 (all ok)/0 (not accepted)/-1 (weird behaviour, interrupt generation)
        """
        lob = LOB(gen_dict["ob"][j - 1, ...].copy(), self.outside_volume)

        # generate event type
        e_logits = model["event"].predict(pred_dict).flatten()
        e_probs = np.exp(e_logits - np.max(e_logits))

        if np.sum(e_probs) == 0:
            print("no event with prob > 0")
            return model, gen_dict, -1
        if not np.all(np.isfinite(e_probs)):
            print('e_logits: ', e_logits)
        e_probs = e_probs / np.sum(e_probs)
        if not np.all(np.isfinite(e_probs)):
            print('e_logits: ', e_logits)
            for k, v in pred_dict.items():
                print(k)
                print(v)
        gen_dict["event"][j] = np.random.choice(e_probs.shape[0], p=e_probs)
        pred_dict["event_new"] = gen_dict["event"][j].reshape((1, 1))

        # generate level
        l_logits = model["level"].predict(pred_dict).flatten()
        l_probs = np.exp(l_logits - np.where(np.max(l_logits) > -np.inf, np.max(l_logits), 0))
        gen_dict["level"][j] = np.random.choice(self.num_levels, p=l_probs / np.sum(l_probs))

        # add absolute price levels
        if gen_dict["event"][j] in [self.event_types['lo buy'], self.event_types['cancellation buy']]:
            gen_dict["abs_level"][j] = lob.ask - gen_dict["level"][j] - 1
        elif gen_dict["event"][j] == self.event_types['mo ask']:
            gen_dict["abs_level"][j] = lob.ask
        elif gen_dict["event"][j] == self.event_types['mo bid']:
            gen_dict["abs_level"][j] = lob.bid
        else:
            gen_dict["abs_level"][j] = lob.bid + gen_dict["level"][j] + 1
        pred_dict["level_new"] = gen_dict["level"][j].reshape((1, 1))

        # generate size
        s_params = model["size"].predict(pred_dict).flatten()
        gen_dict["size"][j], size_ok = self.generate_size(s_params, gen_dict["event"][j])
        pred_dict["size_new"] = gen_dict["size"][j].reshape((1, 1))
        if "abs_size" in self.training_data:
            gen_dict["abs_size"][j] = self.transform_size(gen_dict["size"][j], gen_dict["event"][j], np.abs(
                lob.get_volume(gen_dict["abs_level"][j], absolute_level=True)))
            pred_dict["abs_size_new"] = gen_dict["abs_size"][j].reshape((1, 1))

        # generate time
        t_params = model["time"].predict(pred_dict).flatten()
        gen_dict["time"][j] = self.generate_time(t_params)
        if "abs_time" in self.training_data:
            gen_dict["abs_time"][j] = self.transform_time(gen_dict["time"][j])

        # add change to order book
        size = gen_dict[self.new_keys["size"]][j]
        if self.inverse_event_types[gen_dict["event"][j]] in ["mo bid", "lo sell", "cancellation buy"]:
            change_ok = lob.change_volume(gen_dict["abs_level"].flat[j], size, absolute_level=True)
        else:
            change_ok = lob.change_volume(gen_dict["abs_level"].flat[j], -size, absolute_level=True)

        gen_dict["ob"][j, ...] = lob.data.copy()
        gen_dict["ask"][j] = self.apply_ask(gen_dict["ob"][j, ...])
        gen_dict["bid"][j] = self.apply_bid(gen_dict["ob"][j, ...])

        return model, gen_dict, 1

    def generate_sequence(self, seed_dict, model, max_sequence_len, reset=True, max_time=False, reset_weights=False):
        """
        Generate the continuation of the sequence.

        Parameters
        ----------
        seed_dict : dictionary containing data about the seed sequence to use for simulation
            ob: numpy array of shape (s,num_levels) where 0<s<max_sequence_len
            event: numpy array of event types of shape (s)
            level: numpy array of levels of shape (s)
            abs_level: numpy array of absolute levels of shape (s)
            time: numpy array of time between all events of shape (s)
        model : keras model
        max_sequence_len : int
            length of sequences for the model
        reset : boolean
            whether to reset RNN in beginning
        max_time : boolean
            whether the max_sequence_len is given in time, if False then max_sequence_len given in number of events
        reset_weights : boolean
            whether to reset weights one event back if max time exceeded

        Returns
        -------
        gen_dict
            dictionary containing generated data
        boolean
            indicating success (True) or weird sequence (False)

        """
        with tf.device("CPU:0"):
            if reset:
                if type(model) == dict:
                    for k, v in model.items():
                        v.reset_states()
                else:
                    model.reset_states()
            seed_dict = seed_dict.copy()
            if seed_dict["ob"].ndim < 3:
                for k, v in seed_dict.items():
                    if k in ["ob"]:
                        seed_dict[k] = v.reshape((1, 2, -1))
                    else:
                        seed_dict[k] = np.array(v).reshape(1)

            # run seed sequence through network
            seed_len = seed_dict["ob"].shape[0]
            if seed_len > 1:
                for st in range(seed_len - 1):
                    in_dict = {}
                    for k, v in seed_dict.items():
                        if k.startswith("ob"):
                            in_dict["{}_input".format(k)] = v[np.newaxis, np.newaxis, st, :]
                            in_dict["ask_input"] = self.apply_ask(in_dict["{}_input".format(k)]).reshape((1, 1, 1))
                            in_dict["bid_input"] = self.apply_bid(in_dict["{}_input".format(k)]).reshape((1, 1, 1))
                        else:
                            in_dict["{}_input".format(k)] = v[st].reshape((1, 1, 1))
                            in_dict["{}_new".format(k)] = v[st + 1].reshape((1, 1))

                    for k, v in model.items():
                        v.predict(in_dict)

            # set up dict of data arrays to save generated data in
            gen_dict = {}
            gen_len = (max_sequence_len - seed_len + 1) if not max_time else 1
            for k, v in seed_dict.items():
                if k == "ob":
                    gen_dict["ob"] = np.zeros((gen_len, 2, self.num_levels + 1), dtype=self.training_data[k][0].dtype)
                    gen_dict["ob"][0, ...] = seed_dict["ob"][-1, ...]
                elif k not in ["bid", "ask"]:
                    gen_dict[k] = np.zeros(gen_len, dtype=self.training_data[k][0].dtype)
                    gen_dict[k][0] = v[-1]
            if "abs_level" not in gen_dict:
                gen_dict["abs_level"] = np.zeros(gen_len, dtype=int)

            gen_dict["ask"] = np.zeros(gen_len, self.training_data["ask"][0].dtype)
            gen_dict["ask"][0] = self.apply_ask(seed_dict["ob"][-1, ...])
            gen_dict["bid"] = np.zeros(gen_len, self.training_data["bid"][0].dtype)
            gen_dict["bid"][0] = self.apply_bid(seed_dict["ob"][-1, ...])

            # loop over each event in sequence
            j = 0
            total_time = 0
            done = (j == gen_len - 1) if not max_time else (total_time >= max_sequence_len)

            while not done:
                j += 1
                accepted = False
                if reset_weights:
                    model_weights = {key: m.get_weights() for key, m in model.items()}
                if max_time:
                    for k, v in gen_dict.items():
                        gen_dict[k] = np.append(v, np.zeros((1,) + v.shape[1:], dtype=int), axis=0).astype(v.dtype)

                # set up latest event info to use as input to generate next
                pred_dict = {}
                for k, v in gen_dict.items():
                    if k in ["ob"]:  # , "bid", "ask"]:
                        pred_dict["{}_input".format(k)] = v[j - 1, ...].reshape((1, 1, 2, -1))
                    else:
                        pred_dict["{}_input".format(k)] = v[j - 1].reshape((1, 1, 1))

                # generate next event
                while not accepted:
                    model, gen_dict, success = self.generate_event(model, pred_dict, gen_dict, j)
                    if success == 1:
                        accepted = True
                    elif success == 0:
                        accepted = False
                    elif success == -1:
                        return gen_dict, False
                total_time += gen_dict[self.new_keys["time"]][j]
                done = (j == gen_len - 1) if not max_time else (total_time >= max_sequence_len)
            if max_time and (total_time > max_sequence_len):
                if reset_weights:
                    for key, m in model.items():
                        m.set_weights(model_weights[key])
                for k, v in gen_dict.items():
                    gen_dict[k] = v[:-1, ...]
            return gen_dict, True

    def run_order(self, initial_dict, model, t, order_type="mo ask", size=1):
        """
        Run market order in time t

        Parameters
        ----------
        initial_dict : dictionary containing data about current state
            ob: numpy array
            event: int
            level: int
            abs_level: int
            time: float
        model : keras model
            model to update with the market order
        t : float
            time until market order
        order_type : string
            type of order, one of "mo ask", "mo bid", "lo buy", "lo sell"
        size :  int
            size of the order

        Returns
        -------
        event_dict
            dict of new state of model
        """
        lob = LOB(initial_dict["ob"].copy(), self.outside_volume)
        pred_dict = {}
        for k, v in initial_dict.items():
            if k == "ob":
                pred_dict["{}_input".format(k)] = np.array(v).reshape((1, 1, 2, -1))
            else:
                pred_dict["{}_input".format(k)] = np.array(v).reshape((1, 1, -1))

        pred_dict["ask_input"] = self.apply_ask(initial_dict["ob"]).reshape((1, 1, -1))
        pred_dict["bid_input"] = self.apply_bid(initial_dict["ob"]).reshape((1, 1, -1))

        pred_dict["event_new"] = np.array(self.event_types[order_type]).reshape((1, 1))
        pred_dict["abs_level_new"] = np.array(
            lob.ask if "ask" in order_type else lob.bid).reshape((1, 1)).astype(int).copy()
        pred_dict["level_new"] = np.array([0]).reshape((1, 1))
        if self.new_keys["size"] == "abs_size":
            pred_dict["abs_size_new"] = np.abs(np.array([size]).reshape((1, 1)))
            pred_dict["size_new"] = np.array([self.transform_size_bins(size, self.event_types[order_type], np.abs(
                lob.get_volume(pred_dict["abs_level_new"].flat[0], absolute_level=True)))]).reshape((1, 1))
        else:
            pred_dict["size_new"] = np.abs(np.array([size]).reshape((1, 1)))

        _ = model["event"].predict(pred_dict)
        _ = model["level"].predict(pred_dict)
        _ = model["size"].predict(pred_dict)
        _ = model["time"].predict(pred_dict)

        if self.new_keys["time"] == "abs_time":
            event_dict = {"abs_time": t, "time": self.transform_time_bins(t), "ob": initial_dict["ob"].copy()}
        else:
            event_dict = {"time": t, "ob": initial_dict["ob"].copy()}

        if order_type in ["mo ask", "lo buy"]:
            change_ok = lob.change_volume(pred_dict["abs_level_new"].flat[0], -size, absolute_level=True)
        else:
            change_ok = lob.change_volume(pred_dict["abs_level_new"].flat[0], size, absolute_level=True)
        event_dict["ob"] = lob.data.copy()

        for k in ["event", "level", "size", "abs_level", "abs_size"]:
            if k in self.training_data:
                event_dict[k] = pred_dict["{}_new".format(k)][0]

        return event_dict

    def run_twap(self, model, volume, end_time, timefactor, initial_dict, padding=(0, 0)):
        """
        Runs a TWAP on the model buying volume with fixed timedelta

        Parameters
        ----------
        model : keras model
        volume : int
            total volume to buy
        end_time : float
            total time for TWAP to run
        timefactor : float
            how often to save mid/ask/bid of order book
        initial_dict : dict
            starting point of the LOB and last event
        padding : tuple of two floats
            how much time to add before the first trade/after the last

        Returns
        -------
        twap_dict
            dictionary containing information about the twap
        """
        if type(model) == dict:
            for k, v in model.items():
                v.reset_states()
        else:
            model.reset_states()
        twap_dict = {}

        if volume > 1:
            timedelta = (end_time - padding[0] - padding[1]) / (volume - 1)
        else:
            timedelta = 1

        num_times = int(end_time * timefactor) + 2

        for s in ["mid", "ask", "bid"]:
            twap_dict["{}_vals".format(s)] = np.zeros(num_times)
        twap_dict["time_vals"] = np.zeros(num_times + 1)
        twap_dict["time_vals"][:-1] = np.linspace(-(1 / timefactor), end_time, num_times)
        twap_dict["time_vals"][-1] = np.inf

        seed_dict = initial_dict.copy()
        for k, v in seed_dict.items():
            seed_dict[k] = v.squeeze()

        gen_dict, f = self.generate_sequence(seed_dict, model, padding[0], reset=False, max_time=True,
                                             reset_weights=True)
        if not f:
            twap_dict["finished"] = False
            return twap_dict
        total_time = padding[0]
        t = padding[0] - np.sum(gen_dict["time"][1:])
        for k, v in seed_dict.items():
            seed_dict[k] = gen_dict[k][-1, ...]
        seed_dict = self.run_order(seed_dict, model, t, order_type="mo ask")
        volume_left = volume - 1
        twap_dict["price"] = seed_dict["abs_level"]

        for vol in range(volume_left):
            gen_dict_new, f = self.generate_sequence(seed_dict, model, timedelta, reset=False, max_time=True,
                                                     reset_weights=True)
            for k, v in gen_dict.items():
                gen_dict[k] = np.append(v, gen_dict_new[k], axis=0).astype(v.dtype)
            if not f:
                twap_dict["finished"] = False
                return twap_dict
            total_time += np.sum(gen_dict_new["time"][1:])
            t = padding[0] + timedelta * (vol + 1) - total_time
            total_time = padding[0] + timedelta * (vol + 1)
            for k, v in seed_dict.items():
                seed_dict[k] = gen_dict[k][-1, ...]
            seed_dict = self.run_order(seed_dict, model, t, order_type="mo ask")
            twap_dict["price"] += seed_dict["abs_level"]

        gen_dict_new, f = self.generate_sequence(seed_dict, model, padding[1], reset=False, max_time=True)
        for k, v in gen_dict.items():
            gen_dict[k] = np.append(v, gen_dict_new[k], axis=0).astype(v.dtype)
        if not f:
            twap_dict["finished"] = False
            return twap_dict

        # TWAP dict containing: num_events/num_mo_ask/num_mo_bid/mid_vals/bid_vals/ask_vals
        twap_dict["num_events"] = gen_dict["event"].shape[0] - 1
        twap_dict["num_mo_ask"] = np.sum(gen_dict["event"][1:] == self.event_types["mo ask"]) - volume
        twap_dict["num_mo_bid"] = np.sum(gen_dict["event"][1:] == self.event_types["mo bid"])
        current_time = 0
        time_index = 0
        for i in range(twap_dict["num_events"] + 1):
            if i == twap_dict["num_events"]:
                current_time = end_time
            else:
                current_time += gen_dict["time"][i + 1]
            lob = LOB(gen_dict["ob"][i, ...], self.outside_volume)
            mid_current = lob.mid
            ask_current = lob.ask
            bid_current = lob.bid
            while current_time > twap_dict["time_vals"][time_index]:
                twap_dict["mid_vals"][time_index] = mid_current
                twap_dict["ask_vals"][time_index] = ask_current
                twap_dict["bid_vals"][time_index] = bid_current
                time_index += 1
        twap_dict["mid_vals"][time_index] = mid_current
        twap_dict["ask_vals"][time_index] = ask_current
        twap_dict["bid_vals"][time_index] = bid_current
        twap_dict["finished"] = True
        return twap_dict
