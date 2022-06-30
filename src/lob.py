import numpy as np


class LOB:
    def __init__(self, data, outside_volume=1, include_spread_levels=True):
        self.data = data.reshape((2, -1))
        self.num_levels = self.data.shape[1] - 1
        self.outside_volume = outside_volume
        self.include_spread_levels = include_spread_levels

    @property
    def ask(self):
        return int(self.data[0, 0])

    @property
    def bid(self):
        return int(self.data[0, 0] + self.data[1, 0])

    @property
    def mid(self):
        return (self.bid + self.ask) / 2

    @property
    def vwap_mid(self):
        vwap_a = np.dot(self.data[0, 1:], self.bid + 1 + np.arange(self.num_levels)) / np.sum(self.data[0, 1:])
        vwap_b = np.dot(-self.data[1, 1:], self.ask - 1 - np.arange(self.num_levels)) / np.sum(-self.data[1, 1:])
        vwap = (vwap_a + vwap_b) / 2
        return vwap

    @property
    def microprice(self):
        bid_volume = self.bid_volume if self.bid_volume != 0 else self.outside_volume
        ask_volume = self.ask_volume if self.ask_volume != 0 else self.outside_volume
        if bid_volume == 0 and ask_volume == 0:
            return self.mid
        else:
            return (self.ask * bid_volume + self.bid * ask_volume) / (bid_volume + ask_volume)

    @property
    def spread(self):
        return int(-self.data[1, 0])

    @property
    def relative_bid(self):
        return int(self.data[1, 0])

    @property
    def ask_volume(self):
        return 0 if self.spread > self.num_levels else (
            self.data[0, self.spread] if self.include_spread_levels else self.data[0, 1])

    @property
    def bid_volume(self):
        return 0 if self.spread > self.num_levels else (
            -self.data[1, self.spread] if self.include_spread_levels else -self.data[1, 1])

    def buy_n(self, n=1):
        total_price = 0
        level = 0
        while n > 0:
            if level >= self.num_levels:
                available = self.outside_volume
                if self.outside_volume == 0:
                    print("TOO LARGE VOLUME")
                    return total_price
            else:
                available = self.ask_n_volume(level)
            vol = np.min((n, available))
            n -= vol
            total_price += (self.bid + 1 + level) * vol
            level += 1
        return total_price

    def sell_n(self, n=1):
        total_price = 0
        level = 0
        while n > 0:
            if level >= self.num_levels:
                available = self.outside_volume
                if self.outside_volume == 0:
                    print("TOO LARGE VOLUME")
                    return total_price
            else:
                available = self.bid_n_volume(level)
            vol = np.min((n, available))
            n -= vol
            total_price += (self.ask - 1 - level) * vol
            level += 1
        return total_price

    def ask_n(self, n=0, absolute_level=False):
        if n == 0 and not absolute_level:
            return 0
        level = int(self.relative_bid + n + 1) if self.include_spread_levels else n
        if absolute_level:
            level += self.ask
        return level

    def bid_n(self, n=0, absolute_level=False):
        level = -n - 1 if self.include_spread_levels else int(self.relative_bid - n)
        if absolute_level:
            level += self.ask
        return level

    def ask_n_volume(self, n=0):
        return self.data[0, n + 1]

    def bid_n_volume(self, n=0):
        return -self.data[1, n + 1]

    def q_ask(self, num_levels=None):
        if num_levels is None:
            num_levels = self.num_levels
        x = self.data[0, 1:1 + num_levels]
        return x

    def q_bid(self, num_levels=None):
        if num_levels is None:
            num_levels = self.num_levels
        x = -self.data[1, 1:1 + num_levels]
        return x

    def get_volume(self, level, absolute_level=False):
        if absolute_level:
            level = level - self.ask
        if level == 0:
            return self.ask_volume
        if self.include_spread_levels:
            if (level > 0) and ((self.spread + level - 1) < self.num_levels):
                return self.data[0, self.spread + level]
            elif (level < 0) and ((-level - 1) < self.num_levels):
                return self.data[1, -level]
            else:
                return 0
        else:
            if (level > 0) and (level < self.num_levels):
                return self.data[0, level + 1]
            elif (level < 0) and (-level < self.spread):
                return 0
            elif (level < 0) and (-level - self.spread < self.num_levels):
                return self.data[1, -level - self.spread + 1]
            else:
                return 0

    def order_imbalance(self, depth=None):
        """
        Compute order imbalance for levels up to a specified depth from best bid/ask
        such that high order imbalance means more volume available on bid side (more volume wanting to buy).

        Parameters
        ----------
        depth : int
            specifies how many levels from (and including) best bid and best ask to consider

        Returns
        -------
        float between 0 and 1 of order imbalance

        """
        if depth is None:
            depth = self.num_levels
        vol_sell = np.sum(
            self.data[0, self.spread:self.spread + depth] if self.include_spread_levels else self.data[0, 1:1 + depth])
        vol_buy = np.sum(
            -self.data[1, self.spread:self.spread + depth] if self.include_spread_levels else -self.data[1, 1:1 + depth])
        if vol_buy + vol_sell == 0:
            return 0.5
        else:
            return (vol_buy - vol_sell) / (vol_buy + vol_sell)

    def change_volume(self, level, volume, absolute_level=False, print_info=False):
        if absolute_level:
            level -= self.ask

        offset = self.spread if self.include_spread_levels else 1
        if level == 0:
            if self.ask_volume + volume < -1e-6:
                print('LEVEL 0')
                print(self.data)
                print('level: ', level)
                print('volume: ', volume)
                print(self.ask_volume + volume)
                return False
            self.data[0, offset] = np.round(self.data[0, offset] + volume, decimals=6)

            # if ask volume 0, move ask
            if self.data[0, offset] == 0:
                old_spread = self.spread
                index = np.argwhere(self.data[0, offset:])
                if index.size == 0:
                    if print_info:
                        print(self.data)
                        print(level)
                        print(volume)
                    move_ask = self.num_levels - offset + 1
                    self.data[0, 0] += move_ask
                    self.data[1, 0] -= move_ask
                    self.data[:, 1:] = 0
                else:
                    move_ask = index.flat[0]
                    self.data[0, 0] += move_ask
                    self.data[1, 0] -= move_ask
                    if self.include_spread_levels:
                        self.data[0, old_spread] = 0
                        self.data[1, 1 + move_ask:] = self.data[1, 1:-move_ask]
                        self.data[1, 1:1 + move_ask] = 0
                    else:
                        self.data[0, 1:-move_ask] = self.data[0, 1 + move_ask:]
                        self.data[0, -move_ask:] = self.outside_volume
            return True
        elif level > 0:
            # outside of LOB range
            if level > (self.num_levels - offset):
                return True
            # taking away too much volume not possible
            elif self.data[0, offset + level] + volume < 0:
                return False
            # transition by adding volume on ask side
            else:
                self.data[0, offset + level] = np.round(self.data[0, offset + level] + volume, decimals=6)
                return True

        else:
            # outside of LOB range
            if -level > self.num_levels + self.spread - offset:
                return True

            level_index = -level if self.include_spread_levels else -level - self.spread + 1

            # if adding volume inside the spread
            if self.spread + level > 0:
                old_spread = self.spread

                # new bid
                if volume < 0:
                    self.data[1, 0] = level
                    if self.include_spread_levels:
                        self.data[1, -level] = volume
                        self.data[0, 1:-old_spread - level] = self.data[0, 1 + old_spread + level:]
                        self.data[0, -old_spread - level:] = self.outside_volume
                    else:
                        self.data[1, 1 + old_spread + level:] = self.data[1, 1:-(old_spread + level)]
                        self.data[1, 1:old_spread + level] = 0
                        self.data[1, old_spread + level] = volume
                    return True

                # new ask
                else:
                    old_ask = self.ask
                    self.data[0, 0] = old_ask + level
                    self.data[1, 0] -= level
                    if self.include_spread_levels:
                        self.data[0, self.spread] = volume
                        self.data[1, 1:level] = self.data[1, 1 - level:]
                        self.data[1, level:] = -self.outside_volume
                    else:
                        self.data[0, 1 + old_spread + level:] = self.data[0, 1:-(old_spread + level)]
                        self.data[0, 1:old_spread + level] = 0
                        self.data[0, old_spread + level] = volume
                    return True

            # taking away too much volume not possible
            elif self.data[1, level_index] + volume > 0:
                return False

            # transition by adding volume on bid side
            else:
                self.data[1, level_index] = np.round(self.data[1, level_index] + volume, decimals=6)

                # adjust if best bid changed
                if (self.data[1, level_index] == 0) and (self.relative_bid == level):
                    old_spread = self.spread
                    index = np.argwhere(self.data[1, level_index + 1:])
                    if index.size == 0:
                        if print_info:
                            print(self.data)
                            print(level)
                            print(volume)
                        move_bid = self.num_levels - offset + 1
                        self.data[1, 0] -= move_bid
                        self.data[:, 1:] = 0
                    else:
                        move_bid = index.flat[0] + 1
                        self.data[1, 0] -= move_bid
                        if self.include_spread_levels:
                            self.data[1, old_spread] = 0
                            self.data[0, 1 + move_bid:] = self.data[0, 1:-move_bid]
                            self.data[0, 1:1 + move_bid] = 0
                        else:
                            self.data[1, 1:-move_bid] = self.data[1, 1 + move_bid:]
                            self.data[1, -move_bid] = -self.outside_volume
                return True
