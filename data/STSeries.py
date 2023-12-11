import pandas as pd
import numpy as np
import sys

class STSeries(object):

    def __init__(self, config, start_ts, end_ts, volume, inflow, outflow,  T):
        self.config = config
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.volume = volume
        self.inflow = inflow
        self.outflow = outflow
        self.len_c = int(config['data']['closeness'])
        self.len_p = int(config['data']['period'])
        self.len_t = int(config['data']['trend'])
        self.n_interval = int(config['data']['T'])

        # step frequency
        self.freq = config['quantize']['freq']
        self.T = T

        if T == 48:
            self.ts = pd.date_range(start_ts, end_ts, freq='30min')
        elif T == 1440:
            self.ts = pd.date_range(start_ts, end_ts, freq='1min')

    def generate_instance(self, i, steps):

        x_ = []
        ts_ = []

        for step in steps:

            # generate ts step size behind i
            ip_ts = None
            if self.freq == '30min':
                ip_ts = self.ts[i] - step * pd.Timedelta(30, unit='min')

            # check if input ts is in dataset
            if ip_ts >= self.ts[0] and ip_ts <= self.ts[-1]:
                x_.append([self.volume[ip_ts.strftime('%Y-%m-%d %H:%M:00')],
                           self.inflow[ip_ts.strftime('%Y-%m-%d %H:%M:00')],
                           self.outflow[ip_ts.strftime('%Y-%m-%d %H:%M:00')]]) # (3, 17, 22)
                ts_.append(ip_ts.strftime('%Y-%m-%d %H:%M:00'))
#                ev_.append(self.event[ip_ts.strftime('%Y-%m-%d %H:%M:00')])
            else:
                print('Not in range: ', ip_ts)
                sys.exit()

        return x_, ts_

    def generate_series(self):

        # Create x, X_p, X_t, Y, ts_Y for each set
        c_step = np.arange(1, self.len_c + 1).tolist()
        p_step = [self.n_interval * j for j in np.arange(1, self.len_p + 1)]
        t_step = [7 * self.n_interval * j for j in np.arange(1, self.len_t + 1)]

        #print(c_step)
        #print(p_step)
        #print(t_step)

        #i = max(7 * self.n_interval * self.len_t, self.n_interval * self.len_p, self.len_c)
        i = max(7 * self.T * self.len_t, self.T * self.len_p, self.len_c)

        x_c = []
        x_p = []
        x_t = []
        y = []

        ts_c = []
        ts_p = []
        ts_t = []
        ts_y = []

        while (i < len(self.ts)):

            y.append([self.volume[self.ts[i].strftime('%Y-%m-%d %H:%M:00')],
                      self.inflow[self.ts[i].strftime('%Y-%m-%d %H:%M:00')],
                      self.outflow[self.ts[i].strftime('%Y-%m-%d %H:%M:00')]])
            ts_y.append(self.ts[i].strftime('%Y-%m-%d %H:%M:00'))

            x, t = self.generate_instance(i, c_step)                # (c, 3, 17, 22)
            x_c.append(np.stack(x))
            ts_c.append(t)
            x, t = self.generate_instance(i, p_step)                # (p, 3, 17, 22)
            x_p.append(np.stack(x))
            ts_p.append(t)
            x, t = self.generate_instance(i, t_step)                # (t, 3, 17, 22)
            x_t.append(np.stack(x))
            ts_t.append(t)
            i += 1

        x_c = np.stack(x_c)
        x_p = np.stack(x_p)
        x_t = np.stack(x_t)
        y = np.asarray(y)

        print(x_c.shape, x_p.shape, x_t.shape, y.shape)

        return x_c, x_p, x_t, y, ts_c, ts_p, ts_t, ts_y


