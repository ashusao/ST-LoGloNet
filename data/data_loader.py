import numpy as np

from . import read_data, normalize_data, timestamp2vec
from .STSeries import STSeries


def load_data(config, T):

    print('loading data ...')
    volume, inflow, outflow,  _, _, _, pois = read_data(config, T)

    len_test = int(config['test']['n_days']) * T

    print('Volume:')
    mmn_volume, norm_volume = normalize_data(volume, len_test)
    print('Inflow:')
    mmn_inflow, norm_inflow = normalize_data(inflow, len_test)
    print('Outflow:')
    mmn_outflow, norm_outflow = normalize_data(outflow, len_test)
    print('POI:')
    print(pois.shape)

    mmn = [mmn_volume, mmn_inflow, mmn_outflow]

    XC = []
    XP = []
    XT = []
    Y = []

    timestamps_XC = []
    timestamps_XP = []
    timestamps_XT = []
    timestamps_Y = []

    print('Creating ST Series ...')
    # generate sets of avail data
    start_ts, end_ts = config['data']['avail_data'].split('#')
    st_series = STSeries(config=config, start_ts=start_ts, end_ts=end_ts,
                         volume=norm_volume, inflow=norm_inflow, outflow=norm_outflow, T=T)
    x_c, x_p, x_t, y, ts_c, ts_p, ts_t, ts_y = st_series.generate_series()

    XC.append(x_c)
    XP.append(x_p)
    XT.append(x_t)
    Y.append(y)
    timestamps_XC += ts_c
    timestamps_XP += ts_p
    timestamps_XT += ts_t
    timestamps_Y += ts_y

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)

    print('timestamp array example:')
    print(timestamps_XC[1], timestamps_XP[1], timestamps_XT[1], timestamps_Y[1])

    print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
    print("XC ts: ", len(timestamps_XC), "XP ts: ", len(timestamps_XP), "XT ts: ", len(timestamps_XT),
          "Y ts:", len(timestamps_Y))

    #if config.getboolean('data', 'ohe_ts'):
    timestamps_XC = [timestamp2vec(ts) for ts in timestamps_XC]
    timestamps_XP = [timestamp2vec(ts) for ts in timestamps_XP]
    timestamps_XT = [timestamp2vec(ts) for ts in timestamps_XT]
    timestamps_Y_str = timestamps_Y
    timestamps_Y = timestamp2vec(timestamps_Y)

    XC_train, XP_train, XT_train, Y_train = XC[:-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    TS_c_train, TS_p_train, TS_t_train, TS_Y_train, TS_Y_train_str = \
        timestamps_XC[:-len_test], timestamps_XP[:-len_test], timestamps_XT[:-len_test], timestamps_Y[:-len_test], \
        timestamps_Y_str[:-len_test]

    XC_test, XP_test, XT_test, Y_test = XC[-len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    TS_c_test, TS_p_test, TS_t_test, TS_Y_test, TS_Y_test_str = \
        timestamps_XC[-len_test:], timestamps_XP[-len_test:], timestamps_XT[-len_test:], timestamps_Y[-len_test:], \
        timestamps_Y_str[-len_test:]

    X_train = [XC_train, XP_train, XT_train]
    X_test = [XC_test, XP_test, XT_test]
    meta_train = [TS_c_train, TS_p_train, TS_t_train, TS_Y_train, pois, TS_Y_train_str]
    meta_test = [TS_c_test, TS_p_test, TS_t_test, TS_Y_test, pois, TS_Y_test_str]

    print('X_train shape:')
    for _X in X_train:
        print(_X.shape, )

    print('X_test shape:')
    for _X in X_test:
        print(_X.shape, )

    return X_train, meta_train, Y_train, X_test, meta_test, Y_test, mmn

