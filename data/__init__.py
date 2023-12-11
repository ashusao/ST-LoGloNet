import numpy as np
import pandas as pd
import pickle
import math
import os
import collections

from sklearn.preprocessing import MinMaxScaler


def create_f_name(config, T):
    f_vol = os.path.join(config['data']['dir'],
                         config['data']['city'] + '_vol_' + str(config['grid']['size']) +
                         '_' + str(str(T) + '.pkl'))

    f_inflow = os.path.join(config['data']['dir'],
                         config['data']['city'] + '_inflow_' + str(config['grid']['size']) +
                         '_' + str(str(T) + '.pkl'))

    f_outflow = os.path.join(config['data']['dir'],
                         config['data']['city'] + '_outflow_' + str(config['grid']['size']) +
                         '_' + str(str(T) + '.pkl'))

    f_event = config['data']['dir'] + 'events_by_impact.pkl'

    return f_vol, f_inflow, f_outflow, f_event


def load_pickle(f_name):
    with open(f_name, 'rb') as f:
        data = pickle.load(f)
    f.close()
    return data


def read_data(config, T):
    f_vol, f_inflow, f_outflow, f_event = create_f_name(config, T)
    f_pre_vol, f_pre_inflow, f_pre_outflow, _ = create_f_name(config, T)

    volume = collections.OrderedDict(sorted(load_pickle(f_vol).items()))
    inflow = collections.OrderedDict(sorted(load_pickle(f_inflow).items()))
    outflow = collections.OrderedDict(sorted(load_pickle(f_outflow).items()))

    pre_volume = collections.OrderedDict(sorted(load_pickle(f_pre_vol).items()))
    pre_inflow = collections.OrderedDict(sorted(load_pickle(f_pre_inflow).items()))
    pre_outflow = collections.OrderedDict(sorted(load_pickle(f_pre_outflow).items()))

    #event = collections.OrderedDict(sorted(load_pickle(f_event).items()))
    pois = load_pickle(config['data']['dir'] + '/hann_poi_1km.pkl')

    return volume, inflow, outflow, pre_volume, pre_inflow, pre_outflow, pois


def center_and_normalize(data, len_test):
    keys = data.keys()
    stacked_data = np.stack(list(data.values()))
    print(stacked_data.shape)

    # extract training data and compute mean
    data_train = stacked_data[:-len_test]
    data_mean = np.mean(data_train, axis=0)
    data_train = data_train - data_mean
    print(data_train.shape, data_train.reshape(-1, 1).shape)

    # fit minmax scaler
    mmn = MinMaxScaler(feature_range=(-1, 1))
    mmn.fit(data_train.reshape(-1, 1))
    print("min:", mmn.data_min_, "max:", mmn.data_max_)

    # subtract mean from complete data
    stacked_data = stacked_data - data_mean
    org_shape = stacked_data.shape

    # normalize
    normalized_data = mmn.transform(stacked_data.reshape(-1, 1)).reshape(org_shape)
    normalized_data = collections.OrderedDict(zip(keys, list(normalized_data)))

    return mmn, normalized_data, data_mean



def normalize_data(data, len_test):

    keys = data.keys()
    stacked_data = np.stack(list(data.values()))
    print(stacked_data.shape)

    data_train = stacked_data[:-len_test]
    print(data_train.shape, data_train.reshape(-1, 1).shape)

    mmn = MinMaxScaler(feature_range=(-1, 1))
    mmn.fit(data_train.reshape(-1, 1))
    org_shape = stacked_data.shape
    print("min:", mmn.data_min_, "max:", mmn.data_max_)

    normalized_data = mmn.transform(stacked_data.reshape(-1, 1)).reshape(org_shape)
    normalized_data = collections.OrderedDict(zip(keys, list(normalized_data)))

    return mmn, normalized_data


def cyclic_encode_time_of_day(time):
    minutes_since_midnight = time.hour * 60 + time.minute
    total_minutes_in_a_day = 24 * 60
    angle = 2 * math.pi * minutes_since_midnight / total_minutes_in_a_day
    return math.sin(angle), math.cos(angle)


def timestamp2vec(timestamps):
    # tm_wday range [0, 6], Monday is 0
    vec = []

    for timestamp in timestamps:
        timestamp = pd.Timestamp(timestamp)

        dow = timestamp.day_of_week
        v = [0 for _ in range(7)] # day of week
        v[dow] = 1
        if dow >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday

        # Encode time from timestamp
        time = timestamp.time()
        time_encode_vector = cyclic_encode_time_of_day(time)

        v += time_encode_vector  # dim: 7 (dow) + 1(wd/we) + 2 (time)

        vec.append(v)

    return np.asarray(vec)
