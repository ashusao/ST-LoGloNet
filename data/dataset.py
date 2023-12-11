from torch.utils import data

from .data_loader import load_data

class STDataset(data.Dataset):

    def __init__(self, config, mode, T):

        self.mode = mode

        if self.mode == 'train':
            self.X, self.meta, self.Y, _, _, _, mmn = load_data(config, T)
        elif self.mode == 'test':
            _, _, _, self.X, self.meta, self.Y, mmn = load_data(config, T)
        self.mmn = mmn

        assert len(self.X[0]) == len(self.Y)
        self.data_len = len(self.Y)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        X_c, X_p, X_t = self.X[0][index], self.X[1][index], self.X[2][index]
        TS_c, TS_p, TS_t = self.meta[0][index], self.meta[1][index], self.meta[2][index]
        Y = self.Y[index]
        TS_Y = self.meta[3][index]
        pois = self.meta[4]
        TS_Y_str = self.meta[5][index]

        #return x, X_p, X_t, Y, [TS_c, TS_p, TS_t, TS_Y, ev_c, ev_p, ev_t, ev_Y]
        return X_c, X_p, X_t, Y, [TS_c, TS_p, TS_t, TS_Y, pois, TS_Y_str]

    def denormalize(self, d, channel_id):
        '''

        :param d: data
        :param channel_id: 0: Vol, 1: inflow, 2:Outflow
        :return:
        '''
        # redefine for each channel (vol, inflow, outflow)
        org_shape = d.shape
        return self.mmn[channel_id].inverse_transform(d.reshape(-1, 1)).reshape(org_shape)