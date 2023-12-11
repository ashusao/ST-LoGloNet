import configparser

from train import train

if __name__=='__main__':

    # read the configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')

    train(config)
