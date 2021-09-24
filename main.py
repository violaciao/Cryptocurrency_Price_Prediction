from forcast import CryptoForcaster

if __name__ == '__main__':
    crypto = CryptoForcaster('ETH-USD', '2016-01-01')
    crypto.plot_open_prices()
    crypto.plot_forcast()
    crypto.plot_components()