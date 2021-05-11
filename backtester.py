!pip3 install backtesting
#https://pypi.org/project/Backtesting/
import pandas as pd # Pandas Dataframes
from pandas_datareader import data # read data from yahoo and wikipedia
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import numpy as np

def MUL(a,b,color):
    return np.append(False,a*b*color)
class Candles(Strategy):
     def init(self):
         Close = self.data.Close
         Open = self.data.Open
         BarOn1 = (Close[0:-1]>Open[1:])
         BarOn2 = (Open[0:-1]<Close[1:])
         green = Open<Close
         red = Close<Open
         self.BarOn = self.I(MUL,BarOn1,BarOn2,red[0:-1])
         BarOff1 = (Open[0:-1]>Close[1:])
         BarOff2 = Close[0:-1]<Open[1:]
         self.BarOff =  self.I(MUL,BarOff1,BarOff2,green[0:-1])
     def next(self):
         if not self.position and self.BarOn:
            self.buy()
         elif self.BarOff:
            if self.position.size>0:
               self.position.close()

def RSI(array, n=14):
    gain = pd.Series(array).diff()
    loss = gain.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    rs = gain.ewm(span=n,min_periods=0,adjust=False).mean()/loss.abs().ewm(span=n,min_periods=0,adjust=False).mean()
    return 100 - 100 / (1 + rs)
class RSISystem(Strategy):
     def init(self):
         self.RSIData = self.I(RSI,self.data.Close,14)
     def next(self):
         if not self.position and self.RSIData<30:
            self.buy()
            #print("Buy ")
         elif self.RSIData>70:
            if self.position.size>0:
               self.position.close()

class SmaCross(Strategy):
    n1 = 20
    n2 = 50
    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)
    def next(self):
        if not self.position and crossover(self.sma1,self.sma2):
            self.buy()
        elif crossover(self.sma2,self.sma1):
            if self.position.size>0:
               self.position.close()

df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S&P_500_companies')[0]
tickers = df.Symbol.to_list()

output_df = pd.DataFrame(columns = ['Stock', 'BnH Return',
                                    'Candles','Candles Win Rate',
                                    'RSI','RSI Win Rate',
                                    'SmaCross','SmaCross Win Rate'])

for stock in tickers:
  try:
    df = data.DataReader(stock, 'yahoo', '2011-1-1', '2021-1-1')
    bt = Backtest(df, Candels,cash=1e7, commission=.000,exclusive_orders=False)
    output = bt.run()
    bt1 = Backtest(df, RSISystem,cash=1e7, commission=.000,exclusive_orders=False)
    output1 = bt1.run()
    bt2 = Backtest(df, SmaCross,cash=1e7, commission=.000,exclusive_orders=False)
    output2 = bt2.run()
  except:
    print("Something went wrong with:" + stock)
  else:
    output_df = output_df.append({'Stock' : stock,
                                 'BnH Return' : output['Buy & Hold Return [%]'],
                                 'Candles' : output['Return [%]'],
                                'Candles Win Rate':output['Win Rate [%]'],
                                'RSI' : output1['Return [%]'],
                                'RSI Win Rate':output1['Win Rate [%]'],
                                'SmaCross' : output2['Return [%]'],
                                'SmaCross Win Rate':output2['Win Rate [%]'],},
                                ignore_index = True)

df = data.DataReader('IVV', 'yahoo', '2011-1-1', '2021-1-1')
bt = Backtest(df, RSISystem,cash=1e7, commission=.000,exclusive_orders=False)
outputsnp = bt.run()
print('B&H: ',outputsnp['Buy & Hold Return [%]'],' RSI: ',outputsnp['Return [%]'])
sorted=output_df.sort_values(by=['BnH Return']).reset_index(drop=True)

import matplotlib.pyplot as plt
plt.figure(figsize=[10,8])
plt.semilogy(1+sorted['Candles']/100,'o')
plt.semilogy(1+sorted['RSI']/100,'s')
plt.semilogy(1+sorted['SmaCross']/100,'d')
plt.semilogy(1+sorted['BnH Return']/100,'x')
plt.axhline(y = (1+outputsnp['Buy & Hold Return [%]']/100), color = 'r', linestyle = '-')
plt.xlabel('S&P 500',fontsize=15)
plt.ylabel('Return on the $',fontsize=15)
plt.legend(['Engulfing Patterns','RSI','SMA','Buy and hold','B&H IVV'],fontsize=15)

print('All data')
print('Candles>B&H: ',format((sorted['BnH Return']<sorted['Candels']).sum()/sorted.shape[0], '.3f'))
print('RSI>B&H    : ',format((sorted['BnH Return']<sorted['RSI']).sum()/sorted.shape[0], '.3f'))
print('SMA>B&H    : ',format((sorted['BnH Return']<sorted['SmaCross']).sum()/sorted.shape[0], '.3f'))
print('Top 250')
print('Candles>B&H: ',format((sorted['BnH Return'].iloc[250:]<sorted['Candels'].iloc[250:]).sum()/sorted.iloc[250:].shape[0], '.3f'))
print('RSI>B&H    : ',format((sorted['BnH Return'].iloc[250:]<sorted['RSI'].iloc[250:]).sum()/sorted.iloc[250:].shape[0], '.3f'))
print('SMA>B&H    : ',format((sorted['BnH Return'].iloc[250:]<sorted['SmaCross'].iloc[250:]).sum()/sorted.iloc[250:].shape[0], '.3f'))
print('Bottom 100')
print('Candles>B&H: ',format((sorted['BnH Return'].iloc[:100]<sorted['Candels'].iloc[:100]).sum()/sorted.iloc[:100].shape[0], '.3f'))
print('RSI>B&H    : ',format((sorted['BnH Return'].iloc[:100]<sorted['RSI'].iloc[:100]).sum()/sorted.iloc[:100].shape[0], '.3f'))
print('SMA>B&H    : ',format((sorted['BnH Return'].iloc[:100]<sorted['SmaCross'].iloc[:100]).sum()/sorted.iloc[:100].shape[0], '.3f'))

print('Candles: ',format((sorted['Candels']>1).sum()/sorted.shape[0], '.3f'))
print('RSI    : ',format((sorted['RSI']>1).sum()/sorted.shape[0], '.3f'))
print('SMA    : ',format((sorted['SmaCross']>1).sum()/sorted.shape[0], '.3f'))

sorted.to_csv('TA.csv')