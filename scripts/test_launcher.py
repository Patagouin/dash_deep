import sys        
# We move backward to include Module (ex: Models.)
sys.path.append('..')   
import Models.Shares as sm
import time
import datetime

t=time.time()

## test ##
dateBegin = datetime.datetime.fromisoformat("2022-02-01 00:00:00+00:00")
dateLast = datetime.datetime.fromisoformat("2022-03-05 00:00:00+00:00")


#shM = sm.Shares(readOnlyThosetoUpdate=True)
#df = shM.getRowsByKeysValues(['symbol'],['CAP.PA'])
#shM.getDfDataRangeFromShare(shM.dfShares['CAP.PA'],dateBegin,dateLast)
#print(f"Time: {time.time()-t}s")

import yfinance as yf
from tabulate import tabulate
import pandas as pd





AAPL = yf.Ticker("CAP.PA")

print(AAPL.get_isin())
dateBegin = datetime.datetime.fromisoformat("2022-03-21 15:00:00")
dateLast = datetime.datetime.fromisoformat("2022-03-21 16:00:00")
#print(AAPL.history(start="2020-03-21 15:00:00", end="2020-03-21 16:00:00", interval="1m"))
pdArray = AAPL.history(start=dateBegin, end=dateLast, interval="1m")
print(AAPL.history(start=dateBegin, end=dateLast, interval="1m"))

print("AAPL.actions")
print(tabulate(AAPL.actions))
# show actions (dividends, splits)
print("AAPL.actions")
AAPL.actions

print("AAPL.dividends")
print(AAPL.dividends)

print("AAPL.splits")
print(tabulate(AAPL.splits))

print("AAPL.financials")
print(tabulate(AAPL.financials))

print("AAPL.quarterly_financials")
print(tabulate(AAPL.quarterly_financials))

print("AAPL.major_holders")
print(tabulate(AAPL.major_holders))

print("AAPL.institutional_holders")
print(tabulate(AAPL.institutional_holders))

print("AAPL.balance_sheet")
print(tabulate(AAPL.balance_sheet))

print("AAPL.cashflow")
print(tabulate(AAPL.cashflow))

print("AAPL.quarterly_cashflow")
print(tabulate(AAPL.quarterly_cashflow))

print("AAPL.earnings")
print(tabulate(AAPL.earnings))

print("AAPL.quarterly_cashflow")
print(tabulate(AAPL.quarterly_cashflow))

print("AAPL.sustainability")
print(tabulate(AAPL.sustainability))

print("AAPL.recommendations")
print(tabulate(AAPL.recommendations))

print("AAPL.calendar")
print(tabulate(AAPL.calendar))

print("AAPL.isin")
print(tabulate(AAPL.isin))

print("AAPL.options")
print(tabulate(AAPL.options))


#if __name__ == "__main__":

