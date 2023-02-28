import numpy as np
import utils as ut
import Shares as sm

import matplotlib.pyplot as plt

def display(x, y):
	plt.scatter(x,y, marker = '+')
	plt.show()

def potentialDisplay(serie, k=5, printOutput=True):
	outArray, sumPercent = ut.getPotential(serie, k)
	fig, ax = plt.subplots(figsize=(12, 6))

	ax.plot([outArray[i][0] for i in range(len(outArray))], [outArray[i][1] for i in range(len(outArray))], 'ro')
	ax.plot([outArray[i][2] for i in range(len(outArray))], [outArray[i][3] for i in range(len(outArray))], 'ro')
	ax.plot(serie)
	fig.suptitle(f"Percent potential: {round(sumPercent,2)}")
	plt.show()

def displayQuots(shareObj, dfShare, dateBegin, dateLast):
	data=shareObj.getListDfDataFromDf(dfShare, dateBegin, dateLast)
	data.plot()