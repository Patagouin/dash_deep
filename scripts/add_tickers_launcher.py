## The aim of this script is only to retrieve the info (via yahoo finance api) of the list 
## Of ticker in csv file using deep_market library
import sys        
# We move backward to include Module (ex: Models.)
sys.path.append('..')   

import Models.Shares as sm
import Models.Visualizer as vi
import Models.utils as ut
import Models.SqlCom as sq

# Name of the file (in the same folder as the script if relative path
# This file contains new (or not) ticker to filter
tickerNameFile = "../data/1er_filtrage_yahoo_info_dispo.csv"

shM = sm.Shares(readOnlyThosetoUpdate=False)
shM.addTickersInfoFromFile(tickerNameFile)



