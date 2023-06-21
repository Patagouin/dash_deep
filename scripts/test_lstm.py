import sys        
# We move backward to include Module (ex: Models.)
sys.path.append('..')   
import Models.Shares as sm
import Models.utils as ut
#import lstm as ls

shM = sm.Shares(readOnlyThosetoUpdate=True)
shM.dfShares = shM.getRowsDfByKeysValues("market","us_market")
shM.updateAllSharesModels()


exit (1)