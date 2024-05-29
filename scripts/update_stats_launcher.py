import sys        
# We move backward to include Module (ex: Models.)
sys.path.append('..')   
import Models.Shares as sm
import time
import Models.utils as ut

t=time.time()

## Stats ##

shM = sm.Shares(readOnlyThosetoUpdate=True)

shM.computeStatsForAllShares (shM.computeIsToUpdate)

print(f"Time: {time.time()-t}s")

#if __name__ == "__main__":
