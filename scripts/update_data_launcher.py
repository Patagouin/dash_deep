import sys        
# We move backward to include Module (ex: Models.)
sys.path.append('..')   
import Models.Shares as sm

  
if __name__ == "__main__":
	shM = sm.Shares(readOnlyThosetoUpdate=True)
	if len(sys.argv) > 1 and sys.argv[1] == "-checkDuplicate":
		shM.updateAllSharesCotation(checkDuplicate=True)
	else:
		shM.updateAllSharesCotation(checkDuplicate=True)
