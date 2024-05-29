import os
import sys      
  
# Obtient le chemin absolu du répertoire contenant le script en cours d'exécution
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construit le chemin absolu vers le répertoire 'dash_deep' en remontant d'un niveau
project_dir = os.path.dirname(script_dir)

# Ajoute le chemin du projet au sys.path
sys.path.append(project_dir)

import Models.Shares as sm

  
if __name__ == "__main__":
	shM = sm.Shares(readOnlyThosetoUpdate=True)
	if len(sys.argv) > 1 and sys.argv[1] == "-checkDuplicate":
		shM.updateAllSharesCotation(checkDuplicate=True)
	else:
		shM.updateAllSharesCotation(checkDuplicate=True)
