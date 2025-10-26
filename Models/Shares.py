import numpy as np
import pandas as pd
import csv
import datetime
import re
import logging

import Models.SqlCom as sq
import Models.utils as ut
import Models.prediction_utils as pred_ut

from dotenv import load_dotenv  # Add this to load environment variables
import os

# Charger les variables d'environnement à partir du fichier .env
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)
print(f"Loaded environment variables from: {env_path}")

class Shares:

    def __init__(self, readOnlyThosetoUpdate=False):
        self.dfShares = pd.DataFrame()
        self.__sqlObj = sq.SqlCom(
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            sharesObj=self
        )
        self.readSharesInDB(readOnlyThosetoUpdate)

    def readSharesInDB(self, readOnlyThosetoUpdate=False):
        self.dfShares = self.__sqlObj.readSharesInfos(readOnlyThosetoUpdate)

    def addTickersInfoFromFile(self, listSharesFileName):
        listSharesToAdd = self.readListSharesFromFile(listSharesFileName, withMarketAtEndTicker=False)
        nbSharesTryUpdated = 0
        nbTotalShares =  len(listSharesToAdd)
        for tickerName in listSharesToAdd:
            ut.logOperation(f"Infos: try add/update {tickerName}")
            isOk = self.modifyShareInfos(tickerName)
            nbSharesTryUpdated += 1
            ut.logOperation(f"Nb shares info trying to be added/updated from file: {nbSharesTryUpdated}/{nbTotalShares}")
            ut.logOperation(f"{ut.logMemoryGetAll()}")
        ut.logOperation("Finish: All shares info from file has been treated")
        ut.logOperation(f"Summary: {ut.logMemoryGetAll()}")
        

    def updateAllSharesInfos(self):
        #start = False
        nbTotalShares =  self.dfShares.shape[1] #Column
        nbSharesTryUpdated = 0
        for curShare in self.dfShares.itertuples():
                #if curShare.tickerName == 'MRK.PA':
                #    start = True
                #if start:
            self.modifyShareInfos(curShare.tickerName)
            nbSharesTryUpdated += 1
            ut.logOperation(f"Nb shares info trying to be updated: {nbSharesTryUpdated}/{nbTotalShares}")
            ut.logOperation(f"{ut.logMemoryGetAll()}")

        ut.logOperation("Finish: All shares info has been treated")
        ut.logOperation(f"Summary: {ut.logMemoryGetAll()}")
        
    def modifyShareInfos(self, shareName):
        # Read before
        dataBefore = self.__sqlObj.readInfosShare(shareName)
        isOkAdded= 0
        isOkUpdated = 0
        if dataBefore["values"] == []:
            ut.logOperation(f"Infos: try add {shareName}")
            isOkAdded = self.__sqlObj.addInfosShare(shareName)
            if isOkAdded == 0:
                ut.logOperation(f"Share: {shareName} not added (because ticker doesn't exist ?)")
                ut.logMemoryIncrement("NoAddedInfo")
            elif isOkAdded == 1:
                ut.logOperation(f"Share: {shareName} added")
                ut.logMemoryIncrement("AddedInfo")

        else:
            ut.logOperation(f"Infos: try update {shareName}")
            isOkUpdated = self.__sqlObj.updateInfoShare(shareName)

            # Read after
            dataAfter = self.__sqlObj.readInfosShare(shareName)
            string = ut.compare2linesDB(dataBefore,dataAfter)
            if (string != ""):
                ut.logOperation(f"Share: {shareName} infos updated")
                ut.logOperation(f"\tDetails:\n{string}")
                ut.logMemoryIncrement("NbDiffsByShareUpdateInfo")
                ut.logMemoryIncrement("UpdatedInfo")
            else:
                ut.logOperation(f"Share: {shareName} infos no update required")
                ut.logMemoryIncrement("NoUpdateInfosRequired")


    def getAllShares(self):
         return self.dfShares
   
    def updateAllSharesCotation(self, checkDuplicate=False):
        #start = False
        nbTotalShares =  self.dfShares.shape[0]
        nbSharesUpdated = 0
        # cpt=0
        for curShare in self.dfShares.itertuples():
            # if cpt < 825:
            #     cpt+=1
            #     continue
            
            #if curShare.symbol == "AOS":
            #    start = True
            #else:
            #    nbSharesUpdated += 1
            #if start:
            print(curShare.idShare)
            self.updateShareCotations(curShare, checkDuplicate)
            nbSharesUpdated += 1
            ut.logOperation(f"Nb shares cotation updated: {nbSharesUpdated}/{nbTotalShares}")

        ut.logOperation("Success: All shares cotation updated")

    def train_share_model(self, shareObj, data_info, hps, callbacks=None):
        logging.info(f"[TRAIN|Shares] Start train_share_model for {shareObj.symbol}")
        # 1. Récupérer et nettoyer les données
        logging.info(f"[TRAIN|Shares] get_and_clean_data: features={data_info['features']}")
        df = pred_ut.get_and_clean_data(self, shareObj, data_info['features'])
        logging.info(f"[TRAIN|Shares] get_and_clean_data: shape={getattr(df, 'shape', None)}")
        
        # 2. Créer les ensembles d'entraînement et de test
        logging.info("[TRAIN|Shares] create_train_test: start")
        trainX, trainY, testX, testY = pred_ut.create_train_test(df, data_info)
        try:
            logging.info(f"[TRAIN|Shares] create_train_test: trainX={len(trainX)} trainY={len(trainY)} testX={len(testX)} testY={len(testY)}")
        except Exception:
            logging.info("[TRAIN|Shares] create_train_test: shapes unavailable")

        # Ne pas recalculer nb_days_total à partir du nombre de séquences (unités différentes)
        # Se fier à la valeur calculée dans split_dataset_train_test
        logging.info(f"[TRAIN|Shares] nb_days_total={data_info.get('nb_days_total')} nb_quots_by_day={data_info.get('nb_quots_by_day')}")

        # 3. Entraîner et sélectionner le meilleur modèle
        logging.info(f"[TRAIN|Shares] train_and_select_best_model: start max_trials={hps.get('max_trials')} epochs={hps.get('epochs')}")
        best_model, best_hps, tuner = pred_ut.train_and_select_best_model(
            data_info, hps, trainX, trainY, testX, testY, callbacks=callbacks
        )
        logging.info("[TRAIN|Shares] train_and_select_best_model: done")

        # 4. Sauvegarde déléguée à l'appelant (après l'entraînement final)
        return best_model, best_hps, tuner

    def save_model(self, shareObj, model, model_name, trainScore=None, testScore=None):
        try:
            logging.info("[TRAIN|Shares] save model: start")
            model.save("model.keras")
            with open("model.keras", "rb") as file:
                model_binary = file.read()
            logging.info(f"[TRAIN|Shares] saveModel to DB: name={model_name} train={trainScore} test={testScore}")
            self.__sqlObj.saveModel(shareObj, model_name, model_binary, trainScore, testScore)
            logging.info("[TRAIN|Shares] save model: done")
        except Exception as e:
            logging.exception("[TRAIN|Shares] Failed to save model")
            raise

    def list_models_for_symbol(self, symbol: str):
        df = self.getRowsDfByKeysValues('symbol', symbol)
        if df.empty:
            return []
        share = df.iloc[0]
        return self.__sqlObj.listModelsForShare(share)

    def load_model_from_db(self, model_id):
        """Charge un modèle Keras depuis la BDD (par id) et retourne l'objet modèle Keras."""
        import tempfile
        import tensorflow as tf
        model_bin = self.__sqlObj.getModelBinary(model_id)
        if not model_bin:
            raise ValueError(f"Modèle introuvable en BDD pour id={model_id}")
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=True) as tmp:
            tmp.write(model_bin)
            tmp.flush()
            model = tf.keras.models.load_model(tmp.name, compile=False)
        return model

    def updateAllSharesModels(self, df=None):
        if df is None:
            workingDf = self.dfShares
        else:
            workingDf = df.copy()

        # Configuration des hyperparamètres et des données
        hps = {
            'layers': [1, 2],
            'nb_units': [50, 100],
            'learning_rate': [1e-2, 1e-3],
            'loss': 'mean_squared_error',
            'max_trials': 10,
            'executions_per_trial': 1,
            'directory': 'tuner_results',
            'project_name': 'stock_prediction',
            'patience': 5,
            'epochs': 20
        }
        
        data_info = {
            'look_back_x': 60,
            'stride_x': 1,
            'nb_y': 5,
            'features': ['openPrice', 'volume'],
            'return_type': 'yield',
            'nb_days_to_take_dataset': 'max',
            'percent_train_test': 80
        }

        for curShare in workingDf.itertuples():
            print(f"Training model for {curShare.symbol}...")
            # Mettre à jour data_info avec l'objet share actuel
            data_info['shareObj'] = curShare
            
            # Définir un nom de projet unique pour le tuner
            hps['project_name'] = f'pred_{curShare.symbol}'

            try:
                self.train_share_model(curShare, data_info, hps)
                print(f"Successfully trained and saved model for {curShare.symbol}")
            except Exception as e:
                print(f"Could not train model for {curShare.symbol}. Error: {e}")


    def updateShareCotations(self, shareObj, checkDuplicate=False):
        if not checkDuplicate:
            lastQuotDate = shareObj.lastRecord # self.__sqlObj.getLastDateQuot(shareObj)
        else:
            lastQuotDate = 'max'
        df = self.__sqlObj.downloadDataInDB(shareObj, dateBegin=lastQuotDate, dateEnd='now')
        df = df[1:]
        self.__sqlObj.saveDataInDB(df, shareObj, checkDuplicate)
        self.__sqlObj.computeLastRecord(shareObj)

        ut.logOperation(f"Shares: {shareObj.symbol} updated")

    def updateSharesFromDataFrame(self, dfShare):
        for share in dfShare.itertuples():
            self.updateShare(share)


    def getRowsDfByKeysValues(self, keys, values, op='&', comp_op='==', df=None):
        '''
        Get rows of a Pandas DataFrame that match given key-value pairs.

        @param keys: list of column names in the DataFrame to search on.
        @type keys: list or str
        @param values: list of values to match for each key, in the same order as keys.
        @type values: list or str
        @param op: type of logical operation to use to combine the search criteria. Can be 'union' or '&'. Default is '&'.
        @type op: str
        @param comp_op: comparison operator to use for each search criterion. Default is '=='.
        @type comp_op: str
        @param df: optional Pandas DataFrame to search on. If not provided, uses self.dfShares.
        @type df: pandas.DataFrame
        @return: a Pandas DataFrame containing the matching rows.
        @rtype: pandas.DataFrame
        '''
        if type(keys) is not list:
            keys=[keys]
        if type(values) is not list:
            values=[values]
        if (values != [] and keys != []):
            for i in range(len(values)):
                values[i] = str(values[i])
                if values[i].replace('.', '', 1).isdigit() == False:
                    values[i]= f'\'{values[i]}\''
            if type(df) == pd.DataFrame:
                workingDf=df
            else:
                workingDf=self.dfShares
         
            expr=''
            for key, value in zip(keys,values):
                expr += f''' (workingDf.{key} {comp_op} {value}) '''
                expr += op
            expr = expr[:-1]

            return workingDf[eval(expr)].drop_duplicates()
        return pd.DataFrame()

    def getListDfDataFromDfShares(self, dfShare, dateBegin=None, dateEnd=None):
        """Récupère les données pour plusieurs actions en une seule requête SQL."""
        if dfShare.empty:
            print("getListDfDataFromDfShares - dfShare is empty")
            return []
        
        print(f"getListDfDataFromDfShares - dfShare shape: {dfShare.shape}")
        print(f"getListDfDataFromDfShares - Date range: {dateBegin} to {dateEnd}")
        if dateBegin is not None and dateEnd is not None:
            print(f"getListDfDataFromDfShares - Date range length: {(dateEnd - dateBegin).days} days")
        
        # Si toutes les actions sont demandées en même temps, faire une seule requête
        if len(dfShare) > 1:
            print(f"getListDfDataFromDfShares - Using multiple shares query")
            # Construire la liste des IDs d'actions
            share_ids = dfShare['idShare'].tolist()
            # Récupérer toutes les données en une seule requête
            all_data = self.__sqlObj.getQuotsMultiple(share_ids, dateBegin, dateEnd)
            
            # Séparer les données par action
            listDfData = []
            for share in dfShare.itertuples():
                print(f"getListDfDataFromDfShares - Processing share: {share.symbol}")
                if all_data.empty:
                    print(f"getListDfDataFromDfShares - No data for any share")
                    listDfData.append(pd.DataFrame())
                    continue
                    
                share_data = all_data[all_data['idShare'] == share.idShare].copy()
                print(f"getListDfDataFromDfShares - Share {share.symbol} data shape: {share_data.shape}")
                
                if not share_data.empty:
                    # Supprimer la colonne idShare car elle n'est plus nécessaire
                    share_data = share_data.drop('idShare', axis=1)
                listDfData.append(share_data)
            
            return listDfData
        else:
            print(f"getListDfDataFromDfShares - Using single share query")
            # Pour une seule action, utiliser la méthode existante
            data = self.getDfDataRangeFromShare(dfShare.iloc[0], dateBegin, dateEnd)
            print(f"getListDfDataFromDfShares - Single share data shape: {data.shape}")
            return [data]

    def getDfDataFromSerie(self, serieShare, dateBegin, dateEnd):
        return self.getDfDataRangeFromShare(serieShare, dateBegin, dateEnd)

    def getDfDataRangeFromShare(self, share, dateBegin, dateEnd):
        return self.__sqlObj.getQuots(share, dateBegin, dateEnd)

    def addShare(self, ticker):
        self.dfShares[ticker] = ticker

    def setAllShares(self, dataframe):
        self.dfShares = dataframe

    # Return the list of ticker in array
    def readListSharesFromFile(self, listSharesFileName, withMarketAtEndTicker=True):
        start = 0
        listSharesInFile = []
        with open(listSharesFileName, 'r') as file:
            for line in file:
                    ticker = line.replace('\n','')
                    ticker = ticker.replace(' ','')
                    if ticker == 'JBHT':
                        start = 1
                    if ticker != '' and start==1:
                        listSharesInFile.append(ticker)
                        if (not withMarketAtEndTicker) and (not '.' in ticker):
                            listSharesInFile.append(ticker + ".PA") # Test with market name at end of ticker

        return listSharesInFile

    def getListShareWithSpecificAttributs(self, sector='', industry='', volumeMin='', volumeMax=''):
        return self.__sqlObj.getShareNameWithSector(sector, industry, volumeMin, volumeMax)
        
    def checkAndUpdateVolume(self):
        
        listDateAndTickerName = self.__sqlObj.checkVolumeExcess()
        nbExcess = len(listDateAndTickerName)
        self.__sqlObj.updateVolumeExcess(listDateAndTickerName)
        listDateAndTickerName = self.__sqlObj.checkVolumeExcess()
        nbExcess2 = len(listDateAndTickerName)
        return nbExcess2-nbExcess
#        newVolumeByDateAndTickerName = ut.getNewVolumeByDateAndTickerName(listDateAndTickerName)
#        listDateAndTickerName = self.__sqlObj.updateVolumeExcess(newVolumeByDateAndTickerName)

    def getShareBySymbol(self, tickerName):
        data = self.getRowsDfByKeysValues("symbol", tickerName)
        return data

    def computeStatsForAllShares(self, *methods, df=None):
        if (df == None):
            workingDf = self.dfShares
        else:
            workingDf = df
        nbTotalShares =  workingDf.shape[0]
        for method in methods:
            ut.logOperation(f"{method.__name__} for all shares")
            nbSharesUpdated = 0
            for curShare in workingDf.itertuples():
                method(curShare)
                nbSharesUpdated += 1
                ut.logOperation(f"{curShare.symbol} stats updated: {nbSharesUpdated}/{nbTotalShares}")

        ut.logOperation("Success: All shares stats updated")


    def computeNbRecordsAndAvgByDay(self, shareObj):
        self.__sqlObj.computeNbRecordsAndAvgByDay(shareObj)

    def computeFirstRecord(self, shareObj):
        self.__sqlObj.computeFirstRecord(shareObj)

    def computeLastRecord(self, shareObj):
        self.__sqlObj.computeLastRecord(shareObj)

    def computePotential(self, shareObj):
        self.__sqlObj.computePotential(shareObj)

    def computePotentialForDates(self, shareObj, dates, potential_levels=None):
        self.__sqlObj.computePotentialForDates(shareObj, dates, potential_levels)

    def readPotentialsPercentTotal(self, shareObj, start_date, end_date, potential_levels=None):
        return self.__sqlObj.readPotentialsPercentTotal(shareObj, start_date, end_date, potential_levels)

    def computeIsToUpdate(self, shareObj):
        nbRecordsMinByDay = 100
        self.__sqlObj.discardUpdate(shareObj, nbRecordsMinByDay)

    def computeOpenCloseMarketTime(self, shareObj):
        nbRecordsMinByDay = 100
        self.__sqlObj.compute_general_market_open_close_time(shareObj)

    def get_cotations_data_df(self, shareObj, start_date, end_date):
        return self.__sqlObj.get_cotations_data_df(shareObj, start_date, end_date)
