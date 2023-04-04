import numpy as np
import pandas as pd
import csv
import datetime
import re

import Models.SqlCom as sq
import Models.utils as ut



class Shares:

    def __init__(self, readOnlyThosetoUpdate=False):
        self.listShares = pd.DataFrame()
        self.__sqlObj = sq.SqlCom("postgres", "Rapide23$", "127.0.0.1", "5432", "stocksprices", self)
        self.readSharesInDB(readOnlyThosetoUpdate)

    def readSharesInDB(self, readOnlyThosetoUpdate=False):
        self.__sqlObj.readSharesInfos(readOnlyThosetoUpdate)

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
        nbTotalShares =  self.listShares.shape[1] #Column
        nbSharesTryUpdated = 0
        for curShare in self.listShares.itertuples():
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
         return self.listShares
   
    def updateAllSharesCotation(self, checkDuplicate=False):
        #start = False
        nbTotalShares =  self.listShares.shape[0]
        nbSharesUpdated = 0
        for curShare in self.listShares.itertuples():
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

    def updateShareCotations(self, shareObj, checkDuplicate=False):
        if not checkDuplicate:
            lastQuotDate = self.__sqlObj.getLastDateQuot(shareObj)
        else:
            lastQuotDate = 'max'
        df = self.__sqlObj.downloadDataInDB(shareObj, dateBegin=lastQuotDate, dateEnd='now')
        df = df[1:]
        self.__sqlObj.saveDataInDB(df, shareObj, checkDuplicate)
        
        ut.logOperation(f"Shares: {shareObj.symbol} updated")

    def updateSharesFromDataFrame(self, dfShare):
        for share in dfShare.itertuples():
            self.updateShare(share)


    def getRowsByKeysValues(self, keys, values, op='&', comp_op='==', df=None):
        ''' type=['union'|'&'] if union each catched properties are added to list otherwise list is created when all properties are satisfied '''
        ''' If df != pd.DataFrame instead of self.listShare df is taken as dataframe input '''
        if type(keys) is not list:
            keys=[keys]
        if type(values) is not list:
            values=[values]
        for i in range(len(values)):
            values[i] = str(values[i])
            if values[i].replace('.', '', 1).isdigit() == False:
                values[i]= f'\'{values[i]}\''
        if type(df) == pd.DataFrame:
            workingDf=df
        else:
            workingDf=self.listShares
         
        expr=''
        for key, value in zip(keys,values):
            expr += f''' (workingDf.{key} {comp_op} {value}) '''
            expr += op
        expr = expr[:-1]

        return workingDf[eval(expr)].drop_duplicates()

    def getListDfDataFromDf(self, dfShare, dateBegin, dateEnd):
        listDfData = []
        for row in dfShare.itertuples():
            listDfData.append(self.getDfDataRangeFromShare(row, dateBegin, dateEnd))
        return listDfData

    def getDfDataFromSerie(self, serieShare, dateBegin, dateEnd):
        return self.getDfDataRangeFromShare(serieShare, dateBegin, dateEnd)

    def getDfDataRangeFromShare(self, share, dateBegin, dateEnd):
        return self.__sqlObj.getQuots(share, dateBegin, dateEnd)

    def addShare(self, ticker):
        self.listShares[ticker] = ticker

    def setAllShares(self, dataframe):
        self.listShares = dataframe

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
        data = self.getRowsByKeysValues("symbol", tickerName)
        return data

    def computeStatsForAllShares(self, *methods):
                #start = False
        nbTotalShares =  self.listShares.shape[0]
        for method in methods:
            ut.logOperation(f"{method.__name__} for all shares")
            nbSharesUpdated = 0
            for curShare in self.listShares.itertuples():
                method(curShare)
                nbSharesUpdated += 1
                ut.logOperation(f"{curShare.symbol} stats updated: {nbSharesUpdated}/{nbTotalShares}")

        ut.logOperation("Success: All shares stats updated")

    def computeAllStats(self, shareObj):
        self.__sqlObj.computeNbRecordsAndAvgByDayAndFirstLastRecord(shareObj)
        nbRecordsMinByDay = 100
        self.__sqlObj.discardUpdate(shareObj, nbRecordsMinByDay)

    def computeNbRecordsAndAvgByDayAndFirstLastRecord(self, shareObj):
        self.__sqlObj.computeNbRecordsAndAvgByDayAndFirstLastRecord(shareObj)


    def computeLastRecord(self, shareObj):
        self.__sqlObj.computeLastRecord(shareObj)

    def computeAvgMaxRangeByDay(self, shareObj):
        self.__sqlObj.computeAvgMaxRangeByDay(shareObj)

    def computePotential(self, shareObj):
        self.__sqlObj.computePotential(shareObj)

    def computeIsToUpdate(self, shareObj):
        nbRecordsMinByDay = 100
        self.__sqlObj.discardUpdate(shareObj, nbRecordsMinByDay)
