import psycopg2

import Models.utils as ut
import datetime
import pandas as pd
import numpy as np
import pytz
import re
from pandas.tseries.offsets import BDay
#chatGPT
#from sqlalchemy import create_engine

UPDATE_PROCESS_ENUM = 0
CREATE_PROCESS_ENUM = 1
MAX_DIGIT_PRECISION = 5
class SqlCom:

    def __init__(self, user, password, host, port, database, sharesObj):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database

        self.connection = None
        self.cursor = None

        self.sharesObj = sharesObj

        self.connect()

    def __del__(self):
        self.disconnect()

    def connect(self):
        # From chatGPT

        ## Créer une chaîne de connexion de base de données
        #db_uri = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

        ## Créer une instance de connexion SQLAlchemy
        #engine = create_engine(db_uri)

        ## Utiliser la connexion pour exécuter une requête SQL et récupérer les résultats dans un DataFrame pandas
        #df = pd.read_sql('SELECT * FROM my_table', engine)

        if(self.connection == None or self.connection.closed):
            self.connection = psycopg2.connect(user=self.user, password=self.password, host=self.host, port=self.port, database=self.database)
            self.cursor = self.connection.cursor()

    def disconnect(self):
        if(self.connection != None and not self.connection.closed):
            self.connection.close()


    def saveDataInDB(self, df, share, checkDuplicate=False, nbTry=1):
        ''' Save data (multiple lines) from a share in DB '''
        nbAlreadySavedRow = 0
        tmpRow = None
        insert_query = ""
        try:
            subQuery = f''' SELECT  "sharesInfos"."idShare" FROM "sharesInfos" WHERE "sharesInfos"."symbol"='{share.symbol}' '''
            if checkDuplicate == False:
                lastTime = None # Sometime same time with same share occurs
                for row in df.itertuples():
                    tmpRow = row
                    if lastTime != row.Index:
                        if row.Open == row.Open or row.High == row.High or row.Low == row.Low or row.Close == row.Close: # Detect if any nan value du to yahoo which give bad data
                            insert_query += ( f' INSERT INTO public."sharesPricesQuots"('
                                                    f' "time", "openPrice", "highPrice", "lowPrice", "closePrice", "volume", "dividend", "idShare") '
                                                    f' VALUES (\'{row.Index}\', \'{row.Open}\', \'{row.High}\', \'{row.Low}\', \'{row.Close}\', \'{row.Volume}\', \'{row.Dividends}\', ({subQuery})); ' 
                                                    )

                    lastTime = row.Index
                # Execute all requests at once
                if (insert_query != ""):
                    self.cursor.execute(insert_query)
                    self.connection.commit()
                
                ut.logOperation(f'''Success: {share.symbol} saved successfully {df.shape[0]} row written''')
            else:
                for row in df.itertuples():
                    tmpRow = row
                    select_query_1 = f' SELECT COUNT(*) FROM "sharesPricesQuots" WHERE "time"=\'{row.Index}\' and "idShare"=({subQuery}) '
                    self.cursor.execute(select_query_1)
                    self.connection.commit()
                    data = self.cursor.fetchall()
                    if data[0][0] > 0:
                        nbAlreadySavedRow += 1
                    else:
                        if row.Open == row.Open or row.High == row.High or row.Low == row.Low or row.Close == row.Close: # NaN != NaN == False
                            insert_query += ( f' INSERT INTO public."sharesPricesQuots"('
                                                    f' "time", "openPrice", "highPrice", "lowPrice", "closePrice", "volume", "dividend", "idShare") '
                                                    f' VALUES (\'{row.Index}\', \'{row.Open}\', \'{row.High}\', \'{row.Low}\', \'{row.Close}\', \'{row.Volume}\', \'{row.Dividends}\', ({subQuery})); ' 
                                                    )
                # Execute all requests at once
                if (insert_query != ""):
                    self.cursor.execute(insert_query)
                    self.connection.commit()

                ut.logOperation(f'''Success: {share.symbol} saved successfully new/total:{df.shape[0]-nbAlreadySavedRow}/{df.shape[0]}''')

        except (Exception, psycopg2.DatabaseError) as error :
            print ("Error while saving share: ", error)
            ut.logOperation(insert_query)
            ut.logOperation("Error: {share.symbol} failed", listErrors=[error])
            if (nbTry == 1):
                nbTry += 1
                print ("Retry by removing duplicates in retrieved quots")
                ut.logOperation("Retry by removing duplicates in retrieved quots")
                df = df.drop_duplicates()
                saveDataInDB(df, share, checkDuplicate, nbTry)
                return 0
            exit (-1)

        return nbAlreadySavedRow


    def getLastDateQuot(self, shareObj):
        date=None
        try:
            select_query = f''' SELECT "time" FROM "sharesPricesQuots" INNER JOIN "sharesInfos" ON "sharesPricesQuots"."idShare" = "sharesInfos"."idShare" WHERE "sharesInfos"."symbol"='{shareObj.symbol}'
                                        ORDER BY "sharesPricesQuots"."time" DESC LIMIT 1 '''

            self.cursor.execute(select_query)
            self.connection.commit()
            data = self.cursor.fetchall()
            if len(data) > 0:
                date = data[0][0] #First row, first column
                zone = pytz.timezone(shareObj.exchangeTimezoneName)
                date = ut.convertDateWithoutTimezone(date, zone)

                #date = date.replace(tzinfo=pytz.timezone())
                #date = date.astimezone(pytz.utc)
                #date = date.replace(tzinfo=None)

            print("Last quots's date successful retrieved from PostgreSQL ")

        except (Exception, psycopg2.DatabaseError) as error:
            print ("Error while retrieving last quot: ", error)
            exit(-1)
        return date

    def getFirstDateQuot(self, shareObj):
        date=None
        try:
            select_query = f''' SELECT "time" FROM "sharesPricesQuots" INNER JOIN "sharesInfos" ON "sharesPricesQuots"."idShare" = "sharesInfos"."idShare" WHERE "sharesInfos"."symbol"='{shareObj.symbol}'
                                        ORDER BY "sharesPricesQuots"."time" ASC LIMIT 1 '''

            self.cursor.execute(select_query)
            self.connection.commit()
            data = self.cursor.fetchall()
            if len(data) > 0:
                date = data[0][0] #First row, first column
                zone = pytz.timezone(shareObj.exchangeTimezoneName)
                date = ut.convertDateWithoutTimezone(date, zone)
                print("Last quots's date successful retrieved from PostgreSQL")

        except (Exception, psycopg2.DatabaseError) as error:
            print ("Error while retrieving first quot: ", error)
            exit(-1)
        return date


    def getQuots(self, share, dateBegin, dateEnd):
        dataFrame = pd.DataFrame()
        try:
            select_query = f''' SELECT * FROM "sharesInfos" LIMIT 0'''
            self.cursor.execute(select_query)
            self.connection.commit()
            columns = [desc[0] for desc in self.cursor.description]
            sub_query = f'''SELECT "idShare" FROM "sharesInfos" where "symbol"='{share.symbol}' '''
            select_query =f'''SELECT "time", "openPrice", "highPrice", "lowPrice", "closePrice", "volume", "dividend" FROM "sharesPricesQuots" WHERE "idShare"= ({sub_query}) and "time" >= '{dateBegin}' and "time" < '{dateEnd}' ORDER BY "time" '''
            dataFrame = pd.read_sql(select_query, self.connection, index_col=["time"], parse_dates=["time"], columns=columns)

        except (Exception, psycopg2.DatabaseError) as error :
            print (f"Error while getting quots for {share.symbol}: ", error)
            exit(-1)
        return dataFrame


    def readSharesInfos(self, readOnlyThosetoUpdate=False):
        dataFrame = pd.DataFrame()
        try:
            select_query = f''' SELECT * FROM "sharesInfos" LIMIT 0'''
            self.cursor.execute(select_query)
            self.connection.commit()
            columns = [desc[0] for desc in self.cursor.description]
            select_query = f''' SELECT "sharesInfos".* FROM "sharesInfos"'''
            if readOnlyThosetoUpdate:
                select_query += f''' INNER JOIN "sharesStats" ON "sharesStats"."idShare"="sharesInfos"."idShare" where "isUpdated" != 'false' or "isUpdated" is NULL  '''
            dataFrame = pd.read_sql(select_query, self.connection, columns=columns)
            self.sharesObj.setAllShares(dataFrame)
        except (Exception, psycopg2.DatabaseError) as error :
            print (f"Error while getting shares infos: ", error)

    def downloadDataInDB(self, share, dateBegin='max', dateEnd='now'): # Until now
        
        df = ut.downloadDataFromYahoo(share, dateBegin, dateEnd)
        ut.logOperation(f"Success {share.symbol} downloaded from {dateBegin} until {dateEnd}", nbRowsWritten=df.shape[0])
        return df

    def createInfoColumn(self, shareName):
        try:
            info = ut.downloadInfoFromYahoo(shareName)
            create_column_query = 'ALTER TABLE "sharesInfos" '
            for key, value in info.items():
                if  type(value) == float:
                    create_column_query += f'\n\tADD IF NOT EXISTS "{key}" NUMERIC(12,5),'
                elif type(value) == int:
                    create_column_query += f'\n\tADD IF NOT EXISTS "{key}" NUMERIC(14,0),'
                else:
                    create_column_query += f'\n\tADD IF NOT EXISTS "{key}" CHARACTER VARYING,'
            create_column_query = create_column_query[:-1] # Delete comma
            self.cursor.execute(create_column_query)
            self.connection.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print ("Error while creating column ", error)
            exit(-1)
    # Update shares tables with all info retreived from yahoo
    def prepareUpdateQueryInfo(self, shareName):
        info = ut.downloadInfoFromYahoo(shareName)
        update_query = 'UPDATE "sharesInfos" SET '
        for key, value in info.items():
            if key != 'err': #Sometime error occured with and a key 'err' is added with corresponding value
                if value == None:
                    value = 'NULL'
                else:
                    if type(value)==str:
                        value = value.replace("'","''") # si ' dans la string (echapement)
                    value = f"'{value}'"
                update_query += f''' "{key}" = {value},'''
        update_query = update_query[:-1]
        update_query += f'''\n WHERE "symbol"='{shareName}' '''
        return update_query

    # Update shares tables with all info retreived from yahoo
    def prepareInsertQueryInfo(self, shareName):
        info = ut.downloadInfoFromYahoo(shareName)
        if (len(info.items()) > 10): # because yahoo finance give dict_items([('regularMarketPrice', None), ('logo_url', '')]) when no ticker
            insert_query = 'INSERT INTO "sharesInfos"('
            for key, value in info.items():
                if key != 'err': #Sometime error occured with and a key 'err' is added with corresponding value
                    insert_query += f'''"{key}",'''
            insert_query = insert_query[:-1]
            insert_query += ')\nVALUES ('
            for key, value in info.items():
                if key != 'err':
                    if value == None:
                        value = 'NULL'
                    elif value == 'Infinity':
                        value = 'NULL'
                    else:
                        if type(value)==str:
                            value = value.replace("'","''") # si ' dans la string (echapement)
                        elif type(value)==float:
                            value = round(value,MAX_DIGIT_PRECISION)
                        value = f"'{value}'"
                    insert_query += f"{value},"

            insert_query = insert_query[:-1]
            insert_query += ');'
            return insert_query
        else:
            return ""

    def readInfosShare(self, shareName):
        data = dict()
        select_query = f'''SELECT * from "sharesInfos" WHERE "symbol"='{shareName}' '''
        self.cursor.execute(select_query)
        self.connection.commit()
        columns = [desc[0] for desc in self.cursor.description]
        data["columns"] = columns
        data["values"] = self.cursor.fetchall()
        if data["values"] != []:
            data["values"] = data["values"][0] # Only 1 result because unicity on symbol
        return data

    def updateInfoAllShares(self):
        for shareObj in self.sharesObj.listShares.values():
            self.updateInfoShare(shareObj)

    # Function insert or update a column in table containing shares infos if a column not present add it
    def modifyInfoShare(self, shareName, method=UPDATE_PROCESS_ENUM, tryUpdate=1, columnToAdd=""):
        try:
            if method==UPDATE_PROCESS_ENUM:
                actionStr = "Updated"
            else:
                actionStr = "Added"

            if (tryUpdate>1):
                ut.logOperation(f"Update: Try add {columnToAdd} column")
                # Reinit to clear error?
                self.disconnect()
                self.connect()
                self.createInfoColumn(shareName) # Create all columns relative to the info retrieve for this share
                if (tryUpdate > 10):
                    raise Exception("Too many attempts to update column info (>10)")
            if method==UPDATE_PROCESS_ENUM:
                query = self.prepareUpdateQueryInfo(shareName)
            else:
                query = self.prepareInsertQueryInfo(shareName)
                if query == "":
                    return 0

            self.cursor.execute(query)
            self.connection.commit()
            if (columnToAdd != ""):
                ut.logOperation(f"Update: {columnToAdd} added")
                ut.logMemoryIncrement("AddedColumnInfo")
            return 1

        except (psycopg2.DatabaseError) as error:
            if (type(error) == psycopg2.errors.UndefinedColumn):
                regColumn = re.search('«(.*?)»', error.pgerror)
                if (regColumn != None):
                    nameColumnAdded = regColumn.group(1)
                    tryUpdate+=1
                    self.modifyInfoShare(shareName, method, tryUpdate, nameColumnAdded)
                else:
                    ut.logOperation(f"Error of columns but cannot identified which colum is missing in {actionStr} info {shareName}: ", [error])
                    exit(-1)
            else:
                ut.logOperation(f"Error with DB while saving infos with {shareName}: query: {query}", [error])
                exit(-1)

        except (Exception) as error:
            ut.logOperation(f"Error while saving infos with {shareName}: ", [error])
            exit(-1)

    def updateInfoShare(self, shareName):
        return self.modifyInfoShare(shareName, UPDATE_PROCESS_ENUM)

    def addInfosShare(self, shareName):
        return self.modifyInfoShare(shareName, CREATE_PROCESS_ENUM)

    def checkVolumeExcess(self):
        maxVolume = 1000000000
        listDateAndTickerName = []
        try:
            select_query = f'''select "sharesInfos"."symbol", "sharesPricesQuots"."time", "sharesPricesQuots"."idShare" from "sharesPricesQuots" INNER JOIN "sharesInfos" ON "sharesPricesQuots"."idShare" = "sharesInfos"."idShare" where "sharesPricesQuots"."volume">'{maxVolume}' and "sharesPricesQuots"."time" > '{(datetime.datetime.now()-datetime.timedelta(days=30, minutes=-1))}'  ORDER BY time DESC'''
            #select_query = f'''select "sharesPricesQuots"."idShare", "sharesPricesQuots"."time" from "sharesPricesQuots" where "sharesPricesQuots"."volume">'{maxVolume}' and "sharesPricesQuots"."time" > '{(datetime.datetime.now()-datetime.timedelta(days=30, minutes=-1))}'  ORDER BY time DESC'''
            self.cursor.execute(select_query)
            self.connection.commit()
            listDateAndTickerName = self.cursor.fetchall()
            #for quots in listQuots:
            #    listDateAndTickerName.append([self.sharesObj.getRowsByKeysValues("symbol", quots[0]), quots[1]])
        except (Exception, psycopg2.DatabaseError) as error:
            print (f"Error while requesting volume excess : ", error)
            exit(-11)
        return listDateAndTickerName
        
    def updateVolumeExcess(self, listDateAndTickerName):
        try:
            cpt = 0
            for quot in listDateAndTickerName:
                
                #share = self.sharesObj.getShareByTickerName(quot[0])
                data = ut.downloadDataFromYahooByTickerName(quot[0], quot[1], quot[1]+datetime.timedelta(minutes=1))
                try:
                    data = data.iloc[0]
                    update_query = f'''update "sharesPricesQuots" SET "volume"='{data.Volume}' where "time"='{quot[1]}' and "idShare"='{quot[2]}' '''
                    self.cursor.execute(update_query)
                    self.connection.commit()
                except(Exception) as error:
                    print (f"Error while updating volume excess (maybe yfinance) : ", error)
                print(cpt, data.Volume)
                cpt+=1
            
        except (Exception, psycopg2.DatabaseError) as error:
            print (f"Error while updating volume excess : ", error)

    def computeNbRecordsAndAvgByDayAndFirstLastRecord(self, shareObj):
        try:
            select_query = f''' SELECT "time" FROM "sharesPricesQuots" WHERE "idShare"='{shareObj.idShare}' ORDER BY "time" ASC '''
            self.cursor.execute(select_query)
            self.connection.commit()
            data = self.cursor.fetchall()
            dateFirst = dateLast = None
            if data != []:
                dateFirst = data[0][0] #First row, first column
                dateLast = data[-1][0] #Last row, first column
                nbBusDayTotal = np.busday_count(dateFirst.date(), dateLast.date()) + 1 # +1 car bound are inclusive
                nbRecordsTotal = len(data)
                nbRecordsTotalAvgByDay = nbRecordsTotal/nbBusDayTotal
                data = list(list(zip(*data))[0]) # See https://stackoverflow.com/questions/12142133/how-to-get-first-element-in-a-list-of-tuples
                index = pd.DatetimeIndex(data) 
                data = pd.Series(data, index=index)
                dataOnLastMonth = data[dateLast-datetime.timedelta(days=30):]
                nbRecordsOnLastMonth = dataOnLastMonth.shape[0]
                nbBusDayOnLastMonth = np.busday_count(dateLast.date()-datetime.timedelta(days=30), dateLast.date()) + 1
                nbRecordsOnLastMonthAvgByDay = nbRecordsOnLastMonth/nbBusDayOnLastMonth
        except(Exception) as error:
            print (f"Error while compute stats on average records: ", error)
            exit(-1)

        try:
            if (dateFirst == None or dateLast == None):
                valuesString = f''' '0', '0', '0' ,NULL, NULL '''
            else:
                valuesString = f''' '{nbRecordsTotalAvgByDay}', '{nbRecordsOnLastMonthAvgByDay}', '{nbRecordsTotal}' ,'{dateFirst}', '{dateLast}' '''

            insert_query = f''' INSERT INTO "sharesStats"("idShare", "nbRecordsAvgByDay", "nbRecordsAvgByDayOnLastMonth", "nbRecordsInTotal", "firstRecord", "lastRecord") \
            VALUES ('{shareObj.idShare}', {valuesString}) \
            ON CONFLICT("idShare") DO UPDATE SET ("nbRecordsAvgByDay", "nbRecordsAvgByDayOnLastMonth", "nbRecordsInTotal", "firstRecord", "lastRecord") = \
            ({valuesString}) '''

            self.cursor.execute(insert_query)
            self.connection.commit()
        except(Exception) as error:
            print (f"Error while saving stats on average records: ", error)
            exit(-1)

    def computeLastRecord(self, shareObj):
        try:
            data = None
            select_query = f''' SELECT "time" FROM "sharesPricesQuots" WHERE "idShare"='{shareObj.idShare}' ORDER BY "time" DESC LIMIT 1 '''
            self.cursor.execute(select_query)
            self.connection.commit()
            data = self.cursor.fetchall()
            if data != [] or data != None :
                dateFirst = data[0][0] #LastRecord
                update_query = f''' UPDATE "sharesStats" SET "lastRecord"='{dateFirst}' WHERE "idShare"='{shareObj.idShare}';\n '''
                self.cursor.execute(update_query)
                self.connection.commit()

        except(Exception) as error:
            print (f"Error while saving stat lastRecord: ", error)
            exit(-1)

    def computeAvgMaxRangeByDay(self, shareObj):
        return 0

    # A revoir
    def discardUpdate(self, shareObj, nbRecordsMinByDay, nbDaysMax=30):
        data = []
        try:
            select_query = f''' SELECT "nbRecordsAvgByDayOnLastMonth", "lastRecord" FROM "sharesStats" WHERE "idShare"='{shareObj.idShare}' '''
            self.cursor.execute(select_query)
            self.connection.commit()
            data = self.cursor.fetchall()
            nbRecordsAvgByDayOnLastMonth = data[0][0]
            lastRecord = data[0][1]
        except(Exception) as error:
            print (f"Error while retrieving shares stats: ", error)
            exit(-1)

        try:
            result = 'false'
            isLastQuotLessOlderThanAMonth = lastRecord-datetime.datetime.now() < datetime.timedelta(days=nbDaysMax)
            if (nbRecordsAvgByDayOnLastMonth > nbRecordsMinByDay and isLastQuotLessOlderThanAMonth):
                result = 'true'
            update_query = f''' UPDATE "sharesStats" SET "l"='{result}' WHERE "idShare"='{shareObj.idShare}';\n '''
            self.cursor.execute(update_query)
            self.connection.commit()

        except(Exception) as error:
            print (f"Error while updating shares stats : ", error)
            exit(-1)