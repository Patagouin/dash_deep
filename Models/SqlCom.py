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


    def saveDataInDB(self, df, shareObj, checkDuplicate=False, nbTry=1):
        ''' Save data (multiple lines) from a share in DB '''
        try:
            nbAlreadySavedRow = 0
            insert_query = ""
            lastTime = None

            for row in df.itertuples():
                if checkDuplicate:
                    select_query_1 = f'SELECT COUNT(*) FROM "sharesPricesQuots" WHERE "time"=\'{row.Index}\' and "idShare"={shareObj.idShare}'
                    self.cursor.execute(select_query_1)
                    self.connection.commit()
                    data = self.cursor.fetchall()
                    if data[0][0] > 0:
                        nbAlreadySavedRow += 1
                        continue

                if lastTime != row.Index:
                    if row.Open == row.Open or row.High == row.High or row.Low == row.Low or row.Close == row.Close:  # Detect if any nan value due to yahoo which give bad data
                        insert_query += ( f' INSERT INTO public."sharesPricesQuots"('
                                            f' "time", "openPrice", "highPrice", "lowPrice", "closePrice", "volume", "dividend", "idShare") '
                                            f' VALUES (\'{row.Index}\', \'{row.Open}\', \'{row.High}\', \'{row.Low}\', \'{row.Close}\', \'{row.Volume}\', \'{row.Dividends}\', {shareObj.idShare}); ' 
                                            )
                    lastTime = row.Index

            # Execute all requests at once
            if insert_query != "":
                self.cursor.execute(insert_query)
                self.connection.commit()

            if checkDuplicate:
                ut.logOperation(f'''Success: {shareObj.symbol} saved successfully new/total:{df.shape[0] - nbAlreadySavedRow}/{df.shape[0]}''')
            else:
                ut.logOperation(f'''Success: {shareObj.symbol} saved successfully {df.shape[0]} row written''')

        except (Exception, psycopg2.DatabaseError) as error :
            print ("Error while saving share: ", error)
            ut.logOperation(insert_query)
            ut.logOperation("Error: {shareObj.symbol} failed", listErrors=[error])
            if (nbTry == 1):
                nbTry += 1
                print ("Retry by removing duplicates in retrieved quots")
                ut.logOperation("Retry by removing duplicates in retrieved quots")
                df = df.drop_duplicates()
                self.saveDataInDB(df, shareObj, checkDuplicate, nbTry)
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


    def getQuots(self, shareObj, dateBegin, dateEnd):
        dataFrame = pd.DataFrame()
        try:
            select_query = f''' SELECT * FROM "sharesInfos" LIMIT 0'''
            self.cursor.execute(select_query)
            self.connection.commit()
            columns = [desc[0] for desc in self.cursor.description]
            sub_query = f'''SELECT "idShare" FROM "sharesInfos" where "symbol"='{shareObj.symbol}' '''
            select_query =f'''SELECT "time", "openPrice", "highPrice", "lowPrice", "closePrice", "volume", "dividend" FROM "sharesPricesQuots" WHERE "idShare"= ({sub_query}) and "time" >= '{dateBegin}' and "time" < '{dateEnd}' ORDER BY "time" '''
            dataFrame = pd.read_sql(select_query, self.connection, index_col=["time"], parse_dates=["time"], columns=columns)

        except (Exception, psycopg2.DatabaseError) as error :
            print (f"Error while getting quots for {shareObj.symbol}: ", error)
            exit(-1)
        return dataFrame


    def readSharesInfos(self, readOnlyThosetoUpdate=False):
        dataFrame = pd.DataFrame()
        try:
            select_query = f''' SELECT * FROM "sharesInfos" LIMIT 0'''
            self.cursor.execute(select_query)
            self.connection.commit()
            columns = [desc[0] for desc in self.cursor.description]
            select_query = f''' SELECT "sharesInfos".* FROM "sharesInfos" '''
            if readOnlyThosetoUpdate:
                select_query += '''where "isUpdated" != 'false' or "isUpdated" is NULL'''
            dataFrame = pd.read_sql(select_query, self.connection, columns=columns)
            return dataFrame
            #self.sharesObj.setAllShares(dataFrame)
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
        for shareObj in self.sharesObj.dfShares.values():
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

    def computeNbRecordsAndAvgByDay(self, shareObj):
        nbRecordsTotalAvgByDay = 0
        nbRecordsAvgByDayOnLastMonth = 0
        nbRecordsInTotal = 0
        try:
            select_query = f''' SELECT "time" FROM "sharesPricesQuots" WHERE "idShare"='{shareObj.idShare}' ORDER BY "time" ASC '''
            self.cursor.execute(select_query)
            self.connection.commit()
            data = self.cursor.fetchall()
            dateFirst = shareObj.firstRecord
            dateLast = shareObj.lastRecord
            if data != []:
                #dateFirst = data[0][0] #First row, first column
                #dateLast = data[-1][0] #Last row, first column
                nbBusDayTotal = np.busday_count(dateFirst.date(), dateLast.date()) + 1 # +1 car bound are inclusive
                nbRecordsInTotal = len(data)
                nbRecordsTotalAvgByDay = nbRecordsInTotal/nbBusDayTotal
                data = list(list(zip(*data))[0]) # See https://stackoverflow.com/questions/12142133/how-to-get-first-element-in-a-list-of-tuples
                index = pd.DatetimeIndex(data) 
                data = pd.Series(data, index=index)
                dataOnLastMonth = data[dateLast-datetime.timedelta(days=30):]
                nbRecordsOnLastMonth = dataOnLastMonth.shape[0]
                nbBusDayOnLastMonth = np.busday_count(dateLast.date()-datetime.timedelta(days=30), dateLast.date()) + 1
                nbRecordsAvgByDayOnLastMonth = nbRecordsOnLastMonth/nbBusDayOnLastMonth
        except(Exception) as error:
            print (f"Error while compute stats on average records: ", error)
            exit(-1)

        try:
            insert_query = f'''UPDATE "sharesInfos" \
            SET "nbRecordsTotalAvgByDay" = {nbRecordsTotalAvgByDay}, "nbRecordsAvgByDayOnLastMonth" = {nbRecordsAvgByDayOnLastMonth}, "nbRecordsInTotal" = {nbRecordsInTotal} \
            WHERE "idShare" = '{shareObj.idShare}' '''
            self.cursor.execute(insert_query)
            self.connection.commit()
        except(Exception) as error:
            print (f"Error while saving stats on average records: ", error)
            exit(-1)



    def getFirstRecord(self, shareObj):
        try:
            data = None
            select_query = f''' SELECT "time" FROM "sharesPricesQuots" WHERE "idShare"='{shareObj.idShare}' ORDER BY "time" ASC LIMIT 1 '''
            self.cursor.execute(select_query)
            self.connection.commit()
            data = self.cursor.fetchall()
            if data != [] and data != None :
                dateFirst = data[0][0] #dateFirst
                return dateFirst

        except(Exception) as error:
            print (f"Error retrieving first record: ", error)
            exit(-1)


    def getLastRecord(self, shareObj):
        try:
            data = None
            select_query = f''' SELECT "time" FROM "sharesPricesQuots" WHERE "idShare"='{shareObj.idShare}' ORDER BY "time" DESC LIMIT 1 '''
            self.cursor.execute(select_query)
            self.connection.commit()
            data = self.cursor.fetchall()
            if data != [] and data != None :
                dateLast = data[0][0] #LastRecord
                return dateLast
            return None

        except(Exception) as error:
            print (f"Error retrieving last record: ", error)
            exit(-1)

    ## LastRecord est aussi compute par la fonction computeNbRecordsAndAvgByDay
    def computeLastRecord(self, shareObj, dateLast="ToCompute"):
        try:
            if dateLast == "ToCompute":
                dateLast = self.getLastRecord(shareObj)

            if dateLast != [] and dateLast != None :
                update_query = f''' UPDATE "sharesInfos" SET "lastRecord"='{dateLast}' WHERE "idShare"='{shareObj.idShare}';\n '''
                self.cursor.execute(update_query)
                self.connection.commit()

        except(Exception) as error:
            print (f"Error while saving info lastRecord err: ", error)
            exit(-1)

    # N'est pas à caculer à chaque maj des infos
    def computeFirstRecord(self, shareObj, dateFirst="ToCompute"):
        try:
            if dateFirst == "ToCompute":
                dateFirst = self.getFirstRecord(shareObj)

            if dateFirst != [] and dateFirst != None :
                update_query = f''' UPDATE "sharesInfos" SET "firstRecord"='{dateFirst}' WHERE "idShare"='{shareObj.idShare}';\n '''
                self.cursor.execute(update_query)
                self.connection.commit()

        except(Exception) as error:
            print (f"Error while saving info firstRecord err: ", error)
            exit(-1)

    def get_last_potential_date(self, shareObj):
        try:
            select_query = f'''SELECT "date" FROM "sharesPotentials" WHERE "idShare"='{shareObj.idShare}' ORDER BY "date" DESC LIMIT 1'''
            self.cursor.execute(select_query)
            self.connection.commit()
            data = self.cursor.fetchall()

            if data != [] and data is not None:
                last_date_not_computed = data[0][0]
            else:
                last_date_not_computed = self.getFirstDateQuot(shareObj)

            return last_date_not_computed

        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Error while fetching the last potential date: ", error)
            exit(-1)

    def get_cotations_data(self, shareObj, start_date, end_date, column="openPrice"):
        try:
            select_query = f'''SELECT "{column}", "time" FROM "sharesPricesQuots" WHERE "idShare"='{shareObj.idShare}' AND "time" >= '{start_date}' AND "time" <= '{end_date}' ORDER BY "time" ASC'''
            self.cursor.execute(select_query)
            self.connection.commit()
            data = self.cursor.fetchall()

            if data != [] and data is not None:
                data_quots = np.array(data)[:, 0]
                data_quots = data.astype(np.float)
                nan_indices = np.isnan(data_quots)
                if np.any(nan_indices):
                    non_nan_indices = np.flatnonzero(~nan_indices)
                    # Interpoler les valeurs pour les indices NaN à partir des valeurs voisines
                    interp_values = np.interp(nan_indices.nonzero()[0], non_nan_indices, data_quots[non_nan_indices])
                    # Remplacer les valeurs NaN par les valeurs interpolées
                    data_quots[nan_indices] = interp_values
                data_time = np.array(data)[:, 1]
            else:
                print(f"No data found for idShare '{shareObj.symbol}' ('{shareObj.idShare}') in the specified date range {start_date} and {end_date}")
                data_quots = np.array([])
                data_time = np.array([])

            return data_quots, data_time

        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Error while fetching cotations data: ", error)
            exit(-1)

#semble pas valide
    def get_cotations_data_df(self, shareObj, start_date, end_date):
        try:
            select_query = f'''SELECT * FROM "sharesPricesQuots" WHERE "idShare"='{shareObj.idShare}' AND "time" >= '{start_date}' AND "time" <= '{end_date}' ORDER BY "time" ASC'''
            self.cursor.execute(select_query)
            self.connection.commit()
            data = self.cursor.fetchall()
            column_names = [desc[0] for desc in self.cursor.description]

            if data != [] and data is not None:
                data_df = pd.DataFrame(data, columns=column_names)
                data_df['time'] = pd.to_datetime(data_df['time'])
                data_df.set_index('time', inplace=True)

            else:
                print(f"No data found for idShare '{shareObj.symbol}' ('{shareObj.idShare}') in the specified date range {start_date} and {end_date}")
                data_df = pd.DataFrame()

            return data_df

        except Exception as e:
            print(f"Erreur lors de l'extraction des données: {e}")
            return pd.DataFrame()


    # Créer une requête pour récup' la dernière valeur de mise à jour potential dans sharesPotential
    # Itérer sur les jours ouvrable ou sur tout les jours (faire attention au timezone)
    # Mettre à jour la table (chaque colonne = nb action 1,2,3,4,5,7,10,15,20,30,50,75,100 en % seulement si trouvé sinon null)
        # Plus la valeur du % est elevé pour nb action élevé plus c'est mieux
        # A voir aussi si possibilité de mettre en évidence une idée de la volatilité (si forte augmentation en ajoutant nb action)
    def computePotential(self, shareObj):
        try:
            data = None
            lastDateNotComputed = self.get_last_potential_date(shareObj)
            potential_levels = [30,15,10,7,5,4,3,2,1]
            # Calculate potential for each business day from lastDateNotComputed
            date_range = pd.bdate_range(lastDateNotComputed, pd.Timestamp.today())
            for date in date_range:
                start_of_day = date.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
                end_of_day = start_of_day + datetime.timedelta(days=1, microseconds=-1)
                # cotations_data = self.getQuots(shareObj, start_of_day, end_of_day)
                cotations_data, time_data = self.get_cotations_data(shareObj, start_of_day, end_of_day, "openPrice")
                if (cotations_data.size != 0):
                    # Calculez le potentiel pour la date actuelle en utilisant getPotential
                    resultIndex, resultPercent, resultPercentTotal = ut.getPotential(cotations_data, potential_levels)
                    
                    # Convertir les index de la liste 2D en valeurs du tableau `time_data` (en string)
                    #resultTime = [[time_data[range_index[0]].strftime('%H:%M'), time_data[range_index[1]].strftime('%H:%M')] for range_index in potential] for potential in resultIndex]
                    # Transformation des éléments de time_data en utilisant resultIndex
                    resultTime = [
                        [
                            [time_data[index].strftime('%H:%M') for index in range_index]
                            for range_index in potential
                        ]
                        for potential in resultIndex
                    ]
                    insert_query =  f''' INSERT INTO "sharesPotentials"("idShare", "date" '''
                    for i in potential_levels:
                        insert_query += f''', "{str(i)}_time", "{str(i)}_percentTotal", "{str(i)}_percent" '''
                    insert_query += f''')\nVALUES ('{shareObj.idShare}', '{date}' '''
                    for time, percentTotal, percent in zip(resultTime, resultPercentTotal, resultPercent):
                        if (time != []):
                            time_str = '{' + ','.join(['{' + ','.join(row) + '}' for row in time]) + '}'
                            percent_str = '{' + ','.join(str(x) for x in percent) + '}'
                            insert_query += f''', '{time_str}', '{percentTotal}',  '{percent_str}' '''
                        else:
                            insert_query += f''', NULL, NULL, NULL '''
                    insert_query += f''')\nON CONFLICT ("idShare", "date") DO UPDATE SET '''
                    for i, (time, percentTotal, percent) in enumerate(zip(resultTime, resultPercentTotal, resultPercent)):
                        # pas besoin de tester null vu que Excluded reprend la valeur qu'on a voulu mettre donc null
                        insert_query += f''' "{str(potential_levels[i])}_time" = EXCLUDED."{str(potential_levels[i])}_time", "{str(potential_levels[i])}_percentTotal" = EXCLUDED."{str(potential_levels[i])}_percentTotal", "{str(potential_levels[i])}_percent" = EXCLUDED."{str(potential_levels[i])}_percent",'''
                    insert_query = insert_query.rstrip(',')

                
                    ## Do nothing  in case on conflict, ON CONFLICT("idShare", "date") DO UPDATE SET "potential"='{potential}' '''
                    self.cursor.execute(insert_query)
                    self.connection.commit()

        except(Exception) as error:
            print (f"Error while computing potential: ", error)
            exit(-1)

    # A revoir
    def discardUpdate(self, shareObj, nbRecordsMinByDay, nbDaysMax=30):
        try:
            nbRecordsAvgByDayOnLastMonth = shareObj.nbRecordsAvgByDayOnLastMonth
            lastRecord =shareObj.lastRecord
            if lastRecord != None:
                result = 'false'
                isLastQuotLessOlderThanAMonth = (lastRecord - datetime.datetime.now()) < datetime.timedelta(days=nbDaysMax)
                if (nbRecordsAvgByDayOnLastMonth > nbRecordsMinByDay and isLastQuotLessOlderThanAMonth):
                    result = 'true'
                update_query = f''' UPDATE "sharesInfos" SET "isUpdated"='{result}' WHERE "idShare"='{shareObj.idShare}';\n '''
                self.cursor.execute(update_query)
                self.connection.commit()
            else:
                ut.logOperation(f"Pas de data pour {shareObj.symbol} ou lastRecord pas calculé")
        except(Exception) as error:
            print (f"Error while updating shares stats : ", error)
            exit(-1)

    def compute_frequence_cotation_via_graph(self, shareObj):
        select_query = f'''SELECT "time" FROM "sharesPricesQuots" WHERE "idShare"='{shareObj.idShare}' '''
        self.cursor.execute(select_query)
        self.connection.commit()
        data = self.cursor.fetchall()

        # Calculez le nombre de jours ouvrés entre dateFirst et dateLast

        if data != [] and data is not None:
            data_df = pd.DataFrame(data, columns=['time'])
            import matplotlib.pyplot as plt

            data_df['time'] = pd.to_datetime(data_df['time'])
            data_df['time_of_day'] = data_df['time'].dt.time


            # Convertir les objets 'time' en secondes pour les utiliser dans un histogramme
            def time_to_minutes(time_obj):
                return time_obj.hour * 60 + time_obj.minute

            data_df_copy = data_df.copy()
            data_df_copy['time_minutes'] = data_df['time_of_day'].apply(time_to_minutes)

            # Créer l'histogramme
            plt.figure(figsize=(8, 6), dpi=200)

            plt.hist(data_df_copy['time_minutes'], bins=range(0, 24*60, 1), edgecolor='black')
            # Convertir les secondes en objets timedelta pour un affichage plus lisible
            def minutes_to_timedelta(minutes):
                return datetime.timedelta(minutes=minutes)

            # Définir les positions et les labels personnalisés des graduations
            xtick_minutes = range(0, 24*60, 60)  # Chaque heure
            #xtick_labels = [minutes_to_timedelta(s) for s in xtick_minutes]
            xtick_labels = [datetime.time(hour=int(m // 60), minute=int(m % 60)) for m in xtick_minutes]

            plt.xticks(xtick_minutes, xtick_labels, rotation=45, ha='right')
            plt.xlabel('Heure de la journée (minutes)')
            plt.ylabel('Fréquence des cotations')
            plt.title('Histogramme de la fréquence des cotations au cours d\'une journée')
            plt.savefig(f'repartition_cotation_one_day/{shareObj.symbol}.png', bbox_inches='tight')



    def compute_general_market_open_close_time(self, shareObj, min_duration=30, percentThreshold=30):
        try:
            dateFirst = shareObj.firstRecord;
            dateLast = shareObj.lastRecord;

            select_query = f'''SELECT "time" FROM "sharesPricesQuots" WHERE "idShare"='{shareObj.idShare}' '''
            self.cursor.execute(select_query)
            self.connection.commit()
            data = self.cursor.fetchall()

            if data != [] and data is not None:

                # Calculez le nombre de jours ouvrés entre dateFirst et dateLast
                num_business_days = np.busday_count(dateFirst.date().isoformat(), dateLast.date().isoformat())

                openRichMarketTime = None
                closeRichMarketTime = None
                data_df = pd.DataFrame(data, columns=['time'])

                data_df['time'] = pd.to_datetime(data_df['time'])
                data_df['time_of_day'] = data_df['time'].dt.time
				
                openMarketTimeExtended = data_df['time_of_day'].min()
                closeMarketTimeExtended = data_df['time_of_day'].max()

                openMarketTimeExtended = ut.round_time_to_nearest_minutes(openMarketTimeExtended, 10)
                closeMarketTimeExtended = ut.round_time_to_nearest_minutes(closeMarketTimeExtended, 10)
                
                openMarketTime = openMarketTimeExtended
                closeMarketTime = closeMarketTimeExtended
                if openMarketTimeExtended == datetime.time(4, 00):
                    if closeMarketTimeExtended == datetime.time(20, 00):
                        openMarketTime = datetime.time(9, 30)
                        closeMarketTime = datetime.time(16, 00)

                else:
                    if openMarketTimeExtended == datetime.time(9, 0):
                        if closeMarketTimeExtended == datetime.time(17, 30):
                            openMarketTime = datetime.time(9, 0)
                            closeMarketTime = datetime.time(17, 30)

                # Arrondir les objets 'time' à la minute la plus proche
                data_df['minute_of_day'] = data_df['time'].dt.round('T').dt.time

                # Grouper les données par 'minute_of_day' et compter le nombre de cotations pour chaque minute
                minute_counts = data_df.groupby('minute_of_day').size().reset_index(name='count')
                
                threshold = num_business_days / (100/percentThreshold)

                # Ajoutez une colonne pour indiquer si le nombre d'occurrences dépasse le seuil
                minute_counts['above_threshold'] = (minute_counts['count'] > threshold).astype(int)

                # Use rolling() and sum() to find periods of {min_duration} consecutive minutes above the threshold.
                minute_counts['consecutive_above_threshold'] = minute_counts['above_threshold'].rolling(min_duration).sum()

                # Check if there are any periods with at least {min_duration} consecutive minutes above the threshold.
                if (minute_counts['consecutive_above_threshold'] == min_duration).any():

                    first_minute_above_threshold_index = minute_counts.loc[minute_counts['consecutive_above_threshold'] >= min_duration].index[0]
                    last_minute_above_threshold_index = minute_counts.loc[minute_counts['consecutive_above_threshold'] >= min_duration].index[-1]

                    openRichMarketTime = minute_counts.loc[first_minute_above_threshold_index - (min_duration-1), 'minute_of_day']
                    closeRichMarketTime = minute_counts.loc[last_minute_above_threshold_index, 'minute_of_day']

                    openRichMarketTime = ut.round_time_to_nearest_minutes(openRichMarketTime, 10)
                    closeRichMarketTime = ut.round_time_to_nearest_minutes(closeRichMarketTime, 10)


                try:
                    if dateLast != [] or dateLast != None :
                        update_query = f''' UPDATE "sharesInfos" SET \
                                                    "openMarketTime"='{openMarketTime}', \
                                                    "closeMarketTime"='{closeMarketTime}', \
                                                    "openMarketTimeExtended"='{openMarketTimeExtended}',
                                                    "closeMarketTimeExtended"='{closeMarketTimeExtended}' ''' + \
                                                   (f''', "openRichMarketTime"='{openRichMarketTime}' ''' if openRichMarketTime is not None else '') + \
                                                   (f''', "closeRichMarketTime"='{closeRichMarketTime}' ''' if closeRichMarketTime is not None else '') + \
                                                   f''' WHERE "idShare"='{shareObj.idShare}';\n '''
                        self.cursor.execute(update_query)
                        self.connection.commit()

                except(Exception) as error:
                    print (f"Error while saving stat the market times: ", error)
                    exit(-1)

            else:
                ut.logOperation(f"No data found for idShare '{shareObj.symbol}' ('{shareObj.idShare}')")


        except Exception as e:
            print(f"Erreur lors de l'extraction des données: {e}")

    def saveModel(self, shareObj, modelName, modelBin, trainScore, testScore):
        try:
            current_date = datetime.datetime.now
            update_query = f'''INSERT INTO modelsDeepLearning ("id", "idShare", "model", "trainScore", "testScore", "date") 
                            VALUES ('{modelName}', '{shareObj.idShare}', '{modelBin}', '{trainScore}', '{testScore}', '{current_date}' )
                            ON CONFLICT ("id", "idShare", "model", "trainScore", "testScore", "date") 
                            DO UPDATE
                            SET "id" = EXCLUDED."id", "idShare" = EXCLUDED."idShare", "model" = EXCLUDED."model", "trainScore" = EXCLUDED."trainScore", "testScore" = EXCLUDED."testScore", "date" = EXCLUDED."date"'''

            self.cursor.execute(update_query)
            self.connection.commit()
        except(Exception) as error:
            print (f"Error while updating model: {modelName} pour {shareObj.symbol}: ", error)
            exit(-1)