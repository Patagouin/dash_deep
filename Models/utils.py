import datetime
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pytz
import numpy as np
from collections import OrderedDict
import os
from typing import Optional, Tuple, Dict, Any

logType = {"Success":0, "Warning":1, "Error":2}

nbSecondsInWeek = 604800 - 3600 # Par rapport au chgt d'heure (heure d'�t� heure d'hiver)

memoryLog = dict()
lastMemoryLog = ""

def getLogString(dateNowString, dateBeginString, dateEndString, resol, error="Success", isManual=False):
    if isManual:
        return f'{dateNowString} ; {dateBeginString}_{dateEndString}_{resol}r {error} (Manual)\n'
    else:
        return f'{dateNowString} ; {dateBeginString}_{dateEndString}_{resol}r {error}\n'

# Maintain the size of the file under sizeChar by reducing the size by reduc 
def maintainLogSize(logFile, sizeChar=1000000, reduc=2):
    if (logFile.tell() > sizeChar):
        logFile.seek(0,0) ## We need to precise to startat the beginning (the file poitner is localized at the end when append mode)
        logData = logFile.read()
        logData = logData[int(len(logData)/reduc):]
        logFile.truncate(0)
        logFile.write(logData)

def logOperation(message, listErrors=[],  nbRowsWritten="ND", isTotal=False, newlineAfter=False):
    dateNow = datetime.datetime.now()
    dateNowString = dateNow.strftime("%a-%d-%b-%Y_%Hh%Mm%S")
    nameDay = dateNow.strftime('%A')
    log_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../data'))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'logFile_{nameDay}.log')
    with open(log_path, 'a+', newline='') as logFile:
        maintainLogSize(logFile)
        str1 = f'{message}' + (f" ({nbRowsWritten}) rows downloaded" if (nbRowsWritten!="ND") else "")
        str1 = str1 + (" in total" if (isTotal) else "")
        strLog = f"{dateNowString} : " + str1 + "\n"
        print(str1)
        logFile.write(strLog)
        for error in listErrors:
            logFile.write(f'{error}\n')
            print(f'{error}\n')
        if newlineAfter:
            logFile.write(f'\n')

def logMemoryIncrement(name):
    lastMemoryLog = name
    if name not in memoryLog:
        memoryLog[name] = 0
    memoryLog[name] += 1

def logMemorySet(name, value):
    lastMemoryLog = name
    memoryLog[name] = value

def logMemoryGet(name):
    return memoryLog[name]

def logMemoryGetAll():
    return memoryLog

def splitData(listLine, onlyHourAndMinute=True, byColumn=True):
    splittedData = []
    cptDay = 0
    if len(listLine) > 0:
        oldLine = listLine[0]
    for line in listLine:
        if oldLine[:10] != line[:10]:
            cptDay += 1

        if onlyHourAndMinute:
            linesplitted = line.split(';')
            linesplitted[0] =  linesplitted[0][11:16] # HH:MM
            splittedData += [linesplitted]
        elif byColumn:
            todo=1
        else:
            splittedData += [line.split(';')]

        oldLine = line

    return splittedData

# Check pk avec un Dataframe d'une ligne pour share j'ai pour share.symbol un object
def downloadDataFromYahoo(share, beginDate='max', endDate='max'):
    df = pd.DataFrame()
    listDate = getListOfDateForDownload(beginDate, endDate)

    curTicker = yf.Ticker(share.symbol)
    slices = []
    for date in listDate:
        dfSlice = curTicker.history(interval='1m',start=date[0],end=date[1], rounding=False, 
                                    actions=True, prepost=True, auto_adjust=False)
        # Eviter les concat sur entrées vides (FutureWarning pandas + plus rapide)
        if dfSlice is not None and not dfSlice.empty:
            slices.append(dfSlice)
        if beginDate == 'max':  #TODO UPDATE YFINANCE TO NOT HAVE TO CHECK THAT !!
            beginDateTmp = datetime.datetime.now()-datetime.timedelta(days=30, minutes=-1)
            # On ne peut appliquer ce filtre qu'une fois qu'on a effectivement des données
        else:
            # Rien
            pass

    if len(slices) > 0:
        df = pd.concat(slices)
        if beginDate == 'max':
            beginDateTmp = datetime.datetime.now()-datetime.timedelta(days=30, minutes=-1)
            if df.size > 0 and df.iloc[0].name.replace(tzinfo=None) < beginDateTmp:
                df = df.drop(df.iloc[0].name)
        elif df.size > 0 and df.iloc[0].name.replace(tzinfo=None) < beginDate:  #TODO UPDATE YFINANCE TO NOT HAVE TO CHECK THAT !!
            df = df.drop(df.iloc[0].name)
    return df

def downloadDataFromYahooByTickerName(tickerName, beginDate='max', endDate='now'):
    df = pd.DataFrame()
    listDate = getListOfDateForDownload(beginDate, endDate)

    curTicker = yf.Ticker(tickerName)
    slices = []
    for date in listDate:
        dfSlice = curTicker.history(interval='1m',start=date[0],end=date[1], rounding=False, 
                                    actions=True, prepost=True, auto_adjust=False)
        if dfSlice is not None and not dfSlice.empty:
            slices.append(dfSlice)
    if len(slices) > 0:
        df = pd.concat(slices)
    return df


def yahoo_symbol_status(symbol: str, lookback_days: int = 30) -> Tuple[bool, str]:
    """Vérifie si un symbole est 'actif' côté Yahoo Finance.
    
    Critères (pragmatiques):
    - On tente un historique daily récent (léger). Si vide => symbole probablement invalide/delisté/non supporté.
    - On récupère aussi la timezone si dispo.
    
    Returns:
        (is_active, reason)
        reason: 'ok' | 'no_recent_daily_data' | 'no_timezone' | 'exception:<...>'
    """
    try:
        t = yf.Ticker(symbol)
        tz = None
        try:
            fi = getattr(t, "fast_info", None)
            if fi is not None:
                tz = fi.get("timezone", None)
        except Exception:
            tz = None

        # Demande légère: daily sur un lookback court
        hist = t.history(period=f"{int(lookback_days)}d", interval="1d", actions=False, auto_adjust=False)
        if hist is None or hist.empty:
            # si timezone absente en plus, c'est quasi certain que le symbole n'est plus valide
            if not tz:
                return False, "no_timezone"
            return False, "no_recent_daily_data"
        return True, "ok"
    except Exception as e:
        return False, f"exception:{type(e).__name__}"

def getListOfDateForDownload(beginDate='max', endDate='now'):
    ''' Because yfinance can only manage to download 1 week of 1m granularity, this function make couple of date of 1 week '''
    ''' aligned to Monday '''

    listRangeDate = []
    if type(beginDate) == str or beginDate == None or (datetime.datetime.now()-beginDate).days >= 30:
        beginDate = datetime.datetime.now() - datetime.timedelta(days=30, hours=-1, minutes=-1) # We want current cotation
    if type(endDate) == str or endDate == None:
        endDate = datetime.datetime.now() + datetime.timedelta(minutes=1) # We want current cotation

    endDateTimestamp = datetime.datetime.timestamp(endDate)
    curBeginTimestamp = datetime.datetime.timestamp(beginDate)
    curEndTimestamp = curBeginTimestamp

    while curBeginTimestamp + nbSecondsInWeek < endDateTimestamp:
        curEndTimestamp = curBeginTimestamp + nbSecondsInWeek
        listRangeDate += [[curBeginTimestamp, curEndTimestamp]]
        curBeginTimestamp = curEndTimestamp

    listRangeDate += [[curBeginTimestamp, endDateTimestamp]]

    # Convert all timestamp to datetime
    for i in range(len(listRangeDate)):
        listRangeDate[i][0] = datetime.datetime.fromtimestamp(listRangeDate[i][0])
        listRangeDate[i][1] = datetime.datetime.fromtimestamp(listRangeDate[i][1])

    return listRangeDate

def downloadInfoFromYahoo(shareName):
    return yf.Ticker(shareName).info

def assembleDataFramesByColumn(listDfData, column, dfShare=pd.DataFrame(), listNames=None):
    #
    minIndexShared = listDfData[0].index.min()
    maxIndexShared = listDfData[0].index.max()
    dfOut = pd.DataFrame()
    cpt=0
    for i, curDf in enumerate(listDfData):
        minIndexShared = max(minIndexShared, curDf.index.min())
        maxIndexShared = min(maxIndexShared, curDf.index.max())
        curDf_reindexed = curDf.reindex(pd.date_range(start=minIndexShared,
                                                end=maxIndexShared,
                                                freq='1min'))
        curDf_interpolated = curDf_reindexed.interpolate(method='linear')
        if listNames != None:
            dfOut[f'{listNames[i]}'] = curDf_interpolated[column]
        else:
            dfOut[f'{dfShare.tickerName.iloc[cpt]}'] = curDf_interpolated[column]
        cpt += 1

    return dfOut

def fillMissingValueDf(df, method='ffill'):
    df = df.reindex(pd.date_range(start=df.index.min(),
                                end=df.index.max(),
                                freq='1min'))
    if method == 'ffill':
        df = df.ffill()
    elif method == 'bfill':
        df = df.bfill()
    else:
        df = df.interpolate(method='linear')

    return df

def normalizeByDividMin(df):
    df = df/df.min()
    return df

def prepareDf(df, method='ffill'):
    df = splitDataByDays(df)
    df = fillMissingValueDf(df,method)
    df = normalizeByDividMin(df)

    return df

def splitDataByDays(df):
    df = df.resample('1D')
    return df

        
def printDfInFile(fileName, df):
    prevMaxRows = pd.get_option('display.max_rows')
    prevMaxColumns = pd.get_option('display.max_columns')
    pd.set_option('display.max_rows', None) # max
    pd.set_option('display.max_columns', None) # max

    with open(fileName, 'w') as file:
        file.write(df.__repr__())

    pd.set_option('display.max_rows', prevMaxRows) 
    pd.set_option('display.max_columns', prevMaxColumns) 

def temporalComparison(listDfData, dfShare, columnQuots):
    ''' Return a 2D numpy array with correlation coeff each line is a comparison between two share with different shifting time '''
    shiftingTimeArray = [datetime.timedelta(minutes=1), datetime.timedelta(minutes=2), datetime.timedelta(minutes=3), datetime.timedelta(minutes=5), datetime.timedelta(minutes=8),
                     datetime.timedelta(minutes=13), datetime.timedelta(minutes=20), datetime.timedelta(minutes=30), datetime.timedelta(minutes=45), datetime.timedelta(hours=1),
                     datetime.timedelta(hours=1), datetime.timedelta(hours=1, minutes=30), datetime.timedelta(hours=2, minutes=30), datetime.timedelta(hours=4), datetime.timedelta(hours=6),
                     datetime.timedelta(hours=8), datetime.timedelta(days=1, hours=4), datetime.timedelta(days=2), datetime.timedelta(days=3), datetime.timedelta(days=5), 
                     datetime.timedelta(weeks=1, days=3), datetime.timedelta(weeks=2), datetime.timedelta(weeks=4), datetime.timedelta(weeks=5)]

    dataCorrArray = np.array([])
    for i in range(len(listDfData)-1):
        for j in range(i+1, len(listDfData)):
            listDataTmp = [listDfData[i], listDfData[j]]
            listNames = [dfShare.tickerName.iloc[i], dfShare.tickerName.iloc[j]]
            for shiftTime in shiftingTimeArray:
                listDataTmp.append(listDfData[j].shift(freq=shiftTime))
                listNames += [timedeltaToHumanReadable(shiftTime)]

            curDfData = assembleDataFramesByColumn(listDataTmp, columnQuots, listNames=listNames)


            if dataCorrArray.size == 0:
                dataCorrArray = curDfData.corr().iloc[0].values
            else:
                dataCorrArray = np.vstack([dataCorrArray, curDfData.corr().iloc[0].values])

    return dataCorrArray

def timedeltaToHumanReadable(timedelta):
    nbSeconds = timedelta.total_seconds()
    days=nbSeconds//86400
    hours=nbSeconds%86400//3600
    minutes=nbSeconds%86400%3600//60
    return f'{int(days)},{int(hours)}:{int(minutes)}'

def squared1DMatrix(array):
    mat = np.array(array)
    nb_extend_row = 0
    if mat.shape[1]-mat.shape[0] > 0:
        nb_extend_row = mat.shape[1]-mat.shape[0]
    nb_extend_column = 0
    if mat.shape[0] - mat.shape[1] > 0:
        nb_extend_column = mat.shape[0] - mat.shape[1]

    return np.pad(mat,((0,nb_extend_row),(0,nb_extend_column)),'constant', constant_values=(1, 1))
# not const
def setMultiIndexDayAndTime(df):
    # MultiIndex
    dateIndex = []
    timeIndex = []
    for timestamp in df.index:
        dateIndex += [timestamp.date()]
        timeIndex += [timestamp.time()]
    df.set_index([dateIndex, timeIndex], inplace=True)
    df.index.tickerNames = ['Day', 'Time']
# not const
def inverseMultiIndexedDf(df):
    dt = pd.DataFrame()
    for date, new_df in df.groupby(level=0):
        #print(new_df.index.get_level_values('time'))
        new_df.index = new_df.index.get_level_values('Time')
        new_df.columns = [str(date.date())]
        dt=pd.concat((dt,new_df),axis=1)
    return dt
# not const
def indexAndInterpolation(df):
    df=df.sort_index()
    indexMin = datetime.datetime.strptime(str(df.index.min()),"%H:%M:%S")
    indexMax = datetime.datetime.strptime(str(df.index.max()),"%H:%M:%S")

    df = df.reindex(pd.date_range(start=indexMin,
                                    end=indexMax,
                                    freq='1min').time)
    df = df.interpolate(method='linear',limit_direction='backward')
    return df

# not const
def splitDataframeByDay(df, method='ByColumn'):
    if method == 'ByArray':
        dfGrouped = df.groupby(pd.Grouper(freq='1D'))
        newDf=pd.DataFrame()
        for data in dfGrouped:
            newDf+=[data[1]]
            newDfFusion[str(data[0])]=data[1] 
    else:
        setMultiIndexDayAndTime(df)
        df=inverseMultiIndexedDf(df)
        df=indexAndInterpolation(df)
    return df

def findNextRange(serie, curI, k=1):
    outArray = [[-1,float('Infinity'),-1,-float('Infinity'),0] for i in range(k)] # [indexA, valueA, indexB, valueB, rangeValue]
    previousIInArray=curI
    for i in range(k):
        curMin=True
        maxLocalFound=False
        while (curI < serie.size and not maxLocalFound):
            if curMin:
                if serie[curI] < outArray[i][1]:
                    outArray[i][1] = serie[curI]
                    outArray[i][0] = serie.index[curI]
                else:
                    outArray[i][3] = serie[curI]
                    outArray[i][2] = serie.index[curI]
                    outArray[i][4] = outArray[i][3] - outArray[i][1]
                    curMin=False
                curI+=1
            else:
                if serie[curI] >= outArray[i][3]:
                    outArray[i][3] = serie[curI]
                    outArray[i][2] = serie.index[curI]
                    outArray[i][4] = outArray[i][3] - outArray[i][1]
                    curI+=1
                else:
                    maxLocalFound=True
    if previousIInArray==curI:
        curI+=1
    return curI,outArray

def insertInSortedRangeArray(minRange, value, iArray):  
    for i in range(len(minRange)):
        if minRange[i][1] > value:
            minRange.insert(i,[iArray,value])
            return minRange
        
    minRange+=[[iArray,value]]    
    return minRange

def updateInArray(minRange, value, iArray):
    minRange=insertInSortedRangeArray(minRange,value,iArray)
    for i in range(len(minRange)):
        if minRange[i][0] == iArray:
            del minRange[i]
            return minRange
        
def betterThanMinRange(array, minRange, iBeg, iEnd):
    tmpRange = array[iEnd][3]-array[iBeg][1]
    i=0
    while minRange[i][1] < tmpRange and i < len(minRange):
        if minRange[i][0] < iBeg or minRange[i][0] > iEnd:
            return True
        i+=1
    return False

def nextMinimumIndexOutsideBound(minRange, iBeg, iEnd):
    i=0
    while i < len(minRange):
        if minRange[i][0] < iBeg or minRange[i][0] > iEnd:
            return i
        i+=1
    return None

# From DARN github.com/marekgalovic/articles/blob/master/darn/utils.py
def z_score(x, mean, stddev):
    assert stddev != 0

    return (x - mean) / stddev


def inverse_z_score(x, mean, stddev):
    assert stddev != 0

    return x * stddev + mean

#'mean': float(raw['sum']) / raw['count'],
#'stddev': np.sqrt((float(raw['sum_sq']) / raw['count']) - ((float(raw['sum']) / raw['count']) ** 2)) ,
#'mean_log': float(raw['sum_log']) / raw['count'],
#'stddev_log': np.sqrt((float(raw['sum_sq_log']) / raw['count']) - ((float(raw['sum_log']) / raw['count']) ** 2))
#
def compare2linesDB(line1, line2):
    string = ""
    for column1, column2, value1, value2 in zip(line1["columns"],line2["columns"], line1["values"], line2["values"]):   
        if value1 != value2:
            string += f'''diff["{column1}"]:  {value1} to {value2}\n'''
            logMemoryIncrement("NbDiffsByColumnUpdateInfo")

    return string

# Convert a date without timezone in netry and return a date 
# converted without timezone according to timezone
def convertDateWithoutTimezone(date, timezone):
    date = timezone.localize(date)
    date = date.astimezone(pytz.timezone('Europe/Paris'))
    date = date.replace(tzinfo=None)
    return date

def getAllGrowingRange(array):
    outArray = []
    curI = 1
    isGrowing = False
    indexStart = 0 # declaration valeur d'initialisation n'a pas d'importance
    while (curI < len(array)):
        # Cas d'un d�but de range croissant
        if (not isGrowing and array[curI] >= array[curI-1]):
            indexStart = curI-1
            isGrowing = True
        # Cas d'une fin de range croissant
        elif (isGrowing and array[curI] < array[curI-1]):
            outArray += [[ indexStart, array[indexStart], curI-1, array[curI-1] ]]
            isGrowing = False
        # Cas d'une continuit� de range croissant
        #elif (isGrowing and array[curI] >= array[curI-1]):
            #pass
        # Cas d'une continuit� de range non croissant
        #elif (not isGrowing and array[curI] < array[curI-1]):
            #pass
        curI += 1
    if isGrowing:
        outArray += [[ indexStart, array[indexStart], curI-1, array[curI-1] ]]
    return outArray

# list = [(key, pair))]
# pair = (index, diff)
# dict = key: index in list
#ex list = [ (1,[1,2]), (2,[2,3])]
# dict [2:[2,3], 1:[1,2]]
class dicoSortedValue:
    def __init__(self):
        self.list = []
        self.dico = OrderedDict()

    def __init__(self, listIn):
        self.list = listIn
        self.dico = OrderedDict()
        for el in self.list:
            self.dico[el[0]] = el[1]

    def add(self, key, pair):
        index = self.getIndexInListPrivate(pair[1])
        self.dico[key] = pair
        self.list.insert(index, (key, pair))

    def removeByKey(self, key):
        value = self.dico[key][1]
        index = self.getIndexInListPrivate(value, key)

        del self.dico[key]
        del self.list[index]

    def getFirstKey(self):
        return self.list[0][0]

    def getFirstValue(self):
        return self.list[0][1][1]

    def getFirstIndex(self):
        return self.list[0][1][0]

    def getIndexByKey(self, key):
        return self.dico[key][0]

    def getValueByKey(self, key):
        return self.dico[key][1]

    def getSize(self):
        return len(self.list)

    # Dichotomy to get range where we have value
    def getIndexInListPrivate(self, value, key=0):
        low, mid, high = 0, 0, len(self.list) - 1
        found = False
        while low <= high and not found:
            mid = (low + high) // 2
            if value < self.list[mid][1][1]:
                high = mid - 1
            elif value > self.list[mid][1][1]:
                low = mid + 1
            else:
                found = True
        # Dans le cas not found
        if (low>mid or high<mid):
            mid = low

        # key==-1 pour l'ajout car peut importe ou on ajoute tant que dans la plage contenant la valeur
        if key==0:
            iKey = mid
        else:
            # on est tombe sur la valeur mais comme on cherche la clef exact de cette valeur faut continuer pour borner la recherche
            # Left
            leftRange = low
            rightIndex = mid
            # Right
            rightRange = high
            leftIndex = mid
            # Left
            while leftRange <= rightIndex:
                mid = (leftRange + rightIndex) // 2
                if value < self.list[mid][1][1]:
                    leftRange = mid + 1
                else: # value = ...
                    rightIndex = mid - 1
            # Right
            while leftIndex <= rightRange:
                mid = (leftIndex + rightRange) // 2
                if value > self.list[mid][1][1]:
                    rightRange = mid - 1
                else: # value = ...
                    leftIndex = mid + 1

            # we have (leftRange, rightRange)  self.list[leftRange][1] = self.list[rightRange][1] = value
            iKey = leftRange
            found = False
            while iKey <= rightRange and not found:
                if self.list[iKey][0] == key:
                    found = True
                else:
                    iKey+=1
            if not found:
                iKey = -1
        return iKey

# Example : [1,3,2,5,2,4]
#       x
#            x   
#   x       
#     x   x
# x
# 
# npArrayOfRange = [indA, val1, indB, valB, diff, 0/1/-1, iNext, key, iBefore]
# Entree:  npArrayObj = [1,3,2,5,2,4] k = [1,2,3,5]
# Sortie : res = [k=1 [indexA, indexB], k=2 [indexA, indexB]]
def getPotential(npArrayObj,k):
    # corrige moi la ligne ci dessiys
    indA = 0 # Indice d�but range
    valA = 1 # Valeur d�but range
    indB = 2 # Indice fin range
    valB = 3 # Valeur fin range
    diff = 4 # Diff�rence entre les deux valeurs
    iBMerge = 5 # Si le range peut �tre merg� avec le suivant -1 si supprim�, 0 si pas merg�, 1 si merg�
    iBefore = 6 # Indice du pr�c�dent
    iNext = 7 # Indice du suivant
    iKey = 8 # clef faisant la liaison avec la structure qui est tri�

    k = sorted(k, reverse=True)
    # k = sort(k) pour petite optimisation on le fait pas
    # 1/ R�cup�re une liste de tout les ranges croissants
    npArrayOfRange = getAllGrowingRange(npArrayObj)

    # Example: result = [[0,1,1,3],[2,2,3,5],[4,2,5,4]]
    for i in range(len(npArrayOfRange) - 1):
        if (npArrayOfRange[i+1][valA] > npArrayOfRange[i][valA]) and (npArrayOfRange[i+valA][valB] > npArrayOfRange[i][valB]):
            valDiff = ( (npArrayOfRange[i][valB] / npArrayOfRange[i][valA])  + (npArrayOfRange[i+1][valB] / npArrayOfRange[i+1][valA]) ) - (npArrayOfRange[i+1][valB] / npArrayOfRange[i][valA])
            npArrayOfRange[i] += [valDiff]
            npArrayOfRange[i] += [1] # type : Merge
        else: # pas de fusion
            npArrayOfRange[i] += [npArrayOfRange[i][valB] / npArrayOfRange[i][valA]]
            npArrayOfRange[i] += [0] # type : Delete
        npArrayOfRange[i] += [i-1]
        npArrayOfRange[i] += [i+1]
        npArrayOfRange[i] += [-1] # en attendant la clef


    # Last element
    if len(npArrayOfRange) > 0:
        npArrayOfRange[-1] += [npArrayOfRange[-1][valB] / npArrayOfRange[-1][valA]]
        npArrayOfRange[-1] += [0] # Delete
        npArrayOfRange[-1] += [len(npArrayOfRange) - indB] # len(npArrayOfRange) - 1 - 1 : dernier -1
        npArrayOfRange[-1] += [-1] # Delete on est a la fin donc pas d'index a mettre
        npArrayOfRange[-1] += [-1] # en attendant

    # Example: result = [[0,1,1,3, diff:1,type:1,iMerge:1],[2,2,3,5, diff:3,type:0,iMerge:-1],[4,2,iBMerge,4, diff:2,type:0,iMerge:-1]]

    #Lorsque vous it�rez sur sorted_list en utilisant enumerate, chaque �l�ment renvoy� par enumerate 
    #est un tuple contenant l'index et la valeur correspondante dans la liste d'origine. 
    #Dans ce cas, nous sommes uniquement int�ress�s par l'index, donc nous l'affectons � la variable i et 
    #nous ignorons la valeur correspondante en utilisant '_'.
    sorted_list = sorted(enumerate(npArrayOfRange), key=lambda x: x[1][diff])
    indexSortedArray = [i for i, _ in sorted_list]
    # resIndexSorted [0,2,1]

    uniqueKey = -1 # negatif pour ne pas confondre avec l'index dans la structure dico ordonn�

    # Initial - desynchronized list and dico but get the already sorted np array
    listTmp = []
    for i in range(len(indexSortedArray)):
        # On ajoute la valeur "diff" pour pouvoir ensuite ajouter a ce tableau les prochain elements
        listTmp += [(uniqueKey, [indexSortedArray[i], npArrayOfRange[indexSortedArray[i]][diff]])]

        npArrayOfRange[indexSortedArray[i]][iKey] = uniqueKey # peut etre a supprimer ou pas selon l'utilise de dicoNpArrayOfRange
        uniqueKey -= 1
    # listTmp = (-1, [0,1]), (-2, [2,2]), (-3, [1,3])]
    dicoIndexSorted = dicoSortedValue(listTmp)

    # result =  [[0,1,1,3, diff:1,type:1,iMerge:1,key:1], [2,2,3,5, diff:3,type:0,iMerge:-1,key:3], [4,2,5,4, diff:2,type:0,iMerge:-1,key:2]]
    #result index : [[0,1,1],  [2,3,2],  [1,2,3]]

    firstElementNotDeleted = 0
    resultIndex = []
    resultPercent = []
    resultPercentTotal = []
    curK = 0
    # Cr�er moi un tableau de 5 entiers
    
    while dicoIndexSorted.getSize() >= k[-1] and dicoIndexSorted.getSize() > 0:

        # Si on n'en a pas trouve suffisemment au d�but
        while dicoIndexSorted.getSize() < k[curK] : # and dicoIndexSorted.getSize() > 0 tester deja plus haut
            curK += 1
            resultIndex += [[]]
            resultPercent += [[]] #SI pas de valeur on met a vide
            resultPercentTotal += [[]] #SI pas de valeur on met a vide

        if dicoIndexSorted.getSize() == k[curK]:
            curResult = []
            curPercent = []
            i = firstElementNotDeleted
            while i!= -1 : # ou len(curResult) < k[curK]
                curElement = [npArrayOfRange[i][indA], npArrayOfRange[i][indB]]
                curResult += [curElement]
                curPercent += [(-1+npArrayObj[curElement[1]]/npArrayObj[curElement[0]])*100]
                i =  npArrayOfRange[i][iNext]
            percent=sum(curPercent)
            resultPercentTotal += [percent]
            resultIndex += [curResult]
            resultPercent += [curPercent]
            curK += 1

        if (dicoIndexSorted.getSize() <= k[-1]):
            break
        key = dicoIndexSorted.getFirstKey()

        indexOfWeakestRange = dicoIndexSorted.getIndexByKey(key)
        if npArrayOfRange[indexOfWeakestRange][iBMerge] == 1: # Merge
            indexRangeToMerge = npArrayOfRange[indexOfWeakestRange][iNext] # indexRangeToMerge est forcement != -1 car l'element ne serai pas marque comme mergeable
            npArrayOfRange[indexOfWeakestRange][indB] = npArrayOfRange[indexRangeToMerge][indB] # Copy indexB
            npArrayOfRange[indexOfWeakestRange][valB] = npArrayOfRange[indexRangeToMerge][valB] # Copy valueB
            # Si le range etait un range de merge
            nextRangeBis = npArrayOfRange[indexRangeToMerge][iNext] # On regarde quel �tait le prochain
            # si celui qu'on va supprimer pour cause de merge n'�tait pas mergeable notre range courant peut 
            # le devenir car on part de plus "bas" (indexA est plus faible que l'�tait le range de droite avant fusion)
            if npArrayOfRange[indexRangeToMerge][iBMerge] != 1: 
                if (nextRangeBis!= -1 and (npArrayOfRange[nextRangeBis][valA] > npArrayOfRange[indexOfWeakestRange][valA]) and (npArrayOfRange[nextRangeBis][valB] > npArrayOfRange[indexOfWeakestRange][valB])): # Forcement np[indexOfWeakestRange][valB] > np[nextRangeBis][valA] 
                    npArrayOfRange[indexOfWeakestRange][iBMerge] = 1 # continue a etre mergeable
                    valDiff = ( (npArrayOfRange[indexOfWeakestRange][valB] / npArrayOfRange[indexOfWeakestRange][valA])  + (npArrayOfRange[nextRangeBis][valB] / npArrayOfRange[nextRangeBis][valA]) ) - (npArrayOfRange[nextRangeBis][valB] / npArrayOfRange[indexOfWeakestRange][valA])
                    npArrayOfRange[indexOfWeakestRange][diff] = valDiff # continue a etre mergeable ecart de merge
                else:
                    npArrayOfRange[indexOfWeakestRange][iBMerge] = 0 # n'est plus mergeable
                    npArrayOfRange[indexOfWeakestRange][diff] = npArrayOfRange[indexOfWeakestRange][valB] / npArrayOfRange[indexOfWeakestRange][valA] # calcul nouvel ecart normal
            else: # Si le range �tait mergeable on le reste mais avec un nouveau calcul
                valDiff = ( (npArrayOfRange[indexOfWeakestRange][valB] / npArrayOfRange[indexOfWeakestRange][valA])  + (npArrayOfRange[nextRangeBis][valB] / npArrayOfRange[nextRangeBis][valA]) ) - (npArrayOfRange[nextRangeBis][valB] / npArrayOfRange[indexOfWeakestRange][valA])
                npArrayOfRange[indexOfWeakestRange][diff] = valDiff # continue a etre mergeable ecart de merge

            # MAJ de l'ecart precedent
            beforeRange = npArrayOfRange[indexOfWeakestRange][iBefore]
            if beforeRange >= 0:
                if ( (npArrayOfRange[indexOfWeakestRange][valA] > npArrayOfRange[beforeRange][valA]) and (npArrayOfRange[indexOfWeakestRange][valB] > npArrayOfRange[beforeRange][valB]) ):
                    npArrayOfRange[beforeRange][iBMerge] = 1 # continue a etre mergeable
                    npArrayOfRange[beforeRange][diff] = ( (npArrayOfRange[beforeRange][valB] / npArrayOfRange[beforeRange][valA])  + (npArrayOfRange[indexOfWeakestRange][valB] / npArrayOfRange[indexOfWeakestRange][valA]) ) - (npArrayOfRange[indexOfWeakestRange][valB] / npArrayOfRange[beforeRange][valA])
                    # on supprime l'element d'avant qui est mergeable et qui a chang� de valeur 
                    clefIndexMergeDelete = npArrayOfRange[beforeRange][iKey]
                    dicoIndexSorted.removeByKey(clefIndexMergeDelete)
                    # On le remet au bon endroit
                    # Il ne doit pas changer de key
                    dicoIndexSorted.add(clefIndexMergeDelete, (beforeRange, npArrayOfRange[beforeRange][diff]))
                
            npArrayOfRange[indexOfWeakestRange][iNext] = nextRangeBis # index du prochain �lement
            if nextRangeBis >= 0:
                npArrayOfRange[nextRangeBis][iBefore] = indexOfWeakestRange # index du pr�c�dent element sur le prochain element mergeable




            # Il reste le calcul de la clef et l'entretien du tableau tri�
            # on supprime le premier element tri� (par index)
            dicoIndexSorted.removeByKey(key)

            # on supprime l'element qui a �t� merg� cas unique du merge (par clef)
            clefIndexMergeDelete = npArrayOfRange[indexRangeToMerge][iKey]
            dicoIndexSorted.removeByKey(clefIndexMergeDelete)

            # On ajoute le nouvel element fusionn�
            dicoIndexSorted.add(uniqueKey, (indexOfWeakestRange, npArrayOfRange[indexOfWeakestRange][diff]))

            npArrayOfRange[indexOfWeakestRange][iKey] = uniqueKey # on associe la clef de l'index a ce tableau
            uniqueKey-=1

            npArrayOfRange[indexRangeToMerge][iBMerge] = -1 # To signal the element is deleted


        else: # Delete
            # Changer l'index du suivant du pr�c�dent
            # on saute l'element supprime
            indexPrevious = npArrayOfRange[indexOfWeakestRange][iBefore]
            indexNext = npArrayOfRange[indexOfWeakestRange][iNext]
            # on maj pour le prochain range l'index avant qui dois sauter celui qu'on vient de supprimer
            if indexNext >= 0:
                npArrayOfRange[indexNext][iBefore] = npArrayOfRange[indexOfWeakestRange][iBefore]
            #on maj le pr�c�dent qui doit sauter celui qu'on vient de supprimer
            if indexPrevious >= 0:
                npArrayOfRange[indexPrevious][iNext] = npArrayOfRange[indexOfWeakestRange][iNext]
                # Pour le pr�c�dent sa mergeabilit� peut changer
                if npArrayOfRange[indexNext][iBMerge] != -1: # Si pas supprimer
                    if (indexNext!= -1 and (npArrayOfRange[indexNext][valA] > npArrayOfRange[indexPrevious][valA]) and (npArrayOfRange[indexNext][valB] > npArrayOfRange[indexPrevious][valB])):
                        npArrayOfRange[indexPrevious][iBMerge] = 1 #  est mergeable
                        valDiff = ( (npArrayOfRange[indexPrevious][valB] / npArrayOfRange[indexPrevious][valA])  + (npArrayOfRange[indexNext][valB] / npArrayOfRange[indexNext][valA]) ) - (npArrayOfRange[indexNext][valB] / npArrayOfRange[indexPrevious][valA])
                        npArrayOfRange[indexPrevious][diff] = valDiff

                        # on supprime l'element d'avant qui est mergeable et qui a chang� de valeur 
                        clefIndexMergeDelete = npArrayOfRange[indexPrevious][iKey]
                        dicoIndexSorted.removeByKey(clefIndexMergeDelete)
                        # On le remet au bon endroit
                        # Il ne doit pas changer de key
                        dicoIndexSorted.add(clefIndexMergeDelete, (indexPrevious, npArrayOfRange[indexPrevious][diff]))
                        #uniqueKey-=1

                    else:
                        npArrayOfRange[indexPrevious][iBMerge] = 0 # n'est plus mergeable
                else:
                    npArrayOfRange[indexPrevious][iBMerge] = 0
            ## Que fait ces deux lignes ? 
            #key = dicoIndexSorted.getFirstKey() # Une clef a pu etre rajouter
            dicoIndexSorted.removeByKey(key)
            
            npArrayOfRange[indexOfWeakestRange][iBMerge] = -1 # To signal the element is deleted

            if firstElementNotDeleted == indexOfWeakestRange:
                firstElementNotDeleted = npArrayOfRange[indexOfWeakestRange][iNext]

    return resultIndex, resultPercent, resultPercentTotal

def round_time_to_nearest_minutes(time_obj, minutes=5):
    delta = datetime.timedelta(minutes=minutes)
    as_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    rounded_seconds = round(as_seconds / delta.total_seconds()) * delta.total_seconds()
    return datetime.time(hour=int(rounded_seconds // 3600),
                            minute=int((rounded_seconds % 3600) // 60),
                            second=int(rounded_seconds % 60))



def filter_and_interpolate(group, open_time, close_time):
    if group.empty:
        return pd.DataFrame()

    # Supprimer les doublons d'index pour permettre le reindex
    # Conserver la dernière occurrence si des timestamps sont dupliqués
    if group.index.has_duplicates:
        group = group[~group.index.duplicated(keep='last')]

    # S'assurer d'un index tri croissant
    if not group.index.is_monotonic_increasing:
        group = group.sort_index()

    # Check if there are any values between open_time and open_time + 1 hour
    open_time_td = pd.to_timedelta(str(open_time))
    open_time_plus_1_hour_td = open_time_td + pd.Timedelta(hours=1)
    open_time_plus_1_hour = (pd.Timestamp(0) + open_time_plus_1_hour_td).time()
    data_in_range = group.between_time(open_time, open_time_plus_1_hour)

    if data_in_range.empty:
        return pd.DataFrame()


    # Create a date_range with the open_time and close_time
    date_range = pd.date_range(group.index[0].normalize() + pd.to_timedelta(str(open_time)),
                               group.index[-1].normalize() + pd.to_timedelta(str(close_time)),
                               freq='1min')

    # Reindex the group with the new date_range, forward-fill and backward-fill missing values
    group = group.reindex(date_range).ffill().bfill()

    # Interpolate the data using linear method
    data_df = group.interpolate(method='linear')

    # Keep only data between open_time and close_time
    data_df = data_df.between_time(open_time, close_time)

    return data_df

def prepareData(shareObj, data_df, columns=['openPrice']):
    # Protéger contre les DataFrame vides ou colonnes manquantes
    if data_df is None or data_df.empty:
        return pd.DataFrame(columns=columns)
    # Ajouter les colonnes manquantes si besoin
    for column in columns:
        if column not in data_df.columns:
            data_df[column] = np.nan
    # Ne garder que les colonnes attendues
    data_df = data_df.loc[:, columns]
    for column in columns:
        data_df[column] = pd.to_numeric(data_df[column], errors='coerce')

    result_frames = []
    for _, group in data_df.groupby(pd.Grouper(freq='D')):
        res = filter_and_interpolate(group, open_time=shareObj.openRichMarketTime, close_time=shareObj.closeRichMarketTime)
        if res is not None and not res.empty:
            result_frames.append(res)

    if len(result_frames) > 0:
        return pd.concat(result_frames)
    else:
        # Return empty DataFrame with the expected columns to avoid concat warnings
        return pd.DataFrame(columns=columns)
