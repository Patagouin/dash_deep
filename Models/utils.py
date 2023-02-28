import datetime
import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
import yfinance as yf
import pytz
import seaborn
import numpy as np

logType = {"Success":0, "Warning":1, "Error":2}

nbSecondsInWeek = 604800

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
    with open(f'{__file__}/../../../data/logFile_{nameDay}.log', 'a+', newline='') as logFile:
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
    for date in listDate:
        dfSlice = curTicker.history(interval='1m',start=date[0],end=date[1], rounding=False, actions=True, prepost=True)
        df = pd.concat([df, dfSlice])
        if beginDate == 'max':  #TODO UPDATE YFINANCE TO NOT HAVE TO CHECK THAT !!
            beginDateTmp = datetime.datetime.now()-datetime.timedelta(days=30, minutes=-1)
            if df.size > 0 and df.iloc[0].name.replace(tzinfo=None) < beginDateTmp:
                df=df.drop(df.iloc[0].name)
        elif df.size > 0 and df.iloc[0].name.replace(tzinfo=None) < beginDate:  #TODO UPDATE YFINANCE TO NOT HAVE TO CHECK THAT !!
            df=df.drop(df.iloc[0].name)
    return df

def downloadDataFromYahooByTickerName(tickerName, beginDate='max', endDate='now'):
    df = pd.DataFrame()
    listDate = getListOfDateForDownload(beginDate, endDate)

    curTicker = yf.Ticker(tickerName)
    for date in listDate:
        dfSlice = curTicker.history(interval='1m',start=date[0],end=date[1], rounding=False, actions=True, prepost=True)
        df = pd.concat([df, dfSlice])
    return df

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
        df = df.fillna(method=method)
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


#def assembleDataFramesByColumn(listDfData, column, dfShare=pd.DataFrame(), listNames=None):
#    #
#    minIndexShared = listDfData[0].index.min()
#    maxIndexShared = listDfData[0].index.max()
#    dfOut = pd.DataFrame()
#    cpt=0
#    for i, curDf in enumerate(listDfData):
#        minIndexShared = max(minIndexShared, curDf.index.min())
#        maxIndexShared = min(maxIndexShared, curDf.index.max())
#        curDf_reindexed = curDf.reindex(pd.date_range(start=minIndexShared,
#                                                end=maxIndexShared,
#                                                freq='1min'))
#        curDf_interpolated = curDf_reindexed.interpolate(method='linear')
#        if listNames != None:
#            dfOut[f'{listNames[i]}'] = curDf_interpolated[column]
#        else:
#            dfOut[f'{dfShare.tickerName.iloc[cpt]}'] = curDf_interpolated[column]
#        cpt += 1

#    return dfOut

        
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

# Return: outArray = [indexA, valueA, indexB, valueB, rangeValue] * kRange
# and percent [float number]
def getPotential(serie, kRange, sens='up'):
    # outArray = [indexA, valueA, indexB, valueB, rangeValue] * kRange (+1) for computing 
    outArray = [[-1,float('Infinity'),-1,float('Infinity'),0] for x in range(kRange+1)]
    #1/ Fill all value of array with increase range
    iInArray=0
    # minRange array store index of outArray and value in sorted order of value
    minRange=[]
    iInArray,tmp=findNextRange(serie,iInArray, k=kRange+1) # increment iInArray
    outArray=tmp
    
    for i in range(len(outArray)-1):
        minRange = insertInSortedRangeArray(minRange,outArray[i][4],i)
    # we sort by rangeValue and split We split the end index (which is use to compare with the rest)
    #minRange=sorted(len(minRange), key=lambda x:x[1])
   
    while iInArray < serie.size + 1: # +1 because we do iInArray==serie.size
        #Check if range of the last element is better than the min of minRange array
        previousIInArray=iInArray
        if minRange[0][0]==len(outArray)-2: # -1 because last element index and -1 because the last element is reserved for temporary
            if outArray[len(outArray)-1][3] > outArray[len(outArray)-2][3]:
                outArray[len(outArray)-2][2] = outArray[len(outArray)-1][2]
                outArray[len(outArray)-2][3] = outArray[len(outArray)-1][3]
                outArray[len(outArray)-2][4] = outArray[len(outArray)-2][3] - outArray[len(outArray)-2][1]
                del outArray[len(outArray)-1]
                minRange=updateInArray(minRange, outArray[len(outArray)-1][4], len(outArray)-1)
                iInArray,tmp=findNextRange(serie,iInArray, k=1)
                outArray+=tmp
                
            else: # can't merge lastElement of outArray
                # else: Nothing to do [tmp[0][1],tmp[0][3]] is include in outArray[len(outArray)-1] range
                iInArray,tmp=findNextRange(serie,iInArray, k=1) 
                #we update minRange
                if tmp[0][1] <= outArray[len(outArray)-1][1]: # For shortest range, can be also tmp[0][3] > outArray[len(outArray)-1][3]
                    outArray[len(outArray)-1] = tmp[0]
                elif tmp[0][3] > outArray[len(outArray)-1][3]:
                    outArray[len(outArray)-1][2] = tmp[0][2]
                    outArray[len(outArray)-1][3] = tmp[0][3]
                    outArray[len(outArray)-1][4] = outArray[len(outArray)-1][3] - outArray[len(outArray)-1][1]

        if outArray[len(outArray)-1][4] > minRange[0][1]:
            # Check if the minRange can be merge with the next ones or previous ones
            # Previous
            iEndRangePrev=minRange[0][0]
            iBeginRangePrev=iEndRangePrev-1
            found=False
            while not found and iBeginRangePrev>=0 and \
            outArray[iBeginRangePrev][1]<outArray[iEndRangePrev][1] and \
            outArray[iBeginRangePrev][3]<outArray[iEndRangePrev][3]:
                if outArray[iEndRangePrev][3]-outArray[iBeginRangePrev][1]>outArray[len(outArray)-1][4] \
                or betterThanMinRange(outArray, minRange, iBeginRangePrev, iEndRangePrev):
                    found=True
                else:
                    iBeginRangePrev-=1

            if not found:
                iBeginRangePrev = iEndRangePrev
            tmpRangePrev=outArray[iEndRangePrev][3]-outArray[iBeginRangePrev][1]
            # Next
            iBeginRangeNext=minRange[0][0]
            iEndRangeNext=iBeginRangeNext+1
            found=False
            while not found and iEndRangeNext<len(outArray)-1 and \
            outArray[iBeginRangeNext][1]<outArray[iEndRangeNext][1] and \
            outArray[iBeginRangeNext][3]<outArray[iEndRangeNext][3] :
                    if outArray[iEndRangeNext][3]-outArray[iBeginRangeNext][1]>outArray[len(outArray)-1][4] \
                    or betterThanMinRange(outArray, minRange, iBeginRangePrev, iEndRangePrev):
                        found=True
                    else:
                        iEndRangeNext+=1
            if not found:
                iEndRangeNext = iBeginRangeNext
            tmpRangeNext=outArray[iEndRangeNext][3]-outArray[iBeginRangeNext][1]
            # If become better que new range
            tmpRange=None
            iBeginRange=0
            iEndRange=0
            if tmpRangePrev>tmpRangeNext or \
            (tmpRangePrev==tmpRangeNext and \
             iEndRangePrev-iBeginRangePrev<=iEndRangeNext-iBeginRangeNext) :
                
                iBeginRange=iBeginRangePrev
                iEndRange=iEndRangePrev
                tmpRange=tmpRangePrev
            else:
                iBeginRange=iBeginRangeNext
                iEndRange=iEndRangeNext
                tmpRange=tmpRangeNext
            if tmpRange > outArray[len(outArray)-1][4] \
            or (nextMinimumIndexOutsideBound(minRange, iBeginRange, iEndRange)!= None and minRange[i][1] < tmpRange):
                outArray[iBeginRange][3] = outArray[iEndRange][3]
                outArray[iBeginRange][2] = outArray[iEndRange][2]
                outArray[iBeginRange][4] = outArray[iEndRange][3]-outArray[iBeginRange][1] # = tmpRange
                # Delete ranges except the merged ones 
                del outArray[iBeginRange+1:iEndRange+1]

                # find next range
                iInArray,tmp=findNextRange(serie,iInArray, k=iEndRange-iBeginRange)
                outArray+=tmp

                for i in range(len(minRange)-1,-1,-1):
                    if (minRange[i][0] >= iBeginRange and minRange[i][0] < iEndRange+1):
                        del minRange[i]
                    elif (minRange[i][0] > iEndRange):
                        minRange[i][0] = minRange[i][0] - (iEndRange-iBeginRange)
                minRange=insertInSortedRangeArray(minRange,outArray[iBeginRange][4],iBeginRange)
                for i in range(iEndRange-iBeginRange):
                    minRange=insertInSortedRangeArray(minRange,outArray[len(outArray)-2-i][4],len(outArray)-2-i)
            else: # If not become better que new range
                del outArray[minRange[0][0]] # Index of element before the attempt to merge
                iInArray,tmp=findNextRange(serie,iInArray, k=1)
                outArray+=tmp
                for i in range(len(minRange)):
                    if minRange[i][0] > iEndRange:
                        minRange[i][0] = minRange[i][0] - 1
                del minRange[0]
                minRange=insertInSortedRangeArray(minRange,outArray[len(outArray)-2][4],len(outArray)-2)


        else: #if last (tmp) range is lowest than the previous ones
            # TODO si le nouveau tmp est inferieur au minRange alors tenter de merge sur le dernier element outArray
            if outArray[len(outArray)-1][3] > outArray[len(outArray)-2][3]:
                outArray[len(outArray)-2][2] = outArray[len(outArray)-1][2]
                outArray[len(outArray)-2][3] = outArray[len(outArray)-1][3]
                outArray[len(outArray)-2][4] = outArray[len(outArray)-2][3] - outArray[len(outArray)-2][1]
                del outArray[len(outArray)-1]
                minRange=updateInArray(minRange, outArray[len(outArray)-1][4], len(outArray)-1)
                iInArray,tmp=findNextRange(serie,iInArray, k=1)
                outArray+=tmp
            else: # can't merge lastElement of outArray
                
                # else: Nothing to do [tmp[0][1],tmp[0][3]] is include in outArray[len(outArray)-1] range
                iInArray,tmp=findNextRange(serie,iInArray, k=1) 
                #we update minRange
                if tmp[0][1] <= outArray[len(outArray)-1][1]: # For shortest range, can be also tmp[0][3] > outArray[len(outArray)-1][3]
                    outArray[len(outArray)-1] = tmp[0]
                elif tmp[0][3] > outArray[len(outArray)-1][3]:
                    outArray[len(outArray)-1][2] = tmp[0][2]
                    outArray[len(outArray)-1][3] = tmp[0][3]
                    outArray[len(outArray)-1][4] = outArray[len(outArray)-1][3] - outArray[len(outArray)-1][1]

        
    # Remove the last element which was used as tmp
    outArray = outArray[:-1]
    # Remove the 0 rangeValue at the end
    i=len(outArray)-1
    while i>=0 and outArray[i][4]==0:
        i-=1
    outArray = outArray[:i+1]
    percent=sum(-1+outArray[i][3]/outArray[i][1] for i in range(len(outArray)))
    percent*=100
    return outArray, percent

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
