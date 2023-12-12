from os import name
import numpy as np
import pandas as pand
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import math

file_list = ['Instagram_Android_Data.csv','Instagram_Android_Data2.csv','Instagram_Android_Data3.csv','Instagram_Android_Data4.csv','Instagram_Android_Data5.csv',
             'Instagram_Android_Data6.csv','Instagram_Android_Data7.csv','Instagram_Android_Data8.csv','Instagram_Android_Data9.csv','Instagram_Android_Data10.csv']

dataFrame = pand.DataFrame(pand.read_csv(file_list[0]))

for i in range(1,len(file_list)):
    data = pand.DataFrame(pand.read_csv(file_list[i]))
    dataFrame = pand.concat([dataFrame,data])

#print(dataFrame)
    

proto = dataFrame.groupby(['Proto'])['Proto'].count().reset_index(name='Count').sort_values(['Count'],ascending = False)
print(proto)

dataFrame['totalTime'] = dataFrame.LastSeen - dataFrame.FirstSeen

timeFrame = dataFrame.sort_values(by='totalTime', ascending = False)
print(timeFrame[['Proto','totalTime']])


moreInfo = dataFrame.groupby(by = 'Proto')['BytesSent'].sum().reset_index(name = 'TotalBytesSent')

moreData = dataFrame.groupby(by = 'Proto')['BytesRcvd'].sum().reset_index(name = 'TotalBytesRcvd')
moreInfo = pand.concat([moreInfo,moreData])

moreData = dataFrame.groupby(by = 'Proto')['PktsSent'].sum().reset_index(name = 'TotalPktsSent')
moreInfo = pand.concat([moreInfo,moreData])

moreData = dataFrame.groupby(by = 'Proto')['PktsRcvd'].sum().reset_index(name = 'TotalPktsRcvd')
moreInfo = pand.concat([moreInfo,moreData])
moreInfo = moreInfo.groupby(by = 'Proto').sum()
# moreInfo['TotalKiloBytesRcvd'] = moreInfo['TotalKiloBytesRcvd'].div(1000)
#print(moreInfo.groupby(by = 'Proto').sum())

moreInfo.iloc[2] = moreInfo.iloc[2]/1000
moreInfo.rename(index = {'QUIC':'QUIC KB'}, inplace = True)
print(moreInfo)

sourcePort = dataFrame.groupby(by = 'SrcPort')['SrcPort'].count().reset_index(name='Count').sort_values(['Count'],ascending = False)
print(sourcePort)





#Data Graph/Machine Learning
#dimensionality reduction

#vals - {'Proto', 'totalTime', 'Proto', 'BytesSent', 'BytesRcvd', 'PktsSent', 'PktsRcvd'}

#Maps integer values for the protocols
Protocols = pand.unique(dataFrame.loc[:, 'Proto'].ravel())
Protocols = pand.Series(np.arange(len(Protocols)), Protocols)
#print(dataFrame.loc[:, 'Proto'].map(Protocols.get))

#To change what will be predicted, swap the a value in X and Y
dataX = dataFrame.loc[:, ['PktsSent', 'totalTime', 'BytesSent', 'BytesRcvd', 'PktsRcvd']]
dataX['Proto'] = dataFrame.loc[:, 'Proto'].map(Protocols.get)
dataY = dataFrame.loc[:, ['Proto']]

xTrain, xTest, yTrain, yTest = train_test_split(dataX, dataY, test_size=0.2, random_state=1)

clf = GaussianNB()
clf = clf.fit(xTest, yTest)

yPrediction = clf.predict(xTest)
ConfusionMatrixDisplay.from_predictions(yTest, yPrediction)

print("accuracy_score =", accuracy_score(yTest, yPrediction))


#plots

#Data from line 25
plt.figure(2)
plt.title("Percentages of Protocols Used")
plt.pie(proto['Count'], labels = proto['Proto'], autopct = '%1.1f%%')


#data from line 29
plt.figure(3)
plt.title("Timeframe of each Protocol")
plt.ylabel("Time(1e6 seconds)")
plt.xlabel("Protocol")
plt.scatter(timeFrame['Proto'], timeFrame['totalTime'])


#data from line 48
plt.figure(4)
plt.title("Sourceports and Number of Occurances")
plt.xlabel("Sourceport Number")
plt.ylabel("Occurance Count")
plt.bar(sourcePort['SrcPort'], sourcePort['Count'], width=60)
#plt.scatter(x= sourcePort.reset_index().index, y= sourcePort['SrcPort'])   NOT IN USE



#data from line 52
moreInfo.plot.bar(subplots= True, figsize= (5,9))

plt.show()