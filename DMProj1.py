import pandas as pd
import numpy as np
import pywt
from scipy import signal
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelextrema
from pylab import *



CGMTimeseries=[]
Cgmvalues=[]

def getdata(Cgmvalues,CGMTimeseries):
    for i in range(5):
        CGMTimeseries.append(pd.read_csv(f"CGMDatenumLunchPat{i + 1}.csv", delimiter=",", dtype=np.float64))
        Cgmvalues.append(pd.read_csv(f"CGMSeriesLunchPat{i + 1}.csv", delimiter=",", dtype=np.float32))
    print("getdata fun running")

def pre_processing(Cgmvalues,CGMTimeseries):
    for i in range(5):
        Cgmvalues[i] = Cgmvalues[i].interpolate(direction="both",method="linear",axis=0) 
        CGMTimeseries[i] = CGMTimeseries[i].interpolate(direction="both",method="linear",axis=0)
    
    for i in range(5):
        Cgmvalues[i] = Cgmvalues[i].dropna()
        CGMTimeseries[i] = CGMTimeseries[i].dropna()

    for i in range(5):
        Cgmvalues[i] = Cgmvalues[i].to_numpy()
        CGMTimeseries[i] = CGMTimeseries[i].to_numpy()
    print("pre-processing fun running")


# 1st Feature
def fourier_transform(Cgmvalues,CGMTimeseries):
    fourier_transform=list()
    for i in range(5):
        person_ft=list()
        for r in range(len(Cgmvalues[i])):
            temp=np.fft.irfft(Cgmvalues[i][r],n=2)
            #print(temp)
            person_ft.append(temp)
        fourier_transform.append(person_ft)
    print("fourierTransform-Completed")
    return np.array(fourier_transform)

def fftplot(FFT):
    for i in range(5):
        graph_FFT = pd.DataFrame(FFT[i])
        graph_FFT = graph_FFT.apply(abs)

        plt.plot(graph_FFT, color="blue")
        plt.show()


# 2nd feature

def polyFit(Cgmvalues,CGMTimeseries):
    polylist =[]
    for i in range(5):
        yval = Cgmvalues[i].flatten()
        xval = CGMTimeseries[i].flatten()
        polylist.append(np.polyfit(xval,yval,2))
    print("polyFit running")
    return np.array(polylist)


def pltpft(PFT):
    #%matplotlib inline
    for i in range(5):
        graph = np.polyval(PFT[i], CGMTimeseries[i])
        plt.plot(graph, color="red")
        plt.show()
    print("plot running")


#3rd feature
def MOV(Cgmvalues,CGMTimeseries):
    mov = []
    for i in range(5):
      personCgmvalues = Cgmvalues[i]
      personCGMTimeseries = CGMTimeseries[i]
      person_MOV = []
      window = 4
      for row_i in range(len(personCgmvalues)):
        person_MOVpoint = []
        for col_i in range(window, len(personCgmvalues[row_i])):
          person_MOVpoint.append(
              sum(personCgmvalues[row_i][col_i - window : col_i + 1]) / window)
        person_MOV.append(person_MOVpoint)
      mov.append(person_MOV)
    return np.array(mov)


def pltMov(Mov):
    for j in range(5):
        for i in range(0, len(Mov[j]), 2):
            plt.plot(Mov[j][i], color="green")
        plt.show()




#4th feature
def rootMeanSquare(Cgmvalues,CGMTimeseries):
    rmsvelocity=list()
    for i in range(5):
        personCgmvalues=Cgmvalues[i]
        personCGMTimeseries=CGMTimeseries[i]
        rmsperson=list()

        for r in range(len(personCgmvalues)):
            rmsVelWin=list()
            for c in range(0,len(personCgmvalues[i]),4):
                if c+4 <len(personCgmvalues[i]):
                    temp=sum(personCgmvalues[r][c:c+4])/4
                    rmsVelWin.append(np.sqrt(temp))
                else:
                    temp=sum(personCgmvalues[r][c:])/(len(personCgmvalues[i])-c)
                    rmsVelWin.append(np.sqrt(temp))
            rmsperson.append(rmsVelWin)
        rmsvelocity.append(rmsperson)
    print("rms runniung")
    return np.array(rmsvelocity)


def pltrms(RMS):
    for j in range(5):
        for i in range(2, len(RMS[j]), 10):
            plt.plot(RMS[j][i], color="black")
        plt.show()



def featureMatrix(PFT,Mov,FFT,RMS):
    feature_MOV=pd.DataFrame()
    for personLOC in Mov:
        feature_MOV = feature_MOV.append(personLOC, ignore_index=True)
    feature_FFT = pd.DataFrame()
    for personFFTS in FFT:
        feature_FFT = feature_FFT.append(personFFTS, ignore_index=True)
        feature_FFT = feature_FFT.apply(abs)
    featureRMS = pd.DataFrame()
    for personRMS in RMS:
        featureRMS = featureRMS.append(personRMS, ignore_index=True)
    featurePolynomial = pd.DataFrame()
    for personPoly in PFT:
        polynomial_person_series = pd.Series(personPoly)
        featurePolynomial = featurePolynomial.append(polynomial_person_series, ignore_index=True)
    featurematrix = pd.concat((feature_MOV, feature_FFT, featureRMS, featurePolynomial), axis=1, ignore_index=True)
    print(featurematrix.shape)
    featurematrix = featurematrix.dropna(axis=1)
    print(featurematrix.shape)
    print("featureMatrix-Completed")

    return featurematrix



def pca(featureMatrix):
    pca = PCA(n_components=5)
    pca.fit(featureMatrix)
    print("explained variance ratio", pca.explained_variance_ratio_)
    pca_df = pd.DataFrame(pca.components_, index = ['PC-1','PC-2', 'PC-3', 'PC-4', 'PC-5'])
    print(pca_df)
    plt.bar(['PC-1','PC-2', 'PC-3', 'PC-4', 'PC-5'], pca.explained_variance_ratio_)
    plt.show()

#calling getdata
getdata(Cgmvalues,CGMTimeseries)
print(Cgmvalues[0].head)
#calling pre_processing
pre_processing(Cgmvalues,CGMTimeseries)
#calling FFT(1st feature)
FFT=fourier_transform(Cgmvalues,CGMTimeseries)
#calling plot of fft
fftplot(FFT)
#calling PFT(2nd feature)
PFT=polyFit(Cgmvalues,CGMTimeseries)
#calling plot of PFT
pltpft(PFT)
#calling MOV(3rd feature)
Mov=MOV(Cgmvalues,CGMTimeseries)
#calling plot of MOv
pltMov(Mov)
#calling RMS(4th feature)
RMS= rootMeanSquare(Cgmvalues,CGMTimeseries)
#calling plot of RMS
pltrms(RMS)
#calling FeatureMatrix
FeatureMatrix=featureMatrix(PFT,Mov,FFT,RMS)
print(FeatureMatrix.head)
#calling pca
pca(FeatureMatrix)


