import numpy as np 

def RMSErrors (ypred, ydata): 
    ydata = np.array (ydata)
    ypred = np.array (ypred)

    RMSError = np.sqrt ( (np.sum (ypred - ydata)**2) / len (ypred) )
    return RMSError

def MAPE (ypred, ydata): 
    ydata = np.array (ydata)
    ypred = np.array (ypred)

    MAPError = np.sum ( np.abs (ypred - ydata) / ydata)
    return MAPError

def AkaikeInformationCriterion_c (ypred): 
    n, k  = np.shape (ypred)
    l_max = np.max (ypred)
    correction_term = (2*k**2 + 2*k) / (n - k - 1)
    AICc  = 2*k - 2*np.log (l_max) + correction_term
    return AICc

def BayesianInformationCriterion (ypred): 
    n, k = np.shape (ypred)
    l_max = np.max (ypred)
    BIC = np.log (n)*k - 2*np.log (l_max)
    return BIC 

def SSE (ypred, ydata): 
    return np.sqrt (np.sum(ypred - ydata)**2)

def PRESS (ypred, ydata): 
    p = 3
    N = len (ypred)
    PRESS_stat = SSE (ypred, ydata) * (1 + 2*(p/N)) / N
    return PRESS_stat

def StructuralRiskMinimisation (ypred, ydata): 
    p = 3
    N = len (ypred)
    pn = p/N
    denom = 1 - np.sqrt (abs(pn - pn*np.log(pn) + np.log(pn)/2*N))
    SRM = SSE (ypred, ydata) / (N * denom)
    return SRM 

def FinalPredictionError (ypred, ydata): 
    p = 3
    N = len (ypred)
    FPE = SSE (ypred, ydata) * ( (N+p)/(N-p) ) / N
    return FPE
    