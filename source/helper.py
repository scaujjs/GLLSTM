import numpy as np
import random
import os
from keras.layers import Lambda,merge,concatenate





def generateTrainingData():
    lengthOfSubseq = 101
    numOfCys = 25
    halfwin = lengthOfSubseq // 2
    numOfInfo = 24

    def generateSubexample(seq, pssm, pairs):

        def checkInBonds(index):
            found = 0
            for p1, p2 in pairs:
                if index == p1 - 1 or index == p2 - 1:
                    found = 1
                    break
            return found

        X = np.zeros((numOfCys, lengthOfSubseq, numOfInfo), np.float32)
        T = np.zeros((numOfCys, 2), np.float32)

        index = 0
        cysinaline = 0
        for i in range(len(seq)):
            if seq[i] == "C":
                cysinaline += 1
        ##print(cysinaline)
        if cysinaline > numOfCys or len(seq) > 2000:
            return None, None
        errorFlag = 0
        for i in range(len(seq)):
            if seq[i] == 'C':

                if checkInBonds(i):
                    T[index] = [0., 1.]
                else:
                    T[index] = [1., 0.]

                for j in range(lengthOfSubseq):
                    tempindex = i - halfwin + j
                    if tempindex < 0 or tempindex > len(seq) - 1:
                        X[index, j, 0:20] = np.zeros(20, np.float32)

                    else:
                        if seq[tempindex] not in dict_Hydro:
                            errorFlag = 1
                            break
                        X[index, j, 0:20] = pssm[tempindex]
                        X[index, j, 22] = dict_Hydro[seq[tempindex]]
                        X[index, j, 23] = dict_polarnorm[seq[tempindex]]
                    X[index, j, 20] = tempindex / 10000.0
                    X[index, j, 21] = (len(seq) - tempindex) / 10000.0

                index += 1
        if errorFlag:
            return None, None
        return X, T

    Xraw = np.load('../resource/pssmProteins.npy')
    listOfSeq, listOfPairs = loadSeqsAndBonds()
    ##print(listOfPairs.items())
    listOfIdentity = list()
    for line in open('../resource/nameProteins'):
        listOfIdentity.append(line.strip())

    examples = list()
    for i in range(len(listOfIdentity)):
        seq = listOfSeq[listOfIdentity[i]]
        pair = listOfPairs[listOfIdentity[i]]
        Xi, Ti = generateSubexample(seq, Xraw[i], pair)
        if Xi is None:
            continue

        examples.append((Xi, Ti))
    for i in range(200):
        random.shuffle(examples)
    X = np.zeros((len(examples), numOfCys, lengthOfSubseq, numOfInfo), np.float32)
    T = np.zeros((len(examples), numOfCys, 2), np.float32)

    for i in range(len(examples)):
        X[i] = examples[i][0]
        T[i] = examples[i][1]

    if os.path.exists('../resource/GLBlstmTrain/'):
        pass
    else:
        os.mkdir('../resource/GLBlstmTrain/')
    if os.path.exists('../MODEL/keras/'):
        pass
    else:
        os.mkdir('../MODEL/keras/')
    np.save('../resource/GLBlstmTrain/X.data', X)
    np.save('../resource/GLBlstmTrain/T.data', T)

    print('finished')


dict_polarnorm = {'Q':-0.40766,'R':-0.45766,'H':0.14689,'M':0.50144,'A':-0.10311,'F':0.50144,
              'I':0.51962,'V':0.41962,'E':-0.42584,'L':0.47416,'C':0.54235,'N':-0.29402,
              'Y':0.07416,'D':-0.43947,'W':0.32416,'G':-0.26220,'K':-0.61220,'T':-0.19856,
'S':-0.30311,'P':0,'X':0}

dict_Hydro={'T':-0.01124, 'V': 0.00685, 'R': -0.18437, 'C': 0.08695, 'P': -0.09134, 'F': 0.31693,
                'D': -0.29289, 'Q': -0.12494, 'W': 0.50297, 'K': -0.23088, 'Y': 0.26783,'M':0.08437,
                'E':-0.49703,'A':-0.01899,'I':0.10504,'S':-0.00866,'N':-0.08359,'H':-0.01899,'L':0.16964,
'G':0.02235,'X':0}

## this function read sequences and bonds file, and return two dictory.
## this function is used when generating data

def loadSeqsAndBonds():
    listOfSeq=dict()
    listOfPairs=dict()

    for line in open('../resource/sequences'):
        key,seq=line.strip().split(',')
        listOfSeq[key]=seq


    for line in open('../resource/bonds'):
        items=line.strip().split(',')
        pairs=list()
        for i in range(len(items)-1):
            strpair=items[i+1].split('_')
            pair=(int(strpair[0]),int(strpair[1]))
            pairs.append(pair)

        listOfPairs[items[0]]=pairs

    return(listOfSeq,listOfPairs)


## this funtion will called in Metrics to evaluate the accuracy on protein level.
def acc_on_protein(y_true, y_pred):

    emptyPostion=tf.logical_not(K.cast(tf.reduce_sum(y_true,-1),tf.bool))

    resultOnResidue=K.equal(K.argmax(y_true, axis=-1),
            K.argmax(y_pred, axis=-1))
    resultOnResidue=tf.logical_or(emptyPostion,resultOnResidue)

    resultOnResidue=K.cast(resultOnResidue,tf.float32)
    resultOnprotein=tf.reduce_prod(resultOnResidue,-1)
    ##K.mean(resultOnprotein)
    return K.mean(resultOnprotein)

## reshape the traindata to meet requirement of BLSTM
def convertTrainingData(Xs,Ts):
        example=list()
        for i in range(Xs.shape[0]):
            for j in range(Ts.shape[1]):
                if round(Ts[i][j][0])==0 and round(Ts[i][j][1])==0:
                    break
                else:
                    ##print(Xs[i][j])
                    example.append((Xs[i][j],Ts[i][j]))
        Xs=np.zeros((len(example),Xs.shape[2],Xs.shape[3]),np.float32)
        Ts=np.zeros((len(example),2),np.float32)
        random.shuffle(example)
        for i in range(len(example)):
            Xs[i]=example[i][0]
            Ts[i]=example[i][1]


        return Xs,Ts

def crop():
	def func(x):
		returnResult=list()
		for i in range(x.shape[1]):

			returnResult.append((x[:,i]))
		return returnResult
	return Lambda(func)

## this evaluate function take target of state and predicted state as input
def evaluateResult(T_t,P_t):
    import math
    totalCys = 0
    correctOnr = 0
    fullCorrect = 0
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(T_t.shape[0]):
        correct=1
        empty=1
        for j in range(T_t.shape[1]):
            if round(T_t[i][j][0])==1 or round((T_t[i][j][1]))==1:
                totalCys+=1

            if round(T_t[i][j][0])==0 and round((T_t[i][j][1]))==0:
                ##print(T_t[i][j])
                break
            else:
                empty=0
                if  P_t[i][j].argmax(axis=-1)!=T_t[i][j].argmax(axis=-1):
                    correct=0
                else:
                    correctOnr+=1

                if P_t[i][j].argmax(axis=-1)==1 and T_t[i][j].argmax(axis=-1)==1:
                    tp+=1
                elif P_t[i][j].argmax(axis=-1)==0 and T_t[i][j].argmax(axis=-1)==0:
                    tn+=1
                elif P_t[i][j].argmax(axis=-1)==1 and T_t[i][j].argmax(axis=-1)==0:
                    fn+=1
                elif P_t[i][j].argmax(axis=-1)==0 and T_t[i][j].argmax(axis=-1)==1:
                    fp+=1
        if correct:
            fullCorrect+=1
        assert not empty
    if 1:
        print(tp,fp,tn,fn)
    if (tp+fn)==0:
        sensitivity=-1
    else:
        sensitivity=tp*1.0/(tp+fn)
    if (tn+fp)==0:
        specifity=-1
    else:
        specifity=tn*1.0/(tn+fp)
    accOnCys=correctOnr/1.0/totalCys
    accOnPro=(fullCorrect*1.0)/(P_t.shape[0])
    if math.sqrt((tp+fn)*(tp+fp)*(tn+fp)*(tn+fn))==0:
        mcc=-1
    else:
        mcc=(tp*tn-fp*fn)/math.sqrt((tp+fn)*(tp+fp)*(tn+fp)*(tn+fn))
    totalNumOfPro=P_t.shape[0]

    return ( accOnCys, accOnPro,sensitivity, specifity, mcc,totalNumOfPro,totalCys)

## this is an evaluate function after forcing the number of predicted oxidized cysteines to be even
def adjustEvaluateResult(T_t,P_t):
    import math
    totalCys = 0
    correctOnr = 0
    fullCorrect = 0
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(T_t.shape[0]):
        correct=1
        empty=1
        cysodd=0
        for j in range(T_t.shape[1]):
            if round(T_t[i][j][0])==0 and round((T_t[i][j][1]))==0:
                ##print(T_t[i][j])
                break
            if P_t[i][j].argmax(axis=-1) == 1:
                cysodd = 1 - cysodd
        if cysodd:
            suspect=0
            minValue=100
            for j in range(T_t.shape[1]):
                if round(T_t[i][j][0]) == 0 and round((T_t[i][j][1])) == 0:
                    ##print(T_t[i][j])
                    break
                if abs(P_t[i][j][0]-P_t[i][j][1])<minValue:
                    minValue=abs(P_t[i][j][0]-P_t[i][j][1])
                    suspect=j

            P_t[i][suspect][P_t[i][suspect].argmax(axis=-1)]=0



        for j in range(T_t.shape[1]):
            if round(T_t[i][j][0])==1 or round((T_t[i][j][1]))==1:
                totalCys+=1

            if round(T_t[i][j][0])==0 and round((T_t[i][j][1]))==0:
                ##print(T_t[i][j])
                break
            else:
                empty=0

                if  P_t[i][j].argmax(axis=-1)!=T_t[i][j].argmax(axis=-1):
                    correct=0
                else:
                    correctOnr+=1

                if P_t[i][j].argmax(axis=-1)==1 and T_t[i][j].argmax(axis=-1)==1:
                    tp+=1
                elif P_t[i][j].argmax(axis=-1)==0 and T_t[i][j].argmax(axis=-1)==0:
                    tn+=1
                elif P_t[i][j].argmax(axis=-1)==1 and T_t[i][j].argmax(axis=-1)==0:
                    fn+=1
                elif P_t[i][j].argmax(axis=-1)==0 and T_t[i][j].argmax(axis=-1)==1:
                    fp+=1

        if correct:
            fullCorrect+=1
        assert not empty
    if 1:
        print(tp,fp,tn,fn)
    if (tp+fn)==0:
        sensitivity=-1
    else:
        sensitivity=tp*1.0/(tp+fn)
    if (tn+fp)==0:
        specifity=-1
    else:
        specifity=tn*1.0/(tn+fp)
    accOnCys=correctOnr/1.0/totalCys
    accOnPro=(fullCorrect*1.0)/(P_t.shape[0])
    if math.sqrt((tp+fn)*(tp+fp)*(tn+fp)*(tn+fn))==0:
        mcc=-1
    else:
        mcc=(tp*tn-fp*fn)/math.sqrt((tp+fn)*(tp+fp)*(tn+fp)*(tn+fn))
    totalNumOfPro=P_t.shape[0]

    return ( accOnCys, accOnPro,sensitivity, specifity, mcc,totalNumOfPro,totalCys)
