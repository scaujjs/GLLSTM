import numpy as np
import random
import os
from keras.layers import Lambda,merge,concatenate

dict_polarnorm = {'Q':-0.40766,'R':-0.45766,'H':0.14689,'M':0.50144,'A':-0.10311,'F':0.50144,
              'I':0.51962,'V':0.41962,'E':-0.42584,'L':0.47416,'C':0.54235,'N':-0.29402,
              'Y':0.07416,'D':-0.43947,'W':0.32416,'G':-0.26220,'K':-0.61220,'T':-0.19856,
'S':-0.30311,'P':0,'X':0}

dict_Hydro={'T':-0.01124, 'V': 0.00685, 'R': -0.18437, 'C': 0.08695, 'P': -0.09134, 'F': 0.31693,
                'D': -0.29289, 'Q': -0.12494, 'W': 0.50297, 'K': -0.23088, 'Y': 0.26783,'M':0.08437,
                'E':-0.49703,'A':-0.01899,'I':0.10504,'S':-0.00866,'N':-0.08359,'H':-0.01899,'L':0.16964,
'G':0.02235,'X':0}


## load the sequence infomation and bond information about all dataset
def loadSeqsAndBonds():
    dictOfSeq=dict()
    dictOfPairs=dict()

    for line in open('../resource/sequences'):
        key,seq=line.strip().split(',')
        dictOfSeq[key]=seq


    for line in open('../resource/bonds'):
        items=line.strip().split(',')
        pairs=list()
        for i in range(len(items)-1):
            strpair=items[i+1].split('_')
            pair=(int(strpair[0]),int(strpair[1]))
            pairs.append(pair)

        dictOfPairs[items[0]]=pairs

    return(dictOfSeq,dictOfPairs)



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

## this function used to generated training data in numpy form.
##the shape of the numpy array is (protein_num,25,101,24)
##25 is the thredhold cys number in a proteins
##101 is the window size which cys is at 50, so it is easy to generate trainset of other window size
## for example window 7 X=Xraw[:,:,50-3:51+3,:]

def generatedTrainingData(lengthOfSubseq,numOfCys):
    halfwin = lengthOfSubseq // 2
    numOfInfo=24

    def generateSubexample(seq, pssm, pairs):

        def checkInBonds(index):
            found=0
            for p1, p2 in pairs:
                    if index == p1 - 1 or index == p2 - 1:
                        found=1
            return found

        X = np.zeros((numOfCys,lengthOfSubseq, numOfInfo), np.float32)
        XIndex =np.zeros((numOfCys,1,1),np.float32)
        T=np.zeros((numOfCys,2),np.float32)
        index=0
        cysinaline=0
        for i in range(len(seq)):
            if seq[i]=="C":
                cysinaline+=1
        ##print(cysinaline)
        if cysinaline>numOfCys or len(seq)>2000:
            return None,None,None
        errorFlag=0
        for i in range(len(seq)):
            if seq[i]=='C':
                ##print('CYS')
                if checkInBonds(i):
                    T[index]=[0.,1.]
                else:
                    T[index]=[1.,0.]

                XIndex[index]=(i-len(seq)/2.0)/1000
                for j in range(lengthOfSubseq):
                    tempindex = i - halfwin + j
                    if tempindex < 0 or tempindex > len(seq) - 1:
                        X[index,j,0:20] = np.zeros(20, np.float32)

                    else:
                        if seq[tempindex] not in dict_Hydro:
                            errorFlag=1
                            break
                        X[index,j,0:20] = pssm[tempindex]
                        X[index, j, 22]=dict_Hydro[seq[tempindex]]
                        X[index, j, 23]=dict_polarnorm[seq[tempindex]]
                    X[index, j, 20]=tempindex / 10000.0
                    X[index, j, 21] =(len(seq) - tempindex) / 10000.0

                index+=1
        if errorFlag:
            return None,None,None
        return X,XIndex, T

    Xraw = np.load('../resource/pssmProteins.npy')
    listOfSeq, listOfPairs = loadSeqsAndBonds()
    ##print(listOfPairs.items())
    listOfIdentity=list()
    for line in open('../resource/nameProteins'):
        listOfIdentity.append(line.strip())


    examples=list()
    for i in range(len(listOfIdentity)):
        seq = listOfSeq[listOfIdentity[i]]
        pair=listOfPairs[listOfIdentity[i]]
        X,XIndex, T = generateSubexample(seq, Xraw[i], pair)
        if X is None:
            continue

        if 0:
            print(i)
            print(seq)
            print(len(seq))
            print(listOfIdentity[i])
            print(X)
            print(XIndex)
            ##print(T)
            assert i-10
        examples.append((X,XIndex, T))
    for i in range(200):
        random.shuffle(examples)
    X=np.zeros((len(examples),numOfCys,lengthOfSubseq,numOfInfo),np.float32)
    X_i=np.zeros((len(examples),numOfCys,1,1),np.float32)
    T=np.zeros((len(examples),numOfCys,2),np.float32)

    for i in range(len(examples)):
        X[i]=examples[i][0]
        X_i[i]=examples[i][1]
        T[i]=examples[i][2]

    if os.path.exists('../resource/GLBlstmTrain/'+str(lengthOfSubseq)):
        pass
    else:
        os.mkdir('../resource/GLBlstmTrain/'+str(lengthOfSubseq))
    np.save('../resource/GLBlstmTrain/' + str(lengthOfSubseq) + '/X.data', X)
    np.save('../resource/GLBlstmTrain/' + str(lengthOfSubseq) + '/iX.data', X_i)
    np.save('../resource/GLBlstmTrain/' + str(lengthOfSubseq) + '/T.data', T)

    print('finished')

##this is a keras lambda layer
def crop():
	def func(x):
		returnResult=list()
		for i in range(x.shape[1]):

			returnResult.append((x[:,i]))
		return returnResult
	return Lambda(func)


##return acc on residues level, acc on proteins level mcc value and etc..
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



if 1:
    aminoAcidDict = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y',
                     'X']
    listOfSeq, listOfPairs = loadSeqsAndBonds()

    wins=5
    halfWins=wins//2
    listOfIdentity=list()
    for line in open('../resource/nameProteins'):
        listOfIdentity.append(line.strip())
    for i in range(len(listOfIdentity)):
        print('--------------')
        wob=list()
        seq = listOfSeq[listOfIdentity[i]]
        for j in range(len(seq)):
            if seq[j]=='C':
                a=np.zeros((21),np.int)
                for k in range(wins):
                    if j-halfWins+k<0 or j-halfWins+k>=len(seq):
                        break
                    if k==wins//2:
                        continue
                    aa=seq[j-halfWins+k]
                    if aa in aminoAcidDict:
                        a[aminoAcidDict.index(aa)]+=1
                print(a)
