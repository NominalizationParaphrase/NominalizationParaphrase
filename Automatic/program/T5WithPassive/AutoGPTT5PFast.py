from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
from mlm.models import get_pretrained
import mxnet as mx
import operator
import numpy as np
ctxs = [mx.gpu(0)] # or, e.g., [mx.gpu(0), mx.gpu(1)]
model, vocab, tokenizer = get_pretrained(ctxs, 'gpt2-117m-en-cased')
scorer = LMScorer(model, vocab, tokenizer, ctxs)
import pandas as pd
dfs=pd.read_csv("FinalDatasetAutoDatasetT5PFast.csv",index_col=0)
for index,i in enumerate(dfs['sentence']):
    if not pd.isnull(dfs.loc[index,'v']):
        print(index)
        findVMpO=False
        findVOpM=False
        sentencePatternList=['MVOList','MVpOList','OVMList','OVpMList','VOpMList','VpOpMList','VMpOList','VpMpOList','MVpOPassiveList','MpOVPassiveList','OVpMPassiveList']
        sentencePatternNameList=['MVO','MVpO','OVM','OVpM','VOpM','VpOpM','VMpO','VpMpO','MVpOPassive','MpOVPassive','OVpMPassive']
        for name,pattern in zip(sentencePatternNameList,sentencePatternList):
            if not pd.isnull(dfs.loc[index,pattern]):
                pattern=dfs.loc[index,pattern].split("%")
                pattern=[x for x in pattern if str(x) != 'nan']
                #print(name,pattern)
                tempResult=[]
                for choice in pattern:
                    if not pd.isnull(choice):
                        tempResult.append(scorer.score_sentences([choice])[0])
                        #print(choice,scorer.score_sentences([choice])[0])
                if tempResult!=[]:
                    dfs.loc[index,name]=pattern[tempResult.index(max(tempResult))]
                    if name=='VMpO':
                        VMpOList=pattern
                        bestVMpOIndex=tempResult.index(max(tempResult))
                        findVMpO=True
                    if name=='VOpM':
                        VOpMList=pattern
                        bestVOpMIndex=tempResult.index(max(tempResult))
                        findVOpM=True
        if findVMpO==True and not pd.isnull(dfs.loc[index,'MVpOPassive']):
            VMpOPrepList=dfs.loc[index,'VMpOPrepList'].split("%")
            for checkWord in dfs.loc[index,'pastParticipleVerb'].split():
                if checkWord in dfs.loc[index,'MVpOPassive']:
                    MVpOPassivePrep=dfs.loc[index,'MVpOPassive'].split()[dfs.loc[index,'MVpOPassive'].split().index(checkWord)+1]
            findPrepWord=False
            for findPrep,target in enumerate(VMpOList[bestVMpOIndex].split()):
                if target == VMpOPrepList[bestVMpOIndex]:
                    dfs.loc[index,'MVpOPassiveVMpO']=" ".join(VMpOList[bestVMpOIndex].split()[0:findPrep])+" "+MVpOPassivePrep+" "+" ".join(VMpOList[bestVMpOIndex].split()[findPrep+1:])
                    dfs.loc[index,'MpOVPassiveVMpO']=dfs.loc[index,'MVpOPassiveVMpO']
                    findPrepWord=True
                    break
            if findPrepWord==False:
                dfs.loc[index,'MVpOPassiveVMpO']=VMpOList[bestVMpOIndex].replace(VMpOPrepList[bestVMpOIndex],MVpOPassivePrep)
                dfs.loc[index,'MpOVPassiveVMpO']=dfs.loc[index,'MVpOPassiveVMpO']
        else:
            dfs.loc[index,'MVpOPassive']=np.NaN
            dfs.loc[index,'MpOVPassive']=np.NaN
        if findVOpM==True and not pd.isnull(dfs.loc[index,'OVpMPassive']) :
            VOpMPrepList=dfs.loc[index,'VOpMPrepList'].split("%")
            for checkWord in dfs.loc[index,'pastParticipleVerb'].split():
                if checkWord in dfs.loc[index,'OVpMPassive']:
                    OVpMPassivePrep=dfs.loc[index,'OVpMPassive'].split()[dfs.loc[index,'OVpMPassive'].split().index(checkWord)+1]
            findPrepWord=False
            for findPrep,target in enumerate(VOpMList[bestVOpMIndex].split()):
                if target == VOpMPrepList[bestVOpMIndex]:
                    dfs.loc[index,'OVpMPassiveVOpM']=" ".join(VOpMList[bestVOpMIndex].split()[0:findPrep])+" "+OVpMPassivePrep+" "+" ".join(VOpMList[bestVOpMIndex].split()[findPrep+1:])
                    findPrepWord=True
                    break
            if findPrepWord==False:
                dfs.loc[index,'OVpMPassiveVOpM']=VOpMList[bestVOpMIndex].replace(VOpMPrepList[bestVOpMIndex],OVpMPassivePrep)
        else:
            dfs.loc[index,'OVpMPassive']=np.NaN
sentencePattern=['MVO','MVpO','OVM','OVpM','VOpM','VpOpM','VMpO','VpMpO','MVpOPassive','MpOVPassive','OVpMPassive']
for index,i in enumerate(dfs['sentence']):
    if not pd.isnull(dfs.loc[index,'v']):
        sentenceList=[]
        scoreDict={}
        for pattern in sentencePattern:
            if not pd.isnull(dfs.loc[index,pattern]):
                sentenceList.append(dfs.loc[index,pattern])
                scoreDict[pattern]=0
        #print(scoreDict)
        print(sentenceList)
        #print(subjectList)
        #print(verbList)
        #print(MVOSentenceList)
        tempResult=[]
        for choice,pattern in zip(sentenceList,scoreDict):
                scoreDict[pattern]=scorer.score_sentences([choice])[0]
                dfs.loc[index,pattern+"Score"]=scorer.score_sentences([choice])[0]
        dfs.loc[index,'HighestScorePattern'] = max(scoreDict, key=scoreDict.get)
dfs.to_excel("FinalDatasetAutoGPTT5P.xlsx")