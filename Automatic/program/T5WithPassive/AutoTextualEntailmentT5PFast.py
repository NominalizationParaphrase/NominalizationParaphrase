import pandas as pd
dfs=pd.read_csv("FinalDatasetAutoDatasetT5PFast.csv",index_col=0)
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import numpy as np
predictor2 = Predictor.from_path("snli-roberta.2021-03-11",predictor_name="textual_entailment",cuda_device=0)
for index,i in enumerate(dfs['sentence']):
    if not pd.isnull(dfs.loc[index,'v']):
        print(index)
        findVMpO=False
        findVOpM=False
        sentencePatternList=['MVOList','MVpOList','OVMList','OVpMList','VOpMList','VpOpMList','VMpOList','VpMpOList','MVpOPassiveList','MpOVPassiveList','OVpMPassiveList']
        sentencePatternNameList=['MVO','MVpO','OVM','OVpM','VOpM','VpOpM','VMpO','VpMpO','MVpOPassive','MpOVPassive','OVpMPassive']
        for name,pattern in zip(sentencePatternNameList,sentencePatternList):
            tempResult=[]
            if not pd.isnull(dfs.loc[index,pattern]):
                pattern=dfs.loc[index,pattern].split("%")
                pattern=[x for x in pattern if str(x) != 'nan']
                #print(pattern)
                for hypothesisPattern in pattern:
                    #print(dfs.loc[index,'originalPattern'],hypothesisPattern,predictor2.predict(premise=dfs.loc[index,'originalPattern'],hypothesis=hypothesisPattern)['probs'][0])
                    tempResult.append(predictor2.predict(premise=dfs.loc[index,'originalPattern'],hypothesis=hypothesisPattern)['probs'][0])
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
sentencePattern=['originalPattern','MVO','MVpO','OVM','OVpM','VOpM','VpOpM','VMpO','VpMpO','MVpOPassive','MpOVPassive','OVpMPassive']
for index,i in enumerate(dfs['sentence']):
    if not pd.isnull(dfs.loc[index,'v']):
        print(index)
        scoreDict={}
        for pattern in sentencePattern:
            if not pd.isnull(dfs.loc[index,pattern]):
                scoreDict[pattern]=0
        count=0
        for pattern in scoreDict:
            if count!=0:
                scoreDict[pattern]=predictor2.predict(premise=dfs.loc[index,'originalPattern'],hypothesis=dfs.loc[index,pattern])['probs'][0]
                dfs.loc[index,pattern+"Score"]=predictor2.predict(premise=dfs.loc[index,'originalPattern'],hypothesis=dfs.loc[index,pattern])['probs'][0]
            count=count+1
        dfs.loc[index,'HighestScorePattern'] = max(scoreDict, key=scoreDict.get)
dfs.to_excel("FinalDatasetAutoTextualEntailmentT5P.xlsx")