from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
from mlm.models import get_pretrained
import mxnet as mx
import operator
ctxs = [mx.gpu(0)] # or, e.g., [mx.gpu(0), mx.gpu(1)]
model, vocab, tokenizer = get_pretrained(ctxs, 'distilbert-base-cased')
scorer = MLMScorerPT(model, vocab, tokenizer, ctxs)
import pandas as pd
import numpy as np
dfs=pd.read_csv("FinalDatasetAutoDatasetT5PFast.csv",index_col=0)
for index,i in enumerate(dfs['sentence']):
    if not pd.isnull(dfs.loc[index,'v']):
        print(index)
        sentencePatternList=['MVOList','MVpOList','VOpMList','VpOpMList']
        sentencePatternNameList=['MVO','MVpO','VOpM','VpOpM']
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
for index,i in enumerate(dfs['sentence']):
    if not pd.isnull(dfs.loc[index,'v']):
        if index<275:
            sentenceList=[]
            scoreDict={}
            sentencePattern=['MVO','MVpO']
            for pattern in sentencePattern:
                if not pd.isnull(dfs.loc[index,pattern]):
                    sentenceList.append(dfs.loc[index,pattern])
                    scoreDict[pattern]=0
            #print(scoreDict)
            print(sentenceList)
            #print(subjectList)
            #print(verbList)
            #print(SVOSentenceList)
            tempResult=[]
            for choice,pattern in zip(sentenceList,scoreDict):
                    scoreDict[pattern]=scorer.score_sentences([choice])[0]
                    dfs.loc[index,pattern+"Score"]=scorer.score_sentences([choice])[0]
            dfs.loc[index,'HighestScorePattern'] = max(scoreDict, key=scoreDict.get)
for index,i in enumerate(dfs['sentence']):
    if not pd.isnull(dfs.loc[index,'v']):
        if index>=275:
            runMVO=True
            sentenceList=[]
            scoreDict={}
            sentencePattern=['VOpM','VpOpM']
            for pattern in sentencePattern:
                if not pd.isnull(dfs.loc[index,pattern]):
                    sentenceList.append(dfs.loc[index,pattern])
                    scoreDict[pattern]=0
                    runMVO=False
            #print(scoreDict)
            print(sentenceList)
            #print(subjectList)
            #print(verbList)
            #print(SVOSentenceList)
            if runMVO==False:
                tempResult=[]
                for choice,pattern in zip(sentenceList,scoreDict):
                        scoreDict[pattern]=scorer.score_sentences([choice])[0]
                        dfs.loc[index,pattern+"Score"]=scorer.score_sentences([choice])[0]
                dfs.loc[index,'HighestScorePattern'] = max(scoreDict, key=scoreDict.get)
            if runMVO==True:
                sentencePattern=['MVO','MVpO']
                sentenceList=[]
                scoreDict={}
                for pattern in sentencePattern:
                    if not pd.isnull(dfs.loc[index,pattern]):
                        sentenceList.append(dfs.loc[index,pattern])
                        scoreDict[pattern]=0
                        runMVO=False
                #print(scoreDict)
                print(sentenceList)
                #print(subjectList)
                #print(verbList)
                #print(SVOSentenceList)
                tempResult=[]
                for choice,pattern in zip(sentenceList,scoreDict):
                        scoreDict[pattern]=scorer.score_sentences([choice])[0]
                        dfs.loc[index,pattern+"Score"]=scorer.score_sentences([choice])[0]
                dfs.loc[index,'HighestScorePattern'] = max(scoreDict, key=scoreDict.get)
dfs.to_excel("FinalDatasetAutoBaselineT5P.xlsx")