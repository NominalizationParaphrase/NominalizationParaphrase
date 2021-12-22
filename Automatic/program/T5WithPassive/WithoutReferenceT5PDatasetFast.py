import pandas as pd
dfs=pd.read_excel("AutoEMNLP2021_final.xlsx")
dfs['adj/noun.2']=dfs['adj/noun']
for columns in dfs:
    for index,i in enumerate(dfs['sentence']):
        if not pd.isnull(dfs.loc[index,columns]):
            dfs.loc[index,columns]=str(dfs.loc[index,columns]).strip()
        if "-" in dfs.loc[index,'adj/noun.2']:
            dfs.loc[index,'adj/noun.2']=dfs.loc[index,'adj/noun.2'].replace("-"," ")
from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("bert-masked-lm-2019.09.17")
import spacy
nlp = spacy.load("en_core_web_sm")
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
import torch
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

T5_PATH = 't5-small' # "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU

t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
t5_config = T5Config.from_pretrained(T5_PATH)
t5_mlm = T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(DEVICE)
# Input text
# Input text
def maskedWithT5(text):
    encoded = t5_tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
    input_ids = encoded['input_ids'].to(DEVICE)

    # Generaing 20 sequences with maximum length set to 5
    outputs = t5_mlm.generate(input_ids=input_ids, 
                              num_beams=200, num_return_sequences=17,
                              max_length=4)
    #print(outputs)
    _0_index = text.index('<extra_id_0>')
    _result_prefix = text[:_0_index]
    _result_suffix = text[_0_index+12:]# 12 is the length of <extra_id_0>

    def _filter(output, end_token='<extra_id_1>'):
        # The first token is <unk> (inidex at 0) and the second token is <extra_id_0> (indexed at 32099)
        _txt = t5_tokenizer.decode(output[2:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        #print(_txt)
        if end_token in _txt: 
            find=True
            moreThanOnePrep=0
            moreThanOneDT=0
            _end_token_index = _txt.index(end_token)
            for prep in _txt[:_end_token_index].split():
                doc=nlp(prep)
                #print("Here "+prep)
                if doc[0].text=='ry':
                    find=False
                elif doc[0].tag_=='IN' or doc[0].tag_=='DT' or doc[0].text=='a':
                    doc2=nlp(_result_suffix.split(" ")[1])
                    #print(_result_suffix.split(" ")[1],prep,_result_suffix.split(" ")[1]!=prep)
                    if _result_suffix.split(" ")[1]==prep or (doc[0].tag_=='DT' and doc2[0].tag_=='DT') or (doc[0].tag_=='IN' and doc2[0].tag_=='IN') or (doc[0].tag_=='DT' and doc2[0].text=='a') or (doc[0].text=='a' and doc2[0].tag_=='DT'):
                        find=False
                    if doc[0].tag_=='IN':
                        moreThanOnePrep=moreThanOnePrep+1
                    if doc[0].tag_=='DT':
                        moreThanOneDT=moreThanOneDT+1
                else:
                    find=False
            if find==True:
                if len(_txt[:_end_token_index].strip())==0:
                    return 
                else:
                    return _result_prefix + _txt[:_end_token_index].lower() + _result_suffix
        else:
            find=True
            moreThanOnePrep=0
            moreThanOneDT=0
            for prep in _txt.split():
                doc=nlp(prep)
                if doc[0].text=='ry':
                    find=False
                elif doc[0].tag_=='IN' or doc[0].tag_=='DT' or doc[0].text=='a':
                    doc2=nlp(_result_suffix.split(" ")[1])
                    #print(_result_suffix.split(" ")[1],prep,_result_suffix.split(" ")[1]!=prep)
                    if _result_suffix.split(" ")[1]==prep or (doc[0].tag_=='DT' and doc2[0].tag_=='DT') or (doc[0].tag_=='IN' and doc2[0].tag_=='IN') or (doc[0].tag_=='DT' and doc2[0].text=='a') or (doc[0].text=='a' and doc2[0].tag_=='DT'):
                        find=False
                    if doc[0].tag_=='IN':
                        moreThanOnePrep=moreThanOnePrep+1
                    if doc[0].tag_=='DT':
                        moreThanOneDT=moreThanOneDT+1
                else:
                    find=False
            if moreThanOnePrep>1 or moreThanOneDT>1:
                find=False
            if find==True:
                if len( _txt.strip())==0:
                    return 
                else:
                    return _result_prefix + _txt.strip().lower() + _result_suffix
    results = list(map(_filter, outputs))
    results=[i.strip() for i in results if i]
    results=list(dict.fromkeys(results))
    if len(results)==0:
        results=[]
    return results
# Input text
def maskedWithT5DT(text):
    encoded = t5_tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
    input_ids = encoded['input_ids'].to(DEVICE)

    # Generaing 20 sequences with maximum length set to 5
    outputs = t5_mlm.generate(input_ids=input_ids, 
                              num_beams=200, num_return_sequences=17,
                              max_length=4)
    #print(outputs)
    _0_index = text.index('<extra_id_0>')
    _result_prefix = text[:_0_index]
    _result_suffix = text[_0_index+12:]# 12 is the length of <extra_id_0>

    def _filter(output, end_token='<extra_id_1>'):
        # The first token is <unk> (inidex at 0) and the second token is <extra_id_0> (indexed at 32099)
        _txt = t5_tokenizer.decode(output[2:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        #print(_txt)
        if end_token in _txt: 
            find=True
            moreThanOneDT=0
            _end_token_index = _txt.index(end_token)
            for prep in _txt[:_end_token_index].split():
                doc=nlp(prep)
                #print("Here "+prep)
                if doc[0].text=='ry':
                    find=False
                elif  doc[0].tag_=='DT' or doc[0].text=='a':
                    doc2=nlp(_result_suffix.split(" ")[1])
                    #print(_result_suffix.split(" ")[1],prep,_result_suffix.split(" ")[1]!=prep)
                    if _result_suffix.split(" ")[1]==prep or (doc[0].tag_=='DT' and doc2[0].tag_=='DT') or (doc[0].tag_=='DT' and doc2[0].text=='a') or (doc[0].text=='a' and doc2[0].tag_=='DT'):
                        find=False
                    if doc[0].tag_=='DT':
                        moreThanOneDT=moreThanOneDT+1
                else:
                    find=False
            if find==True:
                if len(_txt[:_end_token_index].strip())==0:
                    return 
                else:
                    return _result_prefix + _txt[:_end_token_index].lower() + _result_suffix
        else:
            find=True
            moreThanOneDT=0
            for prep in _txt.split():
                doc=nlp(prep)
                #print("Here "+prep,doc[0].tag_)
                if doc[0].text=='ry':
                    find=False
                elif  doc[0].tag_=='DT' or doc[0].text=='a':
                    doc2=nlp(_result_suffix.split(" ")[1])
                    #print(_result_suffix.split(" ")[1],prep,_result_suffix.split(" ")[1]!=prep)
                    if _result_suffix.split(" ")[1]==prep or (doc[0].tag_=='DT' and doc2[0].tag_=='DT') or (doc[0].tag_=='DT' and doc2[0].text=='a') or (doc[0].text=='a' and doc2[0].tag_=='DT'):
                        find=False
                    if doc[0].tag_=='DT':
                        moreThanOneDT=moreThanOneDT+1
                else:
                    find=False
            if find==True:
                if len( _txt.strip())==0:
                    return 
                else:
                    return _result_prefix + _txt.strip().lower() + _result_suffix
    results = list(map(_filter, outputs))
    results=[i.strip() for i in results if i]
    results=list(dict.fromkeys(results))
    if len(results)==0:
        results=[text.replace("<extra_id_0> ","")]
    return results
for index,i in enumerate(dfs['sentence']):
    if index<275:
        if not pd.isnull(dfs.loc[index,'v']):
            print(index)
            dfs.loc[index,'originalPattern']=dfs.loc[index,'adj/noun']+" "+dfs.loc[index,'n_v']+" "+dfs.loc[index,'prep']+" "+dfs.loc[index,'pobj']
            wordChoice=[]
            MVOList=[]
            MVpOList=[]
            OVMList=[]
            OVpMList=[]
            VOpMList=[]
            VpOpMList=[]
            VMpOList=[]
            VpMpOList=[]
            MVpOPassiveList=[]
            MpOVPassiveList=[]
            OVpMPassiveList=[]
            VMpOPrepList=[]
            VOpMPrepList=[]
            vWordList=dfs.loc[index,'v'].split()
            pluralWordList=dfs.loc[index,'pluralVerb'].split()
            pastParticipleVerbWordList=dfs.loc[index,'pastParticipleVerb'].split()
            for sWord in dfs.loc[index,'s'].split():
                wordChoice.append(sWord)
                if sWord[-1]=='s' or sWord[-2:]=='ch' or sWord[-2:]=='sh' or sWord[-1]=='z' or sWord[-1]=='x':
                    addSWord=sWord+'es'
                elif sWord[-1]=='y':
                    addSWord=sWord[0:-1]+'ies'
                else:
                    addSWord=sWord+'s'
                wordChoice.append(addSWord)
            for word in wordChoice:
                plural=False
                plural2=False
                doc=nlp(word)
                for vWord,pluralWord,pastParticipleVerb in zip(vWordList,pluralWordList,pastParticipleVerbWordList):
                    verbWord=vWord
                    if doc[len(doc)-1].tag_=='NN' or  doc[len(doc)-1].tag_=='NNP' or "a" == word.split()[0] or "an" == word.split()[0]:
                        verbWord=pluralWord
                        plural=True
                    if doc[len(doc)-1].tag_=='NNS' or  doc[len(doc)-1].tag_=='NNPS':
                        if verbWord[-1]=='s' and verbWord[-2:]!='ss':
                            verbWord=verbWord[0:-1]
                            plural=False
                    MVO="<extra_id_0> "+word+" "+verbWord+" "+dfs.loc[index,'pobj']
                    MVOList.extend(maskedWithT5DT(MVO))
                    MVpO=word+" "+verbWord+" <extra_id_0> "+dfs.loc[index,'pobj']
                    temp=maskedWithT5(MVpO)
                    for mvpo in temp:
                        MVpO=mvpo.replace(word+" "+verbWord,"<extra_id_0> "+word+" "+verbWord)
                        MVpOList.extend(maskedWithT5DT(MVpO))
                    if dfs.loc[index,'prep']=='of':
                        MVpOList=[]
                    if plural==True:
                        MVpOPassive=word+" is "+pastParticipleVerb+" <extra_id_0> "+dfs.loc[index,'pobj']
                    else:
                        MVpOPassive=word+" are "+pastParticipleVerb+" <extra_id_0> "+dfs.loc[index,'pobj']
                    temp=maskedWithT5(MVpOPassive)
                    for mvpopassive in temp:
                        if plural==True:
                            MVpOPassive=mvpopassive.replace(word+" is "+pastParticipleVerb+" ","<extra_id_0> "+word+" is "+pastParticipleVerb+" ")
                        else:
                            MVpOPassive=mvpopassive.replace(word+" are "+pastParticipleVerb+" ","<extra_id_0> "+word+" are "+pastParticipleVerb+" ")
                        MVpOPassiveList.extend(maskedWithT5DT(MVpOPassive))
                    if plural==True:
                        MpOVPassive=word+" "+dfs.loc[index,'prep']+" <extra_id_0> "+dfs.loc[index,'pobj']+" is "+pastParticipleVerb
                    else:
                        MpOVPassive=word+" "+dfs.loc[index,'prep']+" <extra_id_0> "+dfs.loc[index,'pobj']+" are "+pastParticipleVerb
                    temp=maskedWithT5(MpOVPassive)
                    for mpovpassive in temp:
                        MpOVPassive=mpovpassive.replace(word+" "+dfs.loc[index,'prep'],"<extra_id_0> "+word+" "+dfs.loc[index,'prep'])
                        MpOVPassiveList.extend(maskedWithT5DT(MpOVPassive))
                    doc2=nlp(dfs.loc[index,'pobj'].split()[-1])
                    verbWord2=vWord
                    if not "and" in dfs.loc[index,'pobj'].split():
                        print(dfs.loc[index,'pobj'].split()[0])
                        if doc2[len(doc2)-1].tag_=='NN' or  doc2[len(doc2)-1].tag_=='NNP' or "a" == dfs.loc[index,'pobj'].split()[0] or "an" == dfs.loc[index,'pobj'].split()[0]:
                            verbWord2=pluralWord
                            plural2=True
                        if doc2[len(doc2)-1].tag_=='NNS' or  doc2[len(doc2)-1].tag_=='NNPS':
                            if verbWord2[-1]=='s' and verbWord2[-2:]!='ss':
                                verbWord2=verbWord2[0:-1]
                                plural2=False
                    else:
                        plural2=False
                        if verbWord2[-1]=='s' and verbWord2[-2:]!='ss':
                            verbWord2=verbWord2[0:-1]
                            plural2=False
                    OVM=dfs.loc[index,'pobj']+" "+verbWord2+" "+word
                    OVMList.append(OVM)
                    OVpM=dfs.loc[index,'pobj']+" "+verbWord2+" <extra_id_0> "+word
                    OVpMList.extend(maskedWithT5(OVpM))
                    if plural2==True:
                        OVpMPassive=dfs.loc[index,'pobj']+" is "+pastParticipleVerb+" <extra_id_0> "+word
                    else:
                        OVpMPassive=dfs.loc[index,'pobj']+" are "+pastParticipleVerb+" <extra_id_0> "+word
                    OVpMPassiveList.extend(maskedWithT5(OVpMPassive))
                    vWord4=vWord
                    VOpM=vWord4+" "+dfs.loc[index,'pobj']+" <extra_id_0> "+word
                    temp=maskedWithT5(VOpM)
                    VOpMList.extend(temp)
                    for vopm in temp:
                        VOpMPrep=vopm.replace(vWord4+" "+dfs.loc[index,'pobj']+" ","").replace(" "+word,"")
                        VOpMPrepList.append(VOpMPrep)
                        VpOpM=vopm.replace(vWord4+" "+dfs.loc[index,'pobj'],vWord4+" <extra_id_0> "+dfs.loc[index,'pobj'])
                        VpOpMList.extend(maskedWithT5(VpOpM))
                    VMpO=vWord4+" "+word+" <extra_id_0> "+dfs.loc[index,'pobj']
                    temp=maskedWithT5(VMpO)
                    VMpOList.extend(temp)
                    for vmpo in temp:
                        VMpOPrep=vmpo.replace(vWord4+" "+word+" ","").replace(" "+dfs.loc[index,'pobj'],"")
                        VMpOPrepList.append(VMpOPrep)
                        VpMpO=vmpo.replace(vWord4+" "+word,vWord4+" <extra_id_0> "+word)
                        VpMpOList.extend(maskedWithT5(VpMpO))
            dfs.loc[index,'originalPattern']=dfs.loc[index,'adj/noun.2']+" "+dfs.loc[index,'n_v']+" "+dfs.loc[index,'prep']+" "+dfs.loc[index,'pobj']
            dfs.loc[index,'MVOList']="%".join(MVOList)
            dfs.loc[index,'MVpOList']="%".join(MVpOList)
            dfs.loc[index,'OVMList']="%".join(OVMList)
            dfs.loc[index,'OVpMList']="%".join(OVpMList)
            dfs.loc[index,'VOpMList']="%".join(VOpMList)
            dfs.loc[index,'VpOpMList']="%".join(VpOpMList)
            dfs.loc[index,'VMpOList']="%".join(VMpOList)
            dfs.loc[index,'VpMpOList']="%".join(VpMpOList)
            dfs.loc[index,'MVpOPassiveList']="%".join(MVpOPassiveList)
            dfs.loc[index,'MpOVPassiveList']="%".join(MpOVPassiveList)
            dfs.loc[index,'OVpMPassiveList']="%".join(OVpMPassiveList)
            dfs.loc[index,'VOpMPrepList']="%".join(VOpMPrepList)
            dfs.loc[index,'VMpOPrepList']="%".join(VMpOPrepList)
for index,i in enumerate(dfs['sentence']):
    if index>=275:
        if not pd.isnull(dfs.loc[index,'v']):
            print(index)
            dfs.loc[index,'originalPattern']=dfs.loc[index,'adj/noun.2']+" "+dfs.loc[index,'n_v']+" "+dfs.loc[index,'prep']+" "+dfs.loc[index,'pobj']
            wordChoice=[]
            MVOList=[]
            MVpOList=[]
            OVMList=[]
            OVpMList=[]
            MVpOPassiveList=[]
            MpOVPassiveList=[]
            OVpMPassiveList=[]
            VOpMList=[]
            VpOpMList=[]
            VMpOList=[]
            VpMpOList=[]
            MVpOPassiveList=[]
            MpOVPassiveList=[]
            OVpMPassiveList=[]
            VMpOPrepList=[]
            VOpMPrepList=[]
            vWordList=dfs.loc[index,'v'].split()
            pluralWordList=dfs.loc[index,'pluralVerb'].split()
            pastParticipleVerbWordList=dfs.loc[index,'pastParticipleVerb'].split()
            wordChoice.append(dfs.loc[index,'adj/noun.2'])
            if dfs.loc[index,'adj/noun.2'][-1]=='s' or dfs.loc[index,'adj/noun.2'][-2:]=='sh' or dfs.loc[index,'adj/noun.2'][-2:]=='ch' or dfs.loc[index,'adj/noun.2'][-1]=='x' or dfs.loc[index,'adj/noun.2'][-1]=='z':
                addSWord=dfs.loc[index,'adj/noun.2']+'es'
            elif dfs.loc[index,'adj/noun.2'][-1]=='y':
                addSWord=dfs.loc[index,'adj/noun.2'][0:-1]+'ies'
            else:
                addSWord=dfs.loc[index,'adj/noun.2']+'s'
            wordChoice.append(addSWord)
            for word in wordChoice:
                plural=False
                plural2=False
                doc=nlp(word)
                for vWord,pluralWord,pastParticipleVerb in zip(vWordList,pluralWordList,pastParticipleVerbWordList):
                    verbWord=vWord
                    if doc[len(doc)-1].tag_=='NN' or  doc[len(doc)-1].tag_=='NNP' or "a" == word.split()[0] or "an" == word.split()[0]:
                        verbWord=pluralWord
                        plural=True
                    if doc[len(doc)-1].tag_=='NNS' or  doc[len(doc)-1].tag_=='NNPS':
                        if verbWord[-1]=='s' and verbWord[-2:]!='ss':
                            verbWord=verbWord[0:-1]
                            plural=False
                    MVO="<extra_id_0> "+word+" "+verbWord+" "+dfs.loc[index,'pobj']
                    MVOList.extend(maskedWithT5DT(MVO))
                    MVpO=word+" "+verbWord+" <extra_id_0> "+dfs.loc[index,'pobj']
                    temp=maskedWithT5(MVpO)
                    for mvpo in temp:
                        MVpO=mvpo.replace(word+" "+verbWord,"<extra_id_0> "+word+" "+verbWord)
                        MVpOList.extend(maskedWithT5DT(MVpO))
                    if dfs.loc[index,'prep']=='of':
                        MVpOList=[]
                    if plural==True:
                        MVpOPassive=word+" is "+pastParticipleVerb+" <extra_id_0> "+dfs.loc[index,'pobj']
                    else:
                        MVpOPassive=word+" are "+pastParticipleVerb+" <extra_id_0> "+dfs.loc[index,'pobj']
                    temp=maskedWithT5(MVpOPassive)
                    for mvpopassive in temp:
                        if plural==True:
                            MVpOPassive=mvpopassive.replace(word+" is "+pastParticipleVerb+" ","<extra_id_0> "+word+" is "+pastParticipleVerb+" ")
                        else:
                            MVpOPassive=mvpopassive.replace(word+" are "+pastParticipleVerb+" ","<extra_id_0> "+word+" are "+pastParticipleVerb+" ")
                        MVpOPassiveList.extend(maskedWithT5DT(MVpOPassive))
                    if plural==True:
                        MpOVPassive=word+" "+dfs.loc[index,'prep']+" <extra_id_0> "+dfs.loc[index,'pobj']+" is "+pastParticipleVerb
                    else:
                        MpOVPassive=word+" "+dfs.loc[index,'prep']+" <extra_id_0> "+dfs.loc[index,'pobj']+" are "+pastParticipleVerb
                    temp=maskedWithT5(MpOVPassive)
                    for mpovpassive in temp:
                        MpOVPassive=mpovpassive.replace(word+" "+dfs.loc[index,'prep'],"<extra_id_0> "+word+" "+dfs.loc[index,'prep'])
                        MpOVPassiveList.extend(maskedWithT5DT(MpOVPassive))
                    doc2=nlp(dfs.loc[index,'pobj'].split()[-1])
                    verbWord2=vWord
                    if not "and" in dfs.loc[index,'pobj'].split():
                        print(dfs.loc[index,'pobj'].split()[0])
                        if doc2[len(doc2)-1].tag_=='NN' or  doc2[len(doc2)-1].tag_=='NNP' or "a" == dfs.loc[index,'pobj'].split()[0] or "an" == dfs.loc[index,'pobj'].split()[0]:
                            verbWord2=pluralWord
                            plural2=True
                        if doc2[len(doc2)-1].tag_=='NNS' or  doc2[len(doc2)-1].tag_=='NNPS':
                            if verbWord2[-1]=='s' and verbWord2[-2:]!='ss':
                                verbWord2=verbWord2[0:-1]
                                plural2=False
                    else:
                        plural2=False
                        if verbWord2[-1]=='s' and verbWord2[-2:]!='ss':
                            verbWord2=verbWord2[0:-1]
                            plural2=False
                    OVM=dfs.loc[index,'pobj']+" "+verbWord2+" "+word
                    OVMList.append(OVM)
                    OVpM=dfs.loc[index,'pobj']+" "+verbWord2+" <extra_id_0> "+word
                    OVpMList.extend(maskedWithT5(OVpM))
                    if plural2==True:
                        OVpMPassive=dfs.loc[index,'pobj']+" is "+pastParticipleVerb+" <extra_id_0> "+word
                    else:
                        OVpMPassive=dfs.loc[index,'pobj']+" are "+pastParticipleVerb+" <extra_id_0> "+word
                    OVpMPassiveList.extend(maskedWithT5(OVpMPassive))
                    vWord4=vWord
                    VOpM=vWord4+" "+dfs.loc[index,'pobj']+" <extra_id_0> "+word
                    temp=maskedWithT5(VOpM)
                    VOpMList.extend(temp)
                    for vopm in temp:
                        VOpMPrep=vopm.replace(vWord4+" "+dfs.loc[index,'pobj']+" ","").replace(" "+word,"")
                        VOpMPrepList.append(VOpMPrep)
                        VpOpM=vopm.replace(vWord4+" "+dfs.loc[index,'pobj'],vWord4+" <extra_id_0> "+dfs.loc[index,'pobj'])
                        VpOpMList.extend(maskedWithT5(VpOpM))
                    VMpO=vWord4+" "+word+" <extra_id_0> "+dfs.loc[index,'pobj']
                    temp=maskedWithT5(VMpO)
                    VMpOList.extend(temp)
                    for vmpo in temp:
                        VMpOPrep=vmpo.replace(vWord4+" "+word+" ","").replace(" "+dfs.loc[index,'pobj'],"")
                        VMpOPrepList.append(VMpOPrep)
                        VpMpO=vmpo.replace(vWord4+" "+word,vWord4+" <extra_id_0> "+word)
                        VpMpOList.extend(maskedWithT5(VpMpO))
            dfs.loc[index,'originalPattern']=dfs.loc[index,'adj/noun.2']+" "+dfs.loc[index,'n_v']+" "+dfs.loc[index,'prep']+" "+dfs.loc[index,'pobj']
            dfs.loc[index,'MVOList']="%".join(MVOList)
            dfs.loc[index,'MVpOList']="%".join(MVpOList)
            dfs.loc[index,'OVMList']="%".join(OVMList)
            dfs.loc[index,'OVpMList']="%".join(OVpMList)
            dfs.loc[index,'VOpMList']="%".join(VOpMList)
            dfs.loc[index,'VpOpMList']="%".join(VpOpMList)
            dfs.loc[index,'VMpOList']="%".join(VMpOList)
            dfs.loc[index,'VpMpOList']="%".join(VpMpOList)
            dfs.loc[index,'MVpOPassiveList']="%".join(MVpOPassiveList)
            dfs.loc[index,'MpOVPassiveList']="%".join(MpOVPassiveList)
            dfs.loc[index,'OVpMPassiveList']="%".join(OVpMPassiveList)
            dfs.loc[index,'VOpMPrepList']="%".join(VOpMPrepList)
            dfs.loc[index,'VMpOPrepList']="%".join(VMpOPrepList)
dfs.to_csv("FinalDatasetAutoDatasetT5PFast.csv")