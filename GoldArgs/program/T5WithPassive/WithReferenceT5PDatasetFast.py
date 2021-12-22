import pandas as pd
dfs=pd.read_excel("EMNLP2021_final.xlsx")
dfs['adj/noun.2']=dfs['adj/noun.1']
for columns in dfs:
    for index,i in enumerate(dfs['sentence']):
        if not pd.isnull(dfs.loc[index,columns]):
            dfs.loc[index,columns]=str(dfs.loc[index,columns]).strip()
        if "-" in dfs.loc[index,'adj/noun.2']:
            dfs.loc[index,'adj/noun.2']=dfs.loc[index,'adj/noun.2'].replace("-"," ")
import spacy
nlp = spacy.load("en_core_web_sm")
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
import numpy as np
for index,i in enumerate(dfs['sentence']):
    print(index)
    wordChoice=[]
    wordChoice2=[]
    MVOList=[]
    MVpOList=[]
    MVpOPassiveList=[]
    MVpOPassiveVMpOList=[]
    MpOVPassiveList=[]
    MpOVPassiveVMpOList=[]
    OVpMPassiveList=[]
    OVpMPassiveVOpMList=[]
    OVMList=[]
    OVpMList=[]
    VMpOList=[]
    VpMpOList=[]
    VOpMList=[]
    VpOpMList=[]
    VMpOPrepList=[]
    VOpMPrepList=[]
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
        verbWord=dfs.loc[index,'V']
        if doc[len(doc)-1].tag_=='NN' or  doc[len(doc)-1].tag_=='NNP' or "a" == word.split()[0] or "an" == word.split()[0]:
                verbWord=dfs.loc[index,'pluralVerb']
                plural=True
        if doc[len(doc)-1].tag_=='NNS' or  doc[len(doc)-1].tag_=='NNPS':
            if verbWord[-1]=='s' and verbWord[-2:]!='ss':
                verbWord=verbWord[0:-1]
                plural=False
        MVO="<extra_id_0> "+word+" "+verbWord+" "+dfs.loc[index,'pobj']
        MVOList.extend(maskedWithT5DT(MVO))
        MVpO=word+" "+verbWord+" <extra_id_0> "+dfs.loc[index,'pobj']
        for mvpo in maskedWithT5(MVpO):
            MVpO=mvpo.replace(word+" "+verbWord,"<extra_id_0> "+word+" "+verbWord)
            MVpOList.extend(maskedWithT5DT(MVpO))
        if dfs.loc[index,'prep']=='of':
            MVpO=np.NaN
        if plural==True:
            MVpOPassive=word+" is "+dfs.loc[index,'pastParticipleVerb']+" <extra_id_0> "+dfs.loc[index,'pobj']
        else:
            MVpOPassive=word+" are "+dfs.loc[index,'pastParticipleVerb']+" <extra_id_0> "+dfs.loc[index,'pobj']
        for mvpopassive in maskedWithT5(MVpOPassive):
            if plural==True:
                MVpOPassive=mvpopassive.replace(word+" is "+dfs.loc[index,'pastParticipleVerb']+" ","<extra_id_0> "+word+" is "+dfs.loc[index,'pastParticipleVerb']+" ")
            else:
                MVpOPassive=mvpopassive.replace(word+" are "+dfs.loc[index,'pastParticipleVerb']+" ","<extra_id_0> "+word+" are "+dfs.loc[index,'pastParticipleVerb']+" ")
            MVpOPassiveList.extend(maskedWithT5DT(MVpOPassive))
        if plural==True:
            MpOVPassive=word+" "+dfs.loc[index,'prep']+" <extra_id_0> "+dfs.loc[index,'pobj']+" is "+dfs.loc[index,'pastParticipleVerb']
        else:
            MpOVPassive=word+" "+dfs.loc[index,'prep']+" <extra_id_0> "+dfs.loc[index,'pobj']+" are "+dfs.loc[index,'pastParticipleVerb']
        for mpovpassive in maskedWithT5(MpOVPassive):
            MpOVPassive=mpovpassive.replace(word+" "+dfs.loc[index,'prep'],"<extra_id_0> "+word+" "+dfs.loc[index,'prep'])
            MpOVPassiveList.extend(maskedWithT5DT(MpOVPassive))
        verbWord2=dfs.loc[index,'V']
        if not "and" in dfs.loc[index,'pobj'].split():
            doc2=nlp(dfs.loc[index,'pobj'].split()[-1])
            if doc2[len(doc2)-1].tag_=='NN' or  doc2[len(doc2)-1].tag_=='NNP' or "a" == dfs.loc[index,'pobj'].split()[0] or "an" == dfs.loc[index,'pobj'].split()[0]:
                    verbWord2=dfs.loc[index,'pluralVerb']
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
            OVpMPassive=dfs.loc[index,'pobj']+" is "+dfs.loc[index,'pastParticipleVerb']+" <extra_id_0> "+word
        else:
            OVpMPassive=dfs.loc[index,'pobj']+" are "+dfs.loc[index,'pastParticipleVerb']+" <extra_id_0> "+word
        OVpMPassiveList.extend(maskedWithT5(OVpMPassive))
        VOpM=dfs.loc[index,'baseV']+" "+dfs.loc[index,'pobj']+" <extra_id_0> "+word
        VOpMList.extend(maskedWithT5(VOpM))
        for vopm in maskedWithT5(VOpM):
            VOpMPrep=vopm.replace(dfs.loc[index,'baseV']+" "+dfs.loc[index,'pobj']+" ","").replace(" "+word,"")
            VOpMPrepList.append(VOpMPrep)
            VpOpM=vopm.replace(dfs.loc[index,'baseV']+" "+dfs.loc[index,'pobj'],dfs.loc[index,'baseV']+" <extra_id_0> "+dfs.loc[index,'pobj'])
            VpOpMList.extend(maskedWithT5(VpOpM))
        #VMpO&VpMpO
        VMpO=dfs.loc[index,'baseV']+" "+word+" <extra_id_0> "+dfs.loc[index,'pobj']
        VMpOList.extend(maskedWithT5(VMpO))
        for vmpo in maskedWithT5(VMpO):
            VMpOPrep=vmpo.replace(dfs.loc[index,'baseV']+" "+word+" ","").replace(" "+dfs.loc[index,'pobj'],"")
            VMpOPrepList.append(VMpOPrep)
            VpMpO=vmpo.replace(dfs.loc[index,'baseV']+" "+word,dfs.loc[index,'baseV']+" <extra_id_0> "+word)
            VpMpOList.extend(maskedWithT5(VpMpO))
    dfs.loc[index,'originalPattern']=dfs.loc[index,'adj/noun']+" "+dfs.loc[index,'n_v']+" "+dfs.loc[index,'prep']+" "+dfs.loc[index,'pobj']
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
dfs.to_csv("FinalDatasetDatasetT5Fast.csv")