import os.path
import torch
import json
from IPython.display import Image
from datetime import date
import sys
sys.path.append("/perin")
from data.batch import Batch
from config.params import Params
from data.shared_dataset import SharedDataset
from model.model import Model

output_path = "output.mrp"
image_path = "output.png"
checkpoint = {
    "amr": {"name": "base_amr.h5", "path": "1eeDlp90DeqMW_LqieiaRN_3c76rVKkHy"},
    "drg": {"name": "base_drg.h5", "path": "1aifuh2zv62Atbl9P_8v3_l-i92wn6Fyh"},
    "eds": {"name": "base_eds.h5", "path": "1EK1y0zzlaidHC9Brm4DR9Bhd3lWHhD-B"},
    "ptg": {"name": "base_ptg.h5", "path": "1LOeCHo5lAgXxm-j-p54WMnsHtbaHlAOo"},
    "ucca": {"name": "base_ucca.h5", "path": "13fGpzlAbmNjpM8AC2EWBPFjY17qmnE7M"}
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load a pretrained model with dataset vocabularies
#
def load_checkpoint(framework, device):
    filename=checkpoint[framework]["name"]
    state_dict = torch.load(filename, map_location=device)
    args = Params().load_state_dict(state_dict["args"])

    dataset = SharedDataset(args)
    dataset.load_state_dict(args, state_dict["dataset"])

    model = Model(dataset, args, initialize=False).to(device).eval()
    model.load_state_dict(state_dict["model"])
    
    return model, dataset, args



# Parse the input sentences: 
#    1) preprocess them into the correct format, 
#    2) parse them with the pretrained neural network,
#    3) postprocess into the MRP format, 
#    4) clean the output
#
def parse(input, model, dataset, args, framework, language, **kwargs):
    # preprocess
    batches = dataset.load_sentences(input, args, framework, language)
    output = batches.dataset.datasets[dataset.framework_to_id[(framework, language)]].data
    output = list(output.values())
    
    for i, batch in enumerate(batches):
        # parse and postprocess
        with torch.no_grad():
            prediction = model(Batch.to(batch, device), inference=True, **kwargs)[(framework, language)][0]

        for key, value in prediction.items():
            output[i][key] = value

        # clean the output
        output[i]["input"] = output[i]["sentence"]
        output[i] = {k: v for k, v in output[i].items() if k in {"id", "input", "nodes", "edges", "tops"}}
        output[i]["framework"] = framework
        output[i]["time"] = str(date.today())
        
    return output


# Save the parsed graph into json-like MRP format.
#
def save(output, path):
    with open(path, "w", encoding="utf8") as f:
        for sentence in output:
            json.dump(sentence, f, ensure_ascii=False)
            f.write("\n")

framework = "amr"
model, dataset, args = load_checkpoint(framework, device)


import pandas as pd
dfs=pd.read_excel("EMNLP2021_final.xlsx")
displayGraph=[]
for index,i in enumerate(dfs['adj/noun']):
    dfs.loc[index,'sentence']=" ".join(dfs.loc[index,'sentence'].split())

for index,s in enumerate(dfs['sentence']):
    nounChunk=[]
    sentences = [s]
    language = "eng"  # available languages: {"eng", "zho"}
    prediction = parse(sentences, model, dataset, args, framework, language,approximate_anchors=True)
    displayGraph.append(prediction)
    Nfind=False
    for i in prediction[0]['nodes']:
        if prediction[0]['input'][i['anchors'][0]['from']:i['anchors'][0]['to']]==dfs.loc[index,'n_v']:
            source=i['id']
            N=prediction[0]['input'][i['anchors'][0]['from']:i['anchors'][0]['to']]
            dfs.loc[index,'Noun']=N
            dfs.loc[index,'NounIndexStart']=i['anchors'][0]['from']
            dfs.loc[index,'NounIndexEnd']=i['anchors'][0]['to']
            dfs.loc[index,'Verb']=i['label']
            Nfind=True
            print(index,N)
    resultWord={}
    resultNodeIndexEnd={}
    resultNodeIndexStart={}
    resultNodeName={}
    if Nfind==True:
      for i in prediction[0]['edges']:
          if source==i['source']:
              tag=i['label']
              target=i['target']
              resultNodeName[tag]=[]
              resultNodeIndexStart[tag]=[]
              resultNodeIndexEnd[tag]=[]
              resultWord[tag]=[]
    if Nfind==True:
        for i in prediction[0]['edges']:
            if source==i['source']:
                tag=i['label']
                target=i['target']
                for i in prediction[0]['nodes']:
                    if i['id']==target:
                        resultNodeName[tag].append(i['label'])
                        resultNodeIndexStart[tag].append(str(i['anchors'][0]['from']))
                        resultNodeIndexEnd[tag].append(str(i['anchors'][0]['to']))
                        resultWord[tag].append(prediction[0]['input'][i['anchors'][0]['from']:i['anchors'][0]['to']])
        for tag in resultWord:
            dfs.loc[index,"Result"+tag+"NodeName"]=",".join(resultNodeName[tag])
            dfs.loc[index,"Result"+tag+"IndexStart"]=",".join(resultNodeIndexStart[tag])
            dfs.loc[index,"Result"+tag+"IndexEnd"]=",".join(resultNodeIndexEnd[tag])
            dfs.loc[index,"Result"+tag]=",".join(resultWord[tag])
dfs.to_excel("FinalDatasetAMRPredictionResult.xlsx")
with open('FinalDatasetAMRGraph.json', 'w') as f:
    json.dump(displayGraph, f, indent=2)
