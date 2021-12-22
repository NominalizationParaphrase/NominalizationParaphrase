# Paraphrasing Compound Nominalizations

## Abstract
A nominalization uses a deverbal noun to describe an event associated with its underlying
verb. Commonly found in academic and formal texts, nominalizations can be difficult to
interpret because of ambiguous semantic relations between the deverbal noun and its arguments. Our goal is to interpret nominalizations by generating clausal paraphrases. We
address compound nominalizations with both
nominal and adjectival modifiers, as well as
prepositional phrases. In evaluations on a number of unsupervised methods, we obtained the
strongest performance by using a pre-trained
contextualized language model to re-rank paraphrase candidates identified by a textual entailment model.
https://aclanthology.org/2021.emnlp-main.632.pdf

## Tools
1. Spacy == 2.2.4
2. Pandas
3. T5-small
4. TextualEntailment : https://demo.allennlp.org/textual-entailment/roberta-snli
5. GPT & DistilBert : https://github.com/awslabs/mlm-scoring
6. Sentence Embeddings : https://huggingface.co/sentence-transformers/stsb-roberta-large
7. AMR : https://github.com/ufal/perin

<!--
## How to use
1. There are two version : 
  - WithReference 
  - WithoutReference
2. If you use 'WithReference', just use 'EMNLP2021_final.xlsx' and put into the same folder with 'WithReferenceT5PDatasetFast.py'
3. If you use 'WithoutReference', just use 'AutoEMNLP2021_final.xlsx' and put into the same folder with 'WithoutReferenceT5PDatasetFast.py'
## How to generate paraphrase candidate(WithReference)
1. Run the program 'WithReferenceT5PDatasetFast.py', you will get 'FinalDatasetDatasetT5Fast.csv' 
2. Put 'FinalDatasetDatasetT5Fast.csv' into the same folder 'NominalizationParaphrase/WithReference/program/T5WithPassive/'
## How to generate paraphrase candidate(WithoutReference)
1. Run the program 'WithoutReferenceT5PDatasetFast.py', you will get 'FinalDatasetDatasetT5Fast.csv' 
2. Put 'FinalDatasetAutoDatasetT5Fast.csv' into the same folder 'NominalizationParaphrase/WithoutReference/program/T5WithPassive/'
## How to get the langauge model result(WithReference)
1. Run the program inside 'NominalizationParaphrase/WithReference/program/T5WithPassive/' folder (For example : DistilBertT5PFast.py,GPTT5PFast.py)
2. You can get the final result file For example : 'FinalDatasetDistilBertT5P.xlsx'
3. Put it into the 'NominalizationParaphrase/WithReference/EvaluationProgram/T5WithPassive/' folder , run the corresponding ipynb file. Then you can evaluate the result.
## How to get the langauge model result(WithoutReference)
1. Run the program inside 'NominalizationParaphrase/WithoutReference/program/T5WithPassive/' folder (For example : DistilBertT5PFast.py,GPTT5PFast.py)
2. You can get the final result file (For example : 'FinalDatasetAutoDistilBertT5P.xlsx')
3. Put it into the 'NominalizationParaphrase/WithoutReference/EvaluationProgram/T5WithPassive/' folder , run the corresponding ipynb file. Then you can evaluate the result.
**NominalizationParaphrase/NominalizationParaphrase** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
