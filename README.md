# CCE_NLI

pip install pyparsing==2.4.2

to retrain:

pip install -U spacy
python -m spacy download en_core_web_sm
python3 code/snli_train.py

to run explanations:

python3 code/analyze.py