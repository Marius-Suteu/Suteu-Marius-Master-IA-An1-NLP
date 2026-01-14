<<<<<<< HEAD
\# Analiza Sentimentelor cu Word2Vec, FastText si Clasificatoare



\## Descriere

Acest proiect efectueaza analiza sentimentelor pe recenzii textuale utilizand \*\*Word2Vec (CBOW si Skip-Gram)\*\* si \*\*FastText\*\* pentru embeddings, si clasificatoare \*\*Naive Bayes\*\* si \*\*SVM\*\* pentru predicție.  

Proiectul include preprocesarea textului cu \*\*spaCy\*\*, crearea vectorilor de cuvinte și evaluarea performantelor modelelor.



---



\## Fisiere

\- `sentiment\_analysis.py` - codul principal pentru analiza sentimentelor

\- `sentiment\_reviews.csv` - datasetul cu recenzii și etichete de sentiment (Positive/Negative)

\- `requirements.txt` - librarii necesare pentru rularea codului

\- `README.md` - acest fisier



---



\## Cerinte

\- Python 3.x

\- Biblioteci Python:

&nbsp; - pandas

&nbsp; - numpy

&nbsp; - spacy

&nbsp; - gensim

&nbsp; - scikit-learn



---



\## Instalare



1\. Cloneaza repository-ul:



```bash

git clone https://github.com/Marius-Suteu/Suteu-Marius-Master-IA-An1-NLP.git

cd Suteu-Marius-Master-IA-An1-NLP



Instaleaza dependintele: 



pip install -r requirements.txt

python -m spacy download en\_core\_web\_sm



Rulare

python sentiment\_analysis.py


=======
\# Laborator 3 - NLP



Pipeline NLP pentru POS Tagging folosind:

\- CRF (sklearn-crfsuite)

\- spaCy POS Tagger

\- Naive Bayes baseline



\## Functionalitati

\- Preprocesare text

\- Feature engineering

\- CRF + POS integration

\- Evaluare si comparatie



\## Rulare

```bash

pip install -r requirements.txt

python Laborator3\_NLP.py
>>>>>>> 997eb0f62ebb7003218ba1b58a8a12602855a8ef



