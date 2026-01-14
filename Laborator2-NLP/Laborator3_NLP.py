# -*- coding: utf-8 -*-
# Model CRF pentru POS Tagging - Pipeline NLP complet

import pandas as pd
import nltk
from nltk.corpus import treebank
from sklearn.model_selection import train_test_split
import sklearn_crfsuite
from sklearn_crfsuite import metrics

# Download resurse necesare
nltk.download('treebank')
nltk.download('universal_tagset')
nltk.download('stopwords')

# 1. Incarcarea datasetului
sentences = treebank.tagged_sents(tagset='universal')

rows = []
for sent in sentences:
    tokens = [w for w, t in sent]
    tags = [t for w, t in sent]
    rows.append({
        "Sentence": " ".join(tokens),
        "Tokens": tokens,
        "POS": tags
    })

df = pd.DataFrame(rows)
print(df.head())

# 2. Preprocesare text
import spacy
import string
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

def preprocess(sentence):
    doc = nlp(sentence.lower())
    clean_tokens = []
    for token in doc:
        if token.text not in string.punctuation and token.text not in stop_words:
            clean_tokens.append(token.lemma_)
    return clean_tokens

df['Clean_Text'] = df['Sentence'].apply(preprocess)
print(df[['Sentence', 'Clean_Text']].head())

# 3. Construirea modelului CRF

def word2features(sent, i):
    word = sent[i]
    features = {
        'word.lower': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper': word.isupper(),
        'word.istitle': word.istitle(),
        'word.isdigit': word.isdigit(),
    }

    if i > 0:
        prev_word = sent[i - 1]
        features.update({
            '-1:word.lower': prev_word.lower(),
            '-1:word.istitle': prev_word.istitle(),
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        next_word = sent[i + 1]
        features.update({
            '+1:word.lower': next_word.lower(),
            '+1:word.istitle': next_word.istitle(),
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

# Date pentru CRF
X = [sent2features(s) for s in df['Tokens']]
y = df['POS'].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100
)

crf.fit(X_train, y_train)
y_pred = crf.predict(X_test)

print("F1-score CRF:",
      metrics.flat_f1_score(y_test, y_pred, average='weighted'))

# 4. POS Tagging cu model pre-antrenat (spaCy)
example = df.iloc[0]['Sentence']
doc = nlp(example)

print("\nspaCy POS tagging:")
for token in doc:
    print(token.text, token.pos_)

# 5. Integrarea POS spaCy ca feature pentru CRF

def word2features_with_pos(sent, pos_tags, i):
    features = word2features(sent, i)
    features['spacy_pos'] = pos_tags[i]
    return features

def sent2features_pos(sent):
    doc = nlp(" ".join(sent))
    pos_tags = [t.pos_ for t in doc]
    return [word2features_with_pos(sent, pos_tags, i)
            for i in range(len(sent))]

X_pos = [sent2features_pos(s) for s in df['Tokens']]

X_train, X_test, y_train, y_test = train_test_split(
    X_pos, y, test_size=0.2, random_state=42
)

crf_pos = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100
)

crf_pos.fit(X_train, y_train)
y_pred_pos = crf_pos.predict(X_test)

print("F1-score CRF + POS:",
      metrics.flat_f1_score(y_test, y_pred_pos, average='weighted'))

# 6. Naive Bayes - baseline corect (token-level)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

tokens = []
tags = []

for sent, pos in zip(df['Tokens'], df['POS']):
    tokens.extend(sent)
    tags.extend(pos)

vectorizer = CountVectorizer()
X_nb = vectorizer.fit_transform(tokens)

X_train, X_test, y_train, y_test = train_test_split(
    X_nb, tags, test_size=0.2, random_state=42
)

nb = MultinomialNB()
nb.fit(X_train, y_train)

print("Accuracy Naive Bayes:",
      accuracy_score(y_test, nb.predict(X_test)))
