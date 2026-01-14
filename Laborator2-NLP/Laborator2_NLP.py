import pandas as pd
import spacy
import string
from spacy.lang.en.stop_words import STOP_WORDS
from gensim.models import Word2Vec, FastText
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Incarcarea datasetului

df = pd.read_csv("sentiment_reviews.csv")
print("Primele 5 randuri ale datasetului:")
print(df.head())

# Impartire features / labels
X = df['Review']
y = df['Sentiment']

# 2. Preprocesarea textului cu spaCy
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text.lower())  # lowercase
    tokens = [token.lemma_ for token in doc if token.text not in string.punctuation and token.text.lower() not in STOP_WORDS]
    return " ".join(tokens)

df['Clean_Text'] = df['Review'].apply(preprocess_text)
print("\nPrimele 5 randuri dupa preprocesare:")
print(df[['Review', 'Clean_Text']].head())

# 3. Construirea embeddings
sentences = [text.split() for text in df['Clean_Text']]

# Word2Vec CBOW (sg=0)
w2v_cbow = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)
# Word2Vec Skip-Gram (sg=1)
w2v_sg = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)
# FastText
ft_model = FastText(sentences, vector_size=100, window=5, min_count=1)

def get_vector(model, tokens):
    vecs = [model.wv[word] for word in tokens if word in model.wv]
    if len(vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)

df['W2V_CBOW_Vector'] = df['Clean_Text'].apply(lambda x: get_vector(w2v_cbow, x.split()))
df['W2V_SkipGram_Vector'] = df['Clean_Text'].apply(lambda x: get_vector(w2v_sg, x.split()))
df['FastText_Vector'] = df['Clean_Text'].apply(lambda x: get_vector(ft_model, x.split()))

# 4. Construirea modelelor de clasificare
X_train, X_test, y_train, y_test = train_test_split(df.index, y, test_size=0.2, random_state=42)

def prepare_X_vectors(column_name):
    return np.vstack(df.loc[X_train, column_name].values), np.vstack(df.loc[X_test, column_name].values)

# Pregatire vectori
X_train_cbow, X_test_cbow = prepare_X_vectors('W2V_CBOW_Vector')
X_train_sg, X_test_sg = prepare_X_vectors('W2V_SkipGram_Vector')
X_train_ft, X_test_ft = prepare_X_vectors('FastText_Vector')

def evaluate_model(X_tr, X_te, y_tr, y_te, model):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return {
        'accuracy': accuracy_score(y_te, y_pred),
        'precision': precision_score(y_te, y_pred, pos_label='Positive'),
        'recall': recall_score(y_te, y_pred, pos_label='Positive'),
        'f1': f1_score(y_te, y_pred, pos_label='Positive')
    }

# Modele
nb_model = GaussianNB()
svm_model = SVC(kernel='linear')

# Evaluare Naive Bayes
results_nb_cbow = evaluate_model(X_train_cbow, X_test_cbow, y_train, y_test, nb_model)
results_nb_sg = evaluate_model(X_train_sg, X_test_sg, y_train, y_test, nb_model)
results_nb_ft = evaluate_model(X_train_ft, X_test_ft, y_train, y_test, nb_model)

# Evaluare SVM
results_svm_cbow = evaluate_model(X_train_cbow, X_test_cbow, y_train, y_test, svm_model)
results_svm_sg = evaluate_model(X_train_sg, X_test_sg, y_train, y_test, svm_model)
results_svm_ft = evaluate_model(X_train_ft, X_test_ft, y_train, y_test, svm_model)

# 5. Tabel comparativ
comparison_table = pd.DataFrame([
    ['Word2Vec-CBOW', 'Naive Bayes', results_nb_cbow],
    ['Word2Vec-SkipGram', 'Naive Bayes', results_nb_sg],
    ['FastText', 'Naive Bayes', results_nb_ft],
    ['Word2Vec-CBOW', 'SVM', results_svm_cbow],
    ['Word2Vec-SkipGram', 'SVM', results_svm_sg],
    ['FastText', 'SVM', results_svm_ft]
], columns=['Vectorizare', 'Model', 'Metrics'])

print("\nTabel comparativ al performantelor:")
print(comparison_table)

# 6. Analiza comparativa (scrisa)
print("\nAnaliza comparativa:")
print("a) FastText functioneaza cel mai bine pentru cuvinte necunoscute si morfologie complexa.")
print("b) SVM tinde sa dea rezultate mai bune decat Naive Bayes pentru embeddings continue.")
print("c) CBOW e mai rapid si stabileste contextul mediu al cuvintelor, Skip-Gram surprinde mai bine cuvintele rare.")
print("d) FastText poate genera embedding pentru cuvinte necunoscute folosind subcuvinte.")
print("e) Word2Vec + SVM de obicei performeaza mai bine decat Word2Vec + Naive Bayes.")
print("f) Calitatea embedding-urilor influenteaza direct performanta finala a clasificatorului.")

