import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer

# DATAFRAME SIMPLU PENTRU EXEMPLU

df = pd.DataFrame({
    "Tokenized_NLTK": [
        ["am", "febra", "mare"],
        ["tuse", "seaca"],
        ["durere", "abdomen", "severa"],
        ["oxigen", "scazut"],
        ["puls", "crescut"]
    ],
    "No_Stopwords_NLTK": [
        "febra mare",
        "tuse seaca",
        "durere abdomen severa",
        "oxigen scazut",
        "puls crescut"
    ]
})

print("\n=== **DataFrame initial** ===")
print(df.head())

# 2. ONE HOT ENCODING CU MultiLabelBinarizer (NLTK)

print("\n=== **ONE-HOT NLTK cu MultiLabelBinarizer** ===")

mlb = MultiLabelBinarizer()

df_onehot_nltk = pd.DataFrame(
    mlb.fit_transform(df["Tokenized_NLTK"]),
    columns=mlb.classes_
)

print(df_onehot_nltk.head())

print("\nNumar cuvinte unice MLB:", len(mlb.classes_))

# 3. ONE HOT ENCODING CU CountVectorizer (Scikit-learn)

print("\n=== **ONE-HOT Scikit-learn cu CountVectorizer** ===")

vectorizer = CountVectorizer()

df_onehot_sklearn = pd.DataFrame(
    vectorizer.fit_transform(df["No_Stopwords_NLTK"]).toarray(),
    columns=vectorizer.get_feature_names_out()
)

print(df_onehot_sklearn.head())

print("\nPrimele 10 cuvinte din vocabular:", vectorizer.get_feature_names_out()[:10])

# 4. ANALIZA COMPARATIVA

print("\n=== **COMPARATIE MATRICE** ===")
print("Dimensiune MLB:", df_onehot_nltk.shape)
print("Dimensiune CountVectorizer:", df_onehot_sklearn.shape)

print("\n=== **CONCLUZII** ===")
print("1. CountVectorizer produce matrice mai mare deoarece transforma textul complet.")
print("2. Ambele metode pierd ordinea cuvintelor.")
print("3. MLB este bun pentru tokeni, CountVectorizer este bun pentru text brut.")
