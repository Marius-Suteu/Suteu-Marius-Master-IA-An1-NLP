import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Incarcarea datasetului
df = pd.read_csv("customer_reviews.csv")

# Crearea unei noi coloane cu textul la lowercase folosind Pandas
df['Review_lower'] = df['Review'].str.lower()

# Afisarea primelor 5 randuri pentru verificare
print(df.head())

# Cream coloana Lowercase_Pandas
df['Lowercase_Pandas'] = df['Review_lower']

# Incarca modelul mic de lb eng
nlp = spacy.load("en_core_web_sm")

# Functie pentru lowercase folosind spaCy
def lowercase_spacy(text):
    doc = nlp(text)
    return " ".join([token.text.lower() for token in doc])

# Aplicam functia pe coloana Review
df['Lowercase_spaCy'] = df['Review'].apply(lowercase_spacy)

# Afisam primele 5 randuri cu ambele metode
print("\nPrimele 5 randuri cu ambele coloane lowercase:")
print(df[['Review', 'Lowercase_Pandas', 'Lowercase_spaCy']].head())

# Incarcarea stopwords

# Descarcarea resurselor NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Stopwords NLTK
stop_words_nltk = set(stopwords.words('english'))

# Stopwords spaCy
stop_words_spacy = nlp.Defaults.stop_words

# Compararea celor doua liste
common_stopwords = stop_words_nltk.intersection(stop_words_spacy)
unique_nltk = stop_words_nltk - stop_words_spacy
unique_spacy = stop_words_spacy - stop_words_nltk

print("\nNumar stopwords NLTK:", len(stop_words_nltk))
print("Numar stopwords spaCy:", len(stop_words_spacy))
print("Numar comune:", len(common_stopwords))
print("Stopwords doar NLTK:", list(unique_nltk)[:10])   # primele 10 pentru exemplu
print("Stopwords doar spaCy:", list(unique_spacy)[:10]) # primele 10 pentru exemplu

# Eliminarea stopwords folosind NLTK

def remove_stopwords_nltk_simple(text):
    # Impartim textul in cuvinte folosind split pe spatii
    tokens = text.split()
    # Eliminam stopwords
    filtered_tokens = [t for t in tokens if t.lower() not in stop_words_nltk]
    # Reconstruim textul
    return " ".join(filtered_tokens)

# Aplicam functia pe coloana Lowercase_Pandas
df['No_Stopwords_NLTK'] = df['Lowercase_Pandas'].apply(remove_stopwords_nltk_simple)

# Afisam primele 5 randuri pentru verificare
print("\nPrimele 5 randuri dupa eliminarea stopwords NLTK (varianta simpla):")
print(df[['Review', 'Lowercase_Pandas', 'No_Stopwords_NLTK']].head())

# Eliminarea stopwords folosind spaCy

def remove_stopwords_spacy(text):
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if token.text.lower() not in stop_words_spacy]
    return " ".join(filtered_tokens)

# Aplicam functia pe coloana Lowercase_spaCy
df['No_Stopwords_spacy'] = df['Lowercase_spaCy'].apply(remove_stopwords_spacy)

# Afisam primele 5 randuri pentru verificare
print("\nPrimele 5 randuri dupa eliminarea stopwords spaCy:")
print(df[['Review', 'Lowercase_spaCy', 'No_Stopwords_spacy']].head())
