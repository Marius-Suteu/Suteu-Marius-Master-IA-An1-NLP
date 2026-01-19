import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # remove URLs
    text = re.sub(r"<.*?>", "", text)            # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)        # remove non-letter characters
    text = re.sub(r"\s+", " ", text)             # normalize spaces
    return text.strip()

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = clean_text(text)
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha
        and token.text not in STOP_WORDS
        and len(token) > 2
    ]
    return " ".join(tokens)