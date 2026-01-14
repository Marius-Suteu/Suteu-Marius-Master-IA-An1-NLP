import re

# TEXT PREPROCESSING

def lowercase_text(text):
    return text.lower()

def remove_punctuation(text):
    # Keep only basic ascii letters, digits, space, %, /
    text = re.sub(r"[^a-z0-9 %/]", " ", text)
    return text

def tokenize_words(text):
    return text.split()

def tokenize_characters(text):
    chars = []
    for token in text.split():
        for c in token:
            chars.append(c)
    return chars

def normalize(text):
    text = lowercase_text(text)
    text = remove_punctuation(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# RULE-BASED MEDICAL ENGINE

medical_rules = {
    "febra": "Este posibil sa ai o infectie. Hidrateaza-te si monitoreaza temperatura.",
    "temperatura mare": "Temperatura mare poate indica o infectie sau inflamatie.",
    "tuse": "Tusea poate fi un simptom de raceala sau infectie respiratorie.",
    "durere": "Durerea generalizata poate avea multe cauze. Unde simti durerea?",
    "durere gat": "Durerea in gat poate indica o infectie virala sau bacteriana.",
    "ameteli": "Ametelile pot fi legate de tensiune sau hidratare insuficienta.",
    "greturi": "Greturile pot indica probleme digestive sau infectii.",
    "tensiune mare": "Tensiunea mare poate fi periculoasa. Daca depaseste 150/100, cere ajutor medical.",
    "tensiune mica": "Tensiunea mica poate provoca ameteli si oboseala.",
    "oxigen scazut": "Oxigenul scazut poate indica probleme respiratorii. Recomand monitorizare medicala.",
    "puls crescut": "Pulsul crescut poate indica stres, febra sau efort fizic.",
    "dureri abdominale": "Durerile abdominale pot indica o afectiune digestiva.",
    "dureri piept": "Durerile in piept pot fi grave. Recomand evaluare medicala imediata.",
    "bp": "BP inseamna Blood Pressure. Daca e mare sau mica, poate indica probleme cardiace.",
    "spo2": "SpO2 reprezinta saturatia oxigenului. Sub 92% este considerat scazut.",
    "hr": "HR inseamna Heart Rate. Pulsul normal este intre 60 si 100.",
}

# Lista de sinonime pentru a mari acoperirea
synonyms = {
    "am febra": "febra",
    "ma doare gatul": "durere gat",
    "durere in gat": "durere gat",
    "ametesc": "ameteli",
    "greata": "greturi",
    "tensiune 150/100": "tensiune mare",
    "tensiune 90/60": "tensiune mica",
    "oxigen 90": "oxigen scazut",
    "oxigen 90%": "oxigen scazut",
    "puls 120": "puls crescut",
    "durere piept": "dureri piept",
    "dureri in piept": "dureri piept",
}

# CHATBOT ENGINE

def chatbot_response(user_input):

    text = normalize(user_input)
    words = tokenize_words(text)

    # Check synonyms first
    for key in synonyms:
        if key in text:
            return medical_rules.get(synonyms[key], "Nu am o regula pentru acest simptom.")

    # Standard rule matching
    for rule in medical_rules:
        if rule in text:
            return medical_rules[rule]

    # Word-level matching
    for w in words:
        if w in medical_rules:
            return medical_rules[w]

    return "Nu am gasit informatii pentru aceasta problema. Poti descrie mai detaliat?"

# MAIN LOOP

if __name__ == "__main__":
    print("Chatbot Medical - ASCII Edition")
    print("Scrie 'exit' pentru a iesi.")
    print("---------------------------------------------")

    while True:
        user_text = input("Tu: ")
        if user_text.lower() == "exit":
            print("Chatbot: La revedere!")
            break

        response = chatbot_response(user_text)
        print("Chatbot:", response)
