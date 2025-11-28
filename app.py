from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from textblob import TextBlob
import string
import emoji
import re
from contractions import fix
from nltk.corpus import stopwords

def word_tokenizer(text, tokenizer_type):
    tokens = []
    if tokenizer_type == "nltk":
        tokens = word_tokenize(text)
    elif tokenizer_type == "split":
        tokens = text.split() 
    elif tokenizer_type == "spaCy":
        nlp = spacy.load("en_core_web_sm")  # Changé à modèle anglais
        doc = nlp(text)
        tokens = [token.text for token in doc]
    elif tokenizer_type == "TextBlob":
        blob = TextBlob(text)
        tokens = blob.words
    else:
        raise ValueError("Tokenizer type not recognized")
    return tokens

def sentence_tokenizer(text, tokenizer_type):
    sentences = []
    if tokenizer_type == "nltk":
        sentences = sent_tokenize(text)
    elif tokenizer_type == "split":
        sentences = [sentence.strip() for sentence in text.split(".") if sentence.strip()]
    elif tokenizer_type == "spaCy":
        nlp = spacy.load("en_core_web_sm")  # Changé à modèle anglais
        doc = nlp(text)
        sentences = [sentence.text.strip() for sentence in doc.sents]
    elif tokenizer_type == "TextBlob":
        blob = TextBlob(text)
        sentences = [str(sentence) for sentence in blob.sentences]
    else:
        raise ValueError("Tokenizer type not recognized")
    return sentences

def normalize(text, operation): 
    if operation == "Lowercasing":
        return text.lower()
    elif operation == "Contraction Correction":
        return fix(text)
    elif operation == "Punctuation Removal":
        tokens = word_tokenize(text)
        filtered_tokens = [i for i in tokens if i not in string.punctuation]
        return ' '.join(filtered_tokens)
    elif operation == "Multiple Spaces Removal":
        return re.sub(r'\s+', ' ', text).strip() 
    elif operation == "Spelling Correction":
        blob = TextBlob(text)
        return str(blob.correct())
    elif operation == "Emoji to Text Conversion":
        return emoji.replace_emoji(text, replace="")
    else:
        raise ValueError("Normalization operation not recognized")
    
def remove_stop_words(tokens): 
    stop_words = set(stopwords.words('english'))  # Déjà en anglais
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens

def POS_tag(tokens):
    nlp = spacy.load("en_core_web_sm")  # Changé à modèle anglais
    doc = nlp(" ".join(tokens))
    pos_tags = [(token.text, token.pos_) for token in doc]
    return pos_tags