#knee chat

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy



import nltk
nltk.download('stopwords')



data = pd.read_excel("respuestaspacientes.xlsx")  # Archivo con tus entrevistas
text_data = data["Texto"]


# Inicializar herramientas
stop_words = stopwords.words('spanish')
nlp = spacy.load("es_core_news_sm")  # Modelo de lenguaje en español

# Función para limpiar texto
def limpiar_texto(texto):
    # Minúsculas
    texto = texto.lower()
    # Eliminar caracteres especiales
    texto = re.sub(r'[^a-záéíóúñ ]', '', texto)
    # Manejar negaciones
    texto = texto.replace("no me preocupa", "no_me_preocupa")
    # Eliminar stopwords
    palabras = texto.split()
    palabras = [palabra for palabra in palabras if palabra not in stop_words]
    # Lematización
    texto = " ".join([token.lemma_ for token in nlp(" ".join(palabras))])
    return texto

