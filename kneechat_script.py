import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import KMeans
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')

# Cargar el dataset
df = pd.read_excel("respuestas2.xlsx")

# Crear un nuevo dataframe para almacenar los datos expandidos
df_ampliado = pd.DataFrame(columns=['Hablante', 'Entrevista', 'sexo', 'Frase'])

# Iterar sobre cada fila y expandir las frases
for _, row in df.iterrows():
    frases = sent_tokenize(row['Texto'])  # Dividir el texto en frases
    for frase in frases:
        # Añadir una fila por cada frase manteniendo las otras columnas
        df_ampliado = pd.concat(
            [df_ampliado, pd.DataFrame({
                'Hablante': [row['Hablante']],
                'Entrevista': [row['Entrevista']],
                'sexo': [row['sexo']],
                'Frase': [frase]
            })],
            ignore_index=True
        )

# Mostrar el nuevo dataframe

df_ampliado.to_excel("frases.xlsx", index=False)


file_path = 'frases.xlsx'  # Ajusta el nombre si es necesario
responses_data = pd.read_excel(file_path)

texts = responses_data['Frase'].astype(str).tolist()

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def generate_embeddings(texts, tokenizer, model):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())
    return embeddings

embeddings = generate_embeddings(texts, tokenizer, model)


# Clustering con KMeans para agrupar respuestas similares
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(embeddings)

responses_data['Cluster'] = clusters



# Suponiendo que tu DataFrame se llama df y la columna original es "categorias"
mapping = {
    0: "Actitudes reflexivas",
    1: "Afirmaciones rápidas",
    2: "mix",
    3: "Preguntas cortas",
    4: "Sin dudas claras"
}

# Crear la nueva columna "labels" usando el mapeo
responses_data['Temas'] = responses_data['Cluster'].map(mapping)


responses_data.to_excel("clustering5.xlsx", index=False)

print(responses_data)