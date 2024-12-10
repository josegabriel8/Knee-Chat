import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel("clustering5.xlsx")

# Configuración del estilo de seaborn
sns.set(style="whitegrid")

# Crear el gráfico de barras directamente con seaborn
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Entrevista', hue='Temas',  palette='pastel')

# Personalizar el gráfico
plt.title('Frecuencia de Temas', fontsize=16)
plt.xlabel('Temas', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Mostrar el gráfico
plt.show()