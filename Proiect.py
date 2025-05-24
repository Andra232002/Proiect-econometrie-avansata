# Importăm biblioteca pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Citim fișierul CSV
df = pd.read_csv("Baza_date.csv")

# Afișăm primele 5 rânduri din baza de date pentru a verifica că s-a încărcat corect
print("Primele 5 rânduri din baza de date:")
print(df.head())

# Statistici descriptive (media, min, max, etc.)
print("\nStatistici descriptive:\n")
print(df.describe())

#Analiza exploratorie a datelor

# 7. Histogramă pentru fiecare variabilă cantitativă
df.hist(bins=15, figsize=(12, 10), color='skyblue', edgecolor='black')
plt.suptitle("Distribuția variabilelor cantitative", fontsize=16)
plt.tight_layout()
plt.show()

# 8. Heatmap cu corelațiile dintre variabile numerice
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matricea de corelație între variabile")
plt.show()

# 9. Boxplot pentru a detecta valori extreme (outliers)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.select_dtypes(include='number'))
plt.title("Boxplot pentru detectarea valorilor extreme")
plt.xticks(rotation=45)
plt.show()


