# Importăm biblioteca pandas
import pandas as pd

# Citim fișierul CSV
df = pd.read_csv("Baza_date.csv")

# Afișăm primele 5 rânduri din baza de date pentru a verifica că s-a încărcat corect
print("Primele 5 rânduri din baza de date:")
print(df.head())

# Statistici descriptive (media, min, max, etc.)
print("\nStatistici descriptive:\n")
print(df.describe())
