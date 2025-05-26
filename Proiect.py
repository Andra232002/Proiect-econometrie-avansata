# Importăm bibliotecile
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

# Histogramă pentru fiecare variabilă cantitativă
# df.hist(bins=15, figsize=(12, 10), color='skyblue', edgecolor='black')
# plt.suptitle("Distribuția variabilelor cantitative", fontsize=16)
# plt.tight_layout()
# plt.show()


# Histogramă pentru fiecare variabilă cantitativă
# df.hist(bins=15, figsize=(12, 10), color='skyblue', edgecolor='black')

# # Titlu general pentru grafice
# plt.suptitle("Distribuția variabilelor cantitative", fontsize=16)

# # Aranjare automată a elementelor din figură
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # rect e folosit pentru a face loc titlului

# # Salvarea imaginii în folderul curent (Proiect)
# plt.savefig("distributii_variabile.png", dpi=300)

# plt.show()




# Lista variabilelor cantitative (excluzând 'tara')
variabile_cantitative = ['AccesElectricitate%', 'Populatie', 'NrNasteri', 
                         'Natalitate_(pop/nr,nasteri)', 'HumanDevelopmentIndex(HDI)', 'LifeExpectancyAtBirth', 'ExpectedYearsOfSchooling']

# Setăm dimensiunea figurii și layout-ul subploturilor
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
axs = axs.ravel()  # transformăm matricea 2D de axe într-o listă simplă

# Iterăm prin fiecare variabilă și o desenăm
for i, var in enumerate(variabile_cantitative):
    axs[i].hist(df[var].dropna(), bins=15, color='skyblue', edgecolor='black')
    axs[i].set_title(f'Histograma pentru {var}', fontsize=12)
    axs[i].set_xlabel(var)
    axs[i].set_ylabel('Frecvență')

# Eliminăm subploturile goale (dacă sunt mai multe subploturi decât variabile)
for j in range(i+1, len(axs)):
    fig.delaxes(axs[j])

# Ajustăm spațierea
plt.tight_layout()

# Salvăm imaginea
plt.savefig("histograme_variabile_cantitative.png", dpi=300)

# Afișăm imaginea
plt.show()

# Heatmap cu corelațiile dintre variabile numerice
# plt.figure(figsize=(10, 8))
# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Matricea de corelație între variabile")
# plt.show()

# Boxplot pentru a detecta valori extreme (outliers)
# plt.figure(figsize=(12, 6))
# sns.boxplot(data=df.select_dtypes(include='number'))
# plt.title("Boxplot pentru detectarea valorilor extreme")
# plt.xticks(rotation=45)
# plt.show()


