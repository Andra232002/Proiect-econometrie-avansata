# %%
#from google.colab import drive

#drive.mount('/content/drive/')

# %%
import pandas as pd
# print("Hello")

import matplotlib.pyplot as plt
import seaborn as sns

# Deschidere doc. excel
#df = pd.read_excel('/content/drive/MyDrive/Colab Notebooks/Econometrie_avansata_proiect/Baza de date.xlsx')

# %%
# Previzualizare date
print("----- Primele 5 rânduri -----")
print(df.head())

# Informații despre tipurile de date
print("\n----- Info despre dataframe -----")
print(df.info())

# Verificare valori lipsă
print("\n----- Valori lipsă pe coloană -----")
print(df.isnull().sum())

# Statistici descriptive
print("\n----- Statistici descriptive pentru variabile numerice -----")
print(df.describe())

# %% [markdown]
# ## **1) Observăm că există valori lipsă**

# %% [markdown]
# Pentru tratarea problemei valorilor lipsă, grupăm țările în funcție de HDI. Țările cu un indice al dezvoltării umane asemănător au de obicei valori similare pentru restul variabilelor

# %%
# 1. Creare grupe după HDI

def clasificare_HDI(hdi):
    if hdi >= 0.8:
        return 'ridicat'
    elif hdi >= 0.65:
        return 'mediu'
    else:
        return 'scazut'

df['Nivel_dezvoltare'] = df['HumanDevelopmentIndex(HDI) '].apply(clasificare_HDI)

# %%
# 2. Imputarea medianei pe grupuri

for col in ['AccesElectricitate%', 'Populatie', 'NrNasteri', 'LifeExpectancyAtBirth', 'ExpectedYearsOfSchooling']:
    df[col] = df.groupby('Nivel_dezvoltare')[col].transform(lambda x: x.fillna(x.median()))

# %%
# 3. Verificăm dacă mai există valori lipsă
print("\n----- Valori lipsă pe coloană -----")
print(df.isnull().sum())

# %% [markdown]
# Din moment ce acum există o singură valoare lipsă în întreaga bază de date, o putem elimina, deoarece această observație nu are un impact puternic asupra analizei noastre.

# %%
df = df.dropna(subset=['HumanDevelopmentIndex(HDI) '])

# Verificăm dacă mai există valori lipsă
print("\n----- Valori lipsă pe coloană -----")
print(df.isnull().sum())

# %% [markdown]
# ## Continuăm analiza descriptivă

# %%
# Matrice de corelație (numerice)
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("----- Corelația între variabile numerice -----")
plt.show()

# %%
# Histogramă pentru fiecare variabilă numerică

numeric_cols = df.select_dtypes(include=['float64']).columns
num_cols = len(numeric_cols)
cols = 4  # câte coloane de subploturi
rows = (num_cols + cols - 1) // cols  # câte rânduri sunt necesare

plt.figure(figsize=(15, 10))

for i, col in enumerate(numeric_cols, 1):
    ax = plt.subplot(rows, cols, i)
    ax.hist(df[col].dropna(), bins=20, edgecolor='black')
    ax.set_title(col)

    # Axa x: scoate notația științifică
    ax.ticklabel_format(style='plain', axis='x')
    ax.tick_params(axis='x', rotation=45)  # opțional: rotește etichetele dacă sunt lungi

plt.suptitle("----- Distribuții variabile numerice -----", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
# ## 2) Observăm outlieri în cazul variabilelor *Populație* și *NrNasteri*

# %% [markdown]
# Am ales să rezolvăm problema valorilor aberante folosind logaritmarea

# %%
# Prelucrarea datelor din Populație

import numpy as np

df['Populatie_log'] = np.log1p(df['Populatie'])  # log(1 + x)

# Creăm graficele pentru a observa cum s-a schimbat distribuția valorilor
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Histograma originală
axs[0].hist(df['Populatie'].dropna(), bins=30)
axs[0].set_title("Distribuția originală - Populatie")
axs[0].set_xlabel("Populatie")
axs[0].set_ylabel("Frecvență")
axs[0].ticklabel_format(style='plain', axis='x')  # scoate notatia 1e6

# Histograma transformată
axs[1].hist(df['Populatie_log'].dropna(), bins=30, color='orange')
axs[1].set_title("Distribuția log-transformată - Populatie_log")
axs[1].set_xlabel("log(1 + Populatie)")
axs[1].set_ylabel("Frecvență")

plt.suptitle("Compararea distribuției înainte și după transformarea logaritmică", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# %%
# Prelucrarea datelor din NrNasteri

df['NrNasteri_log'] = np.log1p(df['NrNasteri'])  # log(1 + x)

# Creăm graficele pentru a observa cum s-a schimbat distribuția valorilor
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Histograma originală
axs[0].hist(df['NrNasteri'].dropna(), bins=30)
axs[0].set_title("Distribuția originală - NrNasteri")
axs[0].set_xlabel("NrNasteri")
axs[0].set_ylabel("Frecvență")
axs[0].ticklabel_format(style='plain', axis='x')  # scoate notatia 1e6

# Histograma transformată
axs[1].hist(df['NrNasteri_log'].dropna(), bins=30, color='orange')
axs[1].set_title("Distribuția log-transformată - NrNasteri_log")
axs[1].set_xlabel("log(1 + NrNasteri)")
axs[1].set_ylabel("Frecvență")

plt.suptitle("Compararea distribuției înainte și după transformarea logaritmică", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# %% [markdown]
# ## **Estimarea unui model de regresie liniară clasică**

# %% [markdown]
# Alegem ca variabile explicative rata populației cu acces la electricitate, populația (valoarea logaritmată), numărul de nașteri (valoarea logaritmată), speranța de viață, numărul așteptat de ani de școlarizare. Nu includem variabilele *natalitate* și *nivel de dezvoltare*, acestea fiind derivate din populație și numărul de nașteri, respectiv derivată din HDI

# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# %%
# Selectarea variabilelor explicative
X = df[['AccesElectricitate%', 'Populatie_log', 'NrNasteri_log',
        'LifeExpectancyAtBirth', 'ExpectedYearsOfSchooling']]

# Variabila dependentă
y = df['HumanDevelopmentIndex(HDI) ']

# %%
# Analiza statistică a modelului de regresie liniară clasică OLS
X_const = sm.add_constant(X)
ols_model = sm.OLS(y, X_const).fit()
print("\nRezumat statsmodels:")
print(ols_model.summary())

# %% [markdown]
# În urma aplicării modelului OLS pe setul nostru de date, obținem un R-squared = 0.9 și Prob (F-statistic) = 1.36e-84, ceea ce arată o calitate crescută a modelului.
# 
# Cu excepția variabilei AccesElectricitate%, toate celelalte variabile explicative sunt semnificative statistic pentru un prag de 0.05, așadar variabila menționată poate fi exclusă din modelul OLS.
# 
# De asemenea, valoarea testului Durbin-Watson = 2.037 (foarte apropiată de 2) arată că reziduurile nu sunt autocorelate. Trebuie ținut cont și de faptul că reziduurile nu sunt distribuite normal (testul Jarque-Bera are o valoare foarte crescută)
# 
# Refacem modelul, de această dată fără variabila AccesElectricitate%

# %%
# Noua selecție de variabile explicative (fără AccesElectricitate%)
X_nou = df[['Populatie_log', 'NrNasteri_log', 'LifeExpectancyAtBirth', 'ExpectedYearsOfSchooling']]
y_nou = df['HumanDevelopmentIndex(HDI) ']  # ținta rămâne aceeași

# Analiza statistică a modelului de regresie liniară clasică OLS
X_const_nou = sm.add_constant(X_nou)
ols_model_nou = sm.OLS(y_nou, X_const_nou).fit()
print("\nRezumat statsmodels:")
print(ols_model_nou.summary())

# %% [markdown]
# Putem observa că eliminarea variabilei AccesElectricitate% a îmbunătățit interpretabilitatea modelului fără a-i reduce performanța. Toate cele 4 variabile explicative sunt acum semnificative statistic

# %% [markdown]
# ## **Estimarea unui model de regresie liniară prin învățare supervizată**

# %%
# Împărțirea în set de antrenare și test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelul de regresie
model = LinearRegression()
model.fit(X_train, y_train)

# Predicții
y_pred = model.predict(X_test)

# %%
# Evaluare
print("R² score:", r2_score(y_test, y_pred))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

# %% [markdown]
# Valoarea lui R-squared arată faptul că modelul explică 82,6% din variația indicelui de dezvoltare umană.

# %% [markdown]
# 
# Comparăm performanța testului de antrenare cu cea a setului de test:

# %%
# Scor pe setul de antrenare
y_train_pred = model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

print("Train R²:", r2_train)
print("Train RMSE:", rmse_train)

# %% [markdown]
# Se observă o diferență de aproape 0,1 între R-squared pentru setul de antrenare și cel de test, iar valorile pentru RMSE nu diferă nici ele semnificativ, ceea ce indică faptul că nu este prezent cazul de overfitting.

# %%
# Coeficienți
coef_df = pd.DataFrame({
    'Variabilă': X.columns,
    'Coeficient': model.coef_
})
print("\nCoeficienți:")
print(coef_df)

# %%
# Statistici descriptive
print("\n----- Statistici descriptive pentru variabile numerice -----")
print(df.describe())

# %% [markdown]
# # Regresia polinomială

# %% [markdown]
# Vom folosi termeni polinomiali și interacțiuni între variabile pentru a captura relații non-lineare între predictori și variabila țintă.

# %%
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# %%
# Pregătirea datelor
X = df[['Populatie_log', 'NrNasteri_log', 'LifeExpectancyAtBirth', 'ExpectedYearsOfSchooling']]
y = df['HumanDevelopmentIndex(HDI) ']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Construirea modelului polinomial grad 2
degree = 2
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Antrenarea modelului
poly_model.fit(X_train, y_train)

# Predicții
y_pred = poly_model.predict(X_test)

# %%
# Evaluare
print(f"R² score: {r2_score(y_test, y_pred):.4f}")

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

# %% [markdown]
# R-squared pentru regresia liniară simplă (0.9) este mai mare decât cel obținut folosind regresia polinomială (0.787)

# %% [markdown]
# # Random Forest

# %% [markdown]
# Random Forest este un model non-liniar, robust și des folosit pentru regresiile cu mai multe variabile explicative, pentru a captura relații complexe.

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# %%
# Datele selectate
X = df[['Populatie_log', 'NrNasteri_log', 'LifeExpectancyAtBirth', 'ExpectedYearsOfSchooling']]
y = df['HumanDevelopmentIndex(HDI) ']

# Împărțirea datelor în train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Definirea modelului Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Antrenarea modelului
rf.fit(X_train, y_train)

# Predicții pe test
y_pred = rf.predict(X_test)

# %%
# Evaluarea modelului
print(f"R² score: {r2_score(y_test, y_pred):.4f}")

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

# %% [markdown]
# # Feature Selection

# %% [markdown]
# Pentru a îmbunătăți performanța modelelor și a obține o reprezentare mai relevantă a relației dintre variabilele explicative și indicele de dezvoltare umană (HDI), am explorat două abordări complementare:

# %% [markdown]
# 1. Eliminarea variabilelor informativ slabe
# 
# Am aplicat Regresia Lasso pentru a identifica cele mai importante variabile explicative. Acest model penalizează coeficienții irelevanți, împingându-i spre zero, ajutând astfel la selecția caracteristicilor relevante.

# %%
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# Standardizare
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso cu cross-validation
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, y)

# Coeficienți
selected_features = pd.Series(lasso.coef_, index=X.columns)
print(selected_features)

# %% [markdown]
# 2. Crearea de variabile noi (Feature Engineering)
# Am creat o variabilă derivată, deja prezentă în set, dar ignorată anterior:
# 
# Natalitate relativă = Populație / Număr nașteri (notată Natalitate_(pop/nr.nasteri))
# 
# Aceasta exprimă o măsură inversă a ratei natalității, oferind o perspectivă interesantă asupra demografiei.
# 
# Am testat un model Random Forest folosind doar aceste trei variabile

# %%
X_fs = df[['Natalitate_(pop/nr.nasteri)', 'LifeExpectancyAtBirth', 'ExpectedYearsOfSchooling']]
y_fs = df['HumanDevelopmentIndex(HDI) ']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_fs, y_fs, test_size=0.2, random_state=42)

# Model
rf_fs = RandomForestRegressor(random_state=42)
rf_fs.fit(X_train, y_train)
y_pred = rf_fs.predict(X_test)

# Evaluare
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred)**0.5:.4f}")


# %%


# %%


# %%


# %%


# %%
# Statistici descriptive
print("\n----- Statistici descriptive pentru variabile numerice -----")
print(df.describe())

# %%


# %% [markdown]
#   OLS_0:
# R-squared = 0.9
# Prob (F-statistic) = 385.2
# DW = 1.999
# JB = 671.969
# 
# 
#   Regresie liniara - supervizat antrenare:
# R-squared = 0.826
# RMSE = 0.0703
# 
#   Regresie liniara - supervizat test:
# R-squared = 0.921
# RMSE = 0.0411
# 
#   Regresia polinomială:
# R² score: 0.7874  RMSE: 0.0778
# 
#   Random forest:
# R² score: 0.7768
# RMSE: 0.0797

# %%


# %%


# %%



