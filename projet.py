import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from openpyxl import Workbook


def user_input():
    var_1 = st.number_input('Declaration en moins 1 fois au cours entre 2020 et 2022', min_value=1, max_value=2, value=1)
    var_2 = st.number_input('Paiement en moins 1 fois entre 2020 et 2022', min_value=1, max_value=2, value=1)
    var_3 = st.number_input("Nombre d'employé max déclaré", min_value=0, max_value=5000000, value=6000)
    var_4 = st.number_input("Age_moyen de salarié", min_value=19, max_value=100, value=19)
    var_5 = st.number_input('Salaire moyen', min_value=20000, max_value=5000000, value=50000)
    var_6 = st.number_input("Part d'hommes", min_value=0, max_value=1, value=0)
    var_7 = st.number_input("Part de femme", min_value=0, max_value=1, value=1)
    var_8 = st.number_input("Secteur d'activité", min_value=1, max_value=4, value=3)
    var_9 = st.number_input("Sous Secteur", min_value=1, max_value=9, value=8)
    var_10 = st.number_input("Nombre de salarié accidenté entre 2020 et 2022", min_value=0, max_value=100, value=80)
    var_11 = st.number_input("Nombre de femme en congé maternité", min_value=0, max_value=100, value=10)
    var_12 = st.number_input("Type d'affiliation", min_value=0, max_value=1, value=0)
    var_13 = st.number_input('Debouche_moyen', min_value=0, max_value=100, value=4)

    data = {'Declaration en moins 1 fois au cours entre 2020 et 2022': var_1,
            'Paiement en moins 1 fois entre 2020 et 2022': var_2,
            "Nombre d'employé max déclaré":var_3,
            "Age_moyen de salarié": var_4,
            'Salaire moyen': var_5,
            "Part d'hommes": var_6,
            "Part de femme": var_7,
            "Secteur d'activité": var_8,
            "Sous Secteur": var_9,
            "Nombre de salarié accidenté entre 2020 et 2022": var_10,
            "Nombre de femme en congé maternité": var_11,
            "Type d'affiliation": var_12,
            'Debouche_moyen': var_13
            }
    features = pd.DataFrame(data, index=[0])
    return features



st.header('Application de Prédiction Basée sur l’Apprentissage Automatique pour la Production du Département DESA', divider = "blue")

st.write("Cette application est un outil d’aide à la décision qui utilise des algorithmes d’apprentissage automatique pour prédire l’issue future d’une entreprise.")
st.markdown("---")
df = user_input()

st.subheader('Les valeurs saisie')
st.write(df)

# Upload the dataset and read it into a pandas dataframe
uploaded_file = st.file_uploader("charger le fichier data", type="xlsx")
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)

    X = data.iloc[:, :-1]  # assuming the last column is the target variable
    def convert_columns_to_int(df):
        df = df.fillna(0)
        return df.astype(int, errors = 'ignore')
    
    X = convert_columns_to_int(X)

    Y = data.iloc[:, -1]

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, Y)
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)
    prediction_proba = prediction_proba.astype(float)

    st.subheader('Prediction')

    st.write(prediction)
    st.write(prediction_proba)
