## chargement des librarys
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from openpyxl import Workbook
import datetime
import os
from PIL import Image
import base64
import io


st.set_page_config(layout="wide")

# Mise en place du logo
logo_path = "C:/Users/kamil.mohamed/Desktop/application stream lit/logo_cnss23.png"  
logo = Image.open(logo_path)
st.image(logo, use_column_width=False, width=300) 

## partie CSS pour plus d'amélioration pour une futur améliorations
page_bg_img = """
<style>
body {
    background: linear-gradient(135deg, #3498db, #2c3e50); /* Gradient from blue to dark gray */
    color: #ffffff; /* White text */
}

[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0); /* Transparent header */
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
## fonction pour la declaration des variables mais aussi inserer des données dedans. 
def user_input():
    var_1 = st.number_input('Declaration en moins 1 fois au cours entre 2020 et 2022', min_value=1, max_value=2, value=1)
    st.text('Oui: 1  Non: 2')
    st.markdown("-")
    var_2 = st.number_input('Paiement en moins 1 fois entre 2020 et 2022', min_value=1, max_value=2, value=1)
    st.text('Oui: 1  Non: 2')
    st.markdown("-")
    var_3 = st.number_input("Nombre d'employé max déclaré", min_value=0, max_value=5000000, value=6000)
    var_4 = st.number_input("Age_moyen de salarié", min_value=19, max_value=100, value=19)
    var_5 = st.number_input('Salaire moyen', min_value=20000, max_value=5000000, value=50000)
    var_6 = st.number_input("Part d'hommes", min_value=0, max_value=1, value=0)
    var_7 = st.number_input("Part de femme", min_value=0, max_value=1, value=1)
    var_8 = st.number_input("Secteur d'activité", min_value=1, max_value=4, value=3)
    st.text('Administration internationale: 1   Domestique (gens de maison): 2  Secteur Privées: 3  Travailleur Independant : 4')
    st.markdown("-")
    var_9 = st.number_input("Sous Secteur", min_value=1, max_value=9, value=8)
    st.text('Activité Commerciales: 1    Activité de construction et travaux publics: 2   Activité de Production: 3   Activité de service: 4  Activités a caractères éducatifs & culturel & scientifique: 5 Administrations internationales: 6 Domestique: 7 Transport & Communication & Tourisme: 8 Travailleur Independant: 9')
    st.markdown("-")
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


st.title('Application de Prédiction Basée sur l’Apprentissage Automatique du Département DESA')
st.header("Cette application est un outil d’aide à la décision qui utilise des algorithmes d’apprentissage automatique pour prédire l’issue future d’une entreprise.")
st.markdown("---")

df = user_input()

st.subheader('Les valeurs saisie')
st.write(df)

# fichier on back in le fichier ici est un bac in qu'ils vont pas voir et qui va entrainer l'allgori
file_path = "C:/Users/kamil.mohamed/Desktop/application stream lit/fichier_alpha.xlsx"

data = pd.read_excel(file_path)

X = data.iloc[:, :-1]  # derniere variable cible

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
############################################################################################## Partie 2 ####################################################################################################""

### ici qu'il faut rajouter le dataframe et l'afficher 
st.header("Charger une liste d'entreprise.")

uploaded_file = st.file_uploader("charger le fichier data", type="xlsx")
if uploaded_file is not None:
    uploaded_file = pd.read_excel(uploaded_file)

st.header("Prediction sur une liste d'entreprise.")

# Prediction sur la liste d'entreprise.
predictions2 = None 

if uploaded_file is not None and not uploaded_file.empty:
    # Faire des prédictions avec votre modèle
    predictions2 = model.predict(uploaded_file.iloc[:, :-1])

    # Ajouter les prédictions au DataFrame en tant que nouvelle colonne 'statut'
    uploaded_file['Redressement en moins redresser fois au cours entre 2020 et 2022'] = predictions2 ### rajout de la variable prédiction

    st.dataframe(uploaded_file)
else:
    st.warning("Veuillez charger un fichier avant de faire des prédictions.")

def create_download_link_excel(df, title = "Télécharger le fichier Excel", filename = "data.xlsx"):  
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index =False)
    writer.close()
    processed_data = output.getvalue()
    b64 = base64.b64encode(processed_data)
    payload = b64.decode()
    html = f'<a download="{filename}" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{payload}" target="_blank">{title}</a>'
    return html

try:
    if predictions2 is not None and st.button('Télécharger la liste avec les résultats'):
        st.write('Téléchargement en cours...')
        
        # Créer un lien de téléchargement pour le fichier
        link = create_download_link_excel(uploaded_file, title = f"Télécharger les données enregistrées {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", filename = f"data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

        # Afficher le lien de téléchargement
        st.markdown(link, unsafe_allow_html=True)
        st.success('Téléchargement terminé!', icon="✅")
except Exception as e:
    st.error(f"Erreur lors du téléchargement : {e}")
