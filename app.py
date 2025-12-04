import streamlit as st
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Football Predictor", layout="centered")
st.title("⚽ Football Match Predictor")
st.write("Interaktivni demo koji prikazuje kako ML predviđa ishod utakmice.")

# Unosi
forma_home = st.slider("Forma domaćih (pobjede u zadnjih 6)", 0, 6, 3)
forma_away = st.slider("Forma gostiju (pobjede u zadnjih 6)", 0, 6, 3)
gol_diff = st.slider("Prosječna gol razlika domaćih", -2.0, 2.0, 0.0)
injuries = st.slider("Ozljede domaćih igrača", 0, 5, 0)
home_adv = st.checkbox("Igra se doma?", value=True)

# Demo podaci
X = np.array([
    [4,2,1,0,1],
    [1,4,-1,2,1],
    [3,3,0,1,1],
    [5,1,1.5,0,1],
    [0,5,-1.5,2,0]
])
y = np.array([0,2,1,0,2])  # 0=home, 1=draw, 2=away

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

model = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=500)
model.fit(X_scaled, y)

# Predikcija
input_data = np.array([[forma_home, forma_away, gol_diff, injuries, int(home_adv)]])
input_scaled = sc.transform(input_data)

pred = model.predict(input_scaled)[0]
labels = ["🏠 Home Win", "🤝 Draw", "✈️ Away Win"]

st.header("🔮 Predviđanje:"labels[pred])
st.header(labels[pred])
