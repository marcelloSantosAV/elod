import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Configuração inicial para mobile
st.set_page_config(page_title="eLoad Mobile", layout="centered")

st.title("eLoad - Controle de Carga de Treinamento (Mobile)")
st.write("Acesse todas as funcionalidades do eLoad diretamente no seu dispositivo móvel.")

# Questionário de Bem-Estar
st.subheader("Questionário de Bem-Estar")
sleep_quality = st.radio("Como foi a qualidade da sua noite de sono?", ['Péssima', 'Ruim', 'Normal', 'Boa', 'Ótima'])
disposition = st.radio("Como está sua disposição hoje?", ['Péssima', 'Ruim', 'Normal', 'Boa', 'Ótima'])
muscle_pain = st.radio("Como está a sua dor muscular hoje?", ['Péssima', 'Ruim', 'Normal', 'Boa', 'Ótima'])
stress_level = st.radio("Como está o seu nível de stress hoje?", ['Péssima', 'Ruim', 'Normal', 'Boa', 'Ótima'])
mood = st.radio("Como está o seu humor hoje?", ['Péssima', 'Ruim', 'Normal', 'Boa', 'Ótima'])

if st.button('Concluir Questionário'):
    st.success('Questionário concluído com sucesso!')

# Entrada de dados via mobile
athlete_name = st.text_input("Nome do Atleta")
training_session_duration = st.number_input("Duração do Treino (minutos)")
perceived_exertion = st.slider("Percepção de Esforço (1-10)", 1, 10)
wellbeing_score = st.slider("Pontuação de Bem-Estar (1-10)", 1, 10)

# Função para calcular carga de treino
def calculate_training_load(duration, exertion):
    return duration * exertion

training_load = calculate_training_load(training_session_duration, perceived_exertion)

# Exibir dados de treino
st.write(f"Carga de Treino: {training_load}")
st.write(f"Pontuação de Bem-Estar: {wellbeing_score}")

# Dados históricos de treino para o modelo
data = {
    "Carga_Treino": [300, 400, 350, 450, 500, 420, 380],
    "Bem_Estar": [7, 6, 8, 5, 4, 7, 6],
    "Lesao": [0, 1, 0, 1, 1, 0, 0]
}

df = pd.DataFrame(data)
X = df[["Carga_Treino", "Bem_Estar"]]
y = df["Lesao"]

# Treinamento do modelo de previsão de lesão
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Avaliação do modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Acurácia do Modelo: {accuracy * 100:.2f}%")

# Função para prever risco de lesão
def predict_injury_risk(load, wellbeing):
    return model.predict([[load, wellbeing]])[0]

# Previsão de risco de lesão
injury_risk = predict_injury_risk(training_load, wellbeing_score)
st.write(f"Risco de Lesão: {'Alto' if injury_risk else 'Baixo'}")

# Visualização de dados históricos
weekly_loads = [300, 400, 350, 450, 500]
days = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex']

plt.figure(figsize=(10, 5))
plt.plot(days, weekly_loads, marker='o')
plt.title("Carga de Treino Semanal")
plt.xlabel("Dia")
plt.ylabel("Carga de Treino")
plt.grid(True)
st.pyplot(plt)

# Exportação de relatório
data = {
    "Atleta": [athlete_name],
    "Duração do Treino": [training_session_duration],
    "Percepção de Esforço": [perceived_exertion],
    "Carga de Treino": [training_load],
    "Bem-Estar": [wellbeing_score]
}

df = pd.DataFrame(data)
st.dataframe(df)

st.download_button(
    label="Baixar Relatório",
    data=df.to_csv(index=False),
    file_name=f"relatorio_{athlete_name}.csv",
    mime='text/csv'
)
