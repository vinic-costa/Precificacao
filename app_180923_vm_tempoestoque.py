import streamlit as st
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import joblib
from sklearn.preprocessing import StandardScaler

st.set_option('server.basePath', '/Precificacao')

def apply_log_transformation(data):
    return np.log(data)

## -- Modelo  -- #
pkl_file_path = 'precificacao_180923_joblib.pkl'
    
# Carregar o arquivo .pkl com joblib
with open(pkl_file_path, 'rb') as model_file:
    modelo = joblib.load(model_file)
    
# Verificar o tipo do objeto carregado
#st.write(type(modelo))


# Obter valores do usuário
def user_input_features():
    st.sidebar.header("Entre com os dados do produto")
    valor_mercado = st.sidebar.number_input('Digite o valor de mercado do produto:', min_value=0, step=1)
    tempo_estoque= st.sidebar.number_input('Digite o valor de tempo_estoque do produto:', min_value=0, step=1)

    # Verificar se os valores são maiores que zero antes de aplicar o logaritmo
    if valor_mercado > 0 and tempo_estoque > 0:
        valor_mercado_log = apply_log_transformation(valor_mercado)
        tempo_estoque_log = apply_log_transformation(tempo_estoque)
        return np.array([valor_mercado_log, tempo_estoque_log]).reshape(1, -1)
    else:
        st.warning("Os valores de entrada devem ser maiores que zero.")
        # Se os valores não forem válidos, você pode retornar um array de zeros ou valores padrão.
        return np.array([0, 0]).reshape(1, -1)

st.header("Predição de preços Tag2U")
st.write("Este aplicativo faz a predição de preços para itens da Tag2u")

input_features = user_input_features()

if input_features is not None:
    st.subheader("Entrada do usuário")
    st.write(pd.DataFrame(np.exp(input_features), columns=['valor_mercado', 'tempo_estoque']))

    prediction_log = modelo.predict(input_features)

    def reverse_log_transformation(log_value):
        return np.exp(log_value)

    if len(prediction_log) > 0:
        prediction = reverse_log_transformation(prediction_log[0])
        st.subheader("Valor predito")
        st.write(f"${prediction:.2f}")