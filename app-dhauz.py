import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.colors import n_colors
import plotly.express as px

import association_metrics as am

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

import streamlit as st
from PIL import Image

df = pd.read_csv('cancellation-prediction.csv')

df['num_children'] = df['num_children'].fillna(0)

nums = [
    'days_between_booking_arrival', 'changes_between_booking_arrival',
    'num_adults', 'num_children', 'num_babies', 'num_weekend_nights',
    'num_workweek_nights', 'num_previous_cancellations', 'num_previous_stays',
    'required_car_parking_spaces', 'total_of_special_requests', 'avg_price',
]

df[nums] = df[nums].astype(float)

noms = [
    'cancellation', 'type', 'year_arrival_date', 'month_arrival_date',
    'week_number_arrival_date', 'day_of_month_arrival_date', 'deposit_policy',
    'country', 'breakfast', 'market_segment', 'distribution_channel',
    'customer_type'
]

paleta_dhauz = [
   '#7b3afa','#4e28a0','#ffffff','#17e5fd', '#a478fc', '#42d0e1', '#46405f'
]

fig1 = px.pie(df,
             names='cancellation',
             color_discrete_sequence=paleta_dhauz,
             hole=.3,
             template='ggplot2')

fig1.update_layout(title='Contagem de cancelamentos', showlegend=False, width=1000)

fig1.update_traces(textinfo='value+percent+label',
                   marker=dict(line=dict(color='white', width=3)))

df_corr = df[noms][:]

df_corr = df_corr.dropna(subset='country')

df_corr = df_corr.apply(lambda x: pd.factorize(x)[0])

df_corr = df_corr.apply(lambda x: x.astype('str'))

df_corr = df_corr.apply(lambda x: x.astype("category")
                        if x.dtype == "O" else x)

cramers_v = am.CramersV(df_corr)

cfit = cramers_v.fit().round(2)

fig2 = px.imshow(cfit,
                text_auto=True,
                aspect='auto',
                color_continuous_scale=px.colors.sequential.dense)

fig2.update_layout(title='Matriz de correlação nominal (Cramérs V)', width=1000)

nom_vars = list(cfit['cancellation'][1:].loc[lambda x: x > 0.11].index)
nom_vars.remove('distribution_channel')

def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta.round(2)

corrs = {}

for col in nums:
    c = correlation_ratio(df['cancellation'], df[col])
    corrs[col] = c
    
fig3 = px.histogram(x = corrs.keys(), 
                    y = corrs.values(),
                    color = corrs.keys(),
                    color_discrete_sequence=paleta_dhauz)

fig3.update_layout(title='Correlação entre a variável de interesse (cancellation) e variáveis numéricas do conjunto',
                   xaxis_title='variável numérica',
                   yaxis_title='correlação',
                   width=1000)

num_vars = [
    'days_between_booking_arrival', 'total_of_special_requests',
    'required_car_parking_spaces'
]

df = df.dropna(subset=['country'])

for col in nom_vars:
    counts = df[col].value_counts().loc[lambda x: x > 130].index
    df = df.loc[df[col].isin(counts)][:]


st.set_page_config(
    page_title="Case DHAUZ 🧠",
    layout="wide",
    initial_sidebar_state="expanded")

def main_page():
    im1 = Image.open('banner.jfif')
    st.image(im1)
    st.markdown('# Case DHAUZ 🧠- Cancelamentos em Hotéis')
    st.markdown('Victor Andrade Martins')
    st.markdown('### Briefing:')
    st.markdown('''Você foi contratado pela DHAUZ como cientista de dados para analisar uma base de dados de clientes
de uma rede de Hotéis e sua tarefa é investigar os dados em busca de insights que possam ajudar a
empresa a evitar cancelamentos e também construir um modelo preditivo que possa antecipar esses
    cancelamentos, de modo que a empresa tenha tempo hábil para agir com ações de retenção. As classes do conjunto de dados assumem a seguinte distribuição:''')
    st.plotly_chart(fig1)

def page2():
    st.markdown('## Correlações entre variáveis')
    st.plotly_chart(fig2)
    st.plotly_chart(fig3)
    
def page3():
    st.markdown('## Análise de variáveis')
    nom_var = st.selectbox('Selecione uma variável nominal para visualizar:', nom_vars)
    fig = px.histogram(df,
                       x=nom_var,
                       color=nom_var,
                       color_discrete_sequence=paleta_dhauz)

    fig.update_layout(title=f'Distribuição da variável {nom_var}', bargap=0.2, width=1000, height=500, showlegend=False)
    
    fig.update_xaxes(type='category', categoryorder='total descending')

    st.plotly_chart(fig)
    
    fig = px.histogram(df,
                       x=nom_var,
                       color='cancellation',
                       barnorm='percent',
                       text_auto='.2f',
                       color_discrete_sequence=paleta_dhauz[::-1])

    fig.update_layout(title=f'Taxas de cancelamento das variações de {nom_var}', width=1000, height=500)
    
    fig.update_xaxes(type='category')
    fig.update_yaxes(ticksuffix="%")

    st.plotly_chart(fig)
    
    num_var = st.selectbox('Selecione uma variável numérica para visualizar:', num_vars)
    
    fig = px.histogram(df,
                   x=num_var,
                   color='cancellation',
                   barmode='overlay',
                   color_discrete_sequence=paleta_dhauz)

    fig.update_layout(title=f'Distribuição de {num_var}',
                      bargap=0.2, width=1000, height=500)
    st.plotly_chart(fig)

def page4():
    st.markdown('## Modelagem preditiva')
    st.markdown('Após testar vários algoritmos, o SVC Linear foi selecionado e afinado para modelar a variável de cancelamento. Abaixo, é possível variar o conjunto a ser utilizado na modelagem.')
    
    X = df.copy()
    y = df['cancellation']
    
    ano = st.radio('Selecione o ano para filtrar a modelagem:', ['Todos', 2015, 2016, 2017])
    
    if ano != 'Todos':
        X = X.loc[X['year_arrival_date']==ano][:]
        y = df['cancellation'].loc[df['year_arrival_date']==ano][:]
    
    X = X[nom_vars + num_vars][:]

    X = pd.get_dummies(X, columns=nom_vars)
    
    scaler = MinMaxScaler()
    X[num_vars] = scaler.fit_transform(X[num_vars])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    clf = LinearSVC(random_state=0, C=0.01, fit_intercept=True, class_weight='balanced')
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)

    fig = px.imshow(cm,
                    text_auto=True,
                    color_continuous_scale=px.colors.sequential.Purples,
                    title='Matriz de confusão')

    fig.update_layout(yaxis={'title':'Reais'}, 
                      xaxis={'title':'Previstos'},
                      coloraxis_showscale=False, 
                      font_size=10,
                      width=1000, height=700)

    fig.update_xaxes(dtick=1)
    fig.update_yaxes(dtick=1)
    st.plotly_chart(fig)
    
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    
    df_report = pd.DataFrame(report)
    
    st.markdown('Tabela de métricas')
    st.table(df_report)

page_names_to_funcs = {
    "📘 Introdução": main_page,
    "📑 Análise de correlação": page2,
    "📑 Análise de variáveis": page3,
    "📑 Modelo preditivo": page4,
}

selected_page = st.sidebar.radio("Páginas", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()









