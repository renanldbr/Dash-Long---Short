import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from datetime import datetime

import dash_bootstrap_components as dbc
from dash import Dash, html, Input, Output, dcc
from dash_bootstrap_templates import load_figure_template

import plotly.graph_objects as go

import yfinance as yf
from datetime import datetime, timedelta, date

df = pd.read_csv(r'assets/ativos_b3.csv', sep=';')

linkedin_btn = html.A(
    dbc.Button(
        html.Img(src='https://www.svgrepo.com/show/922/linkedin.svg',style={'width': '30px', 'height': '30px'}),
        color='link',
        class_name='mr-2'),
    href='https://www.linkedin.com/in/renanldbr',
    target='_brank'
)

##### Botão Github IO #####  
git_hub_io = html.A(
    dbc.Button(
        html.Img(src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png',style={'width': '30px', 'height': '30px'}),
        color='link',
        class_name='mr-2'),
    href='https://renanldbr.github.io/',
    target='_brank'
)

##### Botão Github #####  
git_hub = html.A(
    dbc.Button(
        html.Img(src='data:image/png;base64,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',style={'width': '30px', 'height': '30px'}),
        color='link',
        class_name='mr-2'),
    href='https://github.com/renanldbr',
    target='_brank'
)

disclaimer = """
Aviso Legal: Este dashboard é fornecido apenas para fins didáticos e não deve ser interpretado como uma recomendação de investimento. O criador do dashboard não se responsabiliza por quaisquer operações realizadas pelo usuário com base nas informações apresentadas. É importante ressaltar que investimentos envolvem riscos e podem resultar em perdas financeiras. Recomendamos que os usuários busquem aconselhamento financeiro profissional antes de tomar qualquer decisão de investimento.

O criador do dashboard não garante a precisão, confiabilidade ou integridade das informações apresentadas e não se responsabiliza por quaisquer erros ou omissões. As informações contidas neste dashboard são fornecidas "como estão" e "conforme disponíveis" sem qualquer garantia expressa ou implícita.

Este aviso legal segue as legislações brasileiras mais atuais."""

modelo = """
O modelo de Long & Short por cointegração é uma estratégia de investimento que busca encontrar pares de ativos que mantêm um padrão de comportamento estável e previsível, com variações de curto prazo que permitem ser exploradas em operações de Long (compra) e Short (venda) do par. O conceito da estratégia é o retorno à média no resíduo de um par cointegrado. Portanto, o que se faz é definir uma determinada distância da média considerada uma discrepância para um resíduo estacionário. Uma operação de Long & Short por cointegração é iniciada, ou apresenta sinais de entrada, quando o resíduo da regressão toca os desvios padrões. Como esse resíduo é estacionário, a tendência é o retorno à média, quando dar-se a zeragem da operação.
"""

#app
app = Dash(__name__, external_stylesheets=[dbc.themes.ZEPHYR])
server = app.server

load_figure_template('zephyr')

data_final = datetime.today()
data_inicial = data_final - 3*timedelta(days=365)
data_inicial = f'{data_inicial.year}-{data_inicial.month}-{data_inicial.day}'
data_final = f'{datetime.now().year}-{datetime.now().month}-{datetime.now().day}'

app.layout = html.Div([
    dbc.Row([
        #coluna 1
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.H6('Escolha a Data Inicial:'),
                    html.H6('MM-DD-AAAA'),
                    dcc.DatePickerSingle(
                            id='data_inicial',
                            min_date_allowed=date(2010, 1, 1),
                            max_date_allowed=datetime.now(),
                            initial_visible_month=datetime.now(),
                            date=data_inicial)
                ]),
                dbc.Col([html.H6('Escolha a Data Final:'),
                        html.H6("MM-DD-AAAA"),
                        dcc.DatePickerSingle(
                            id='data_final',
                            min_date_allowed=date(2010, 1, 1),
                            max_date_allowed=datetime.now(),
                            initial_visible_month=datetime.now(),
                            date=data_final)            
                ]),
            ]),
            html.Hr(),
            html.P(modelo, style={'textAlign':'justify'}),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        html.H1('RLdBr Labs', style={'font-style':'oblique 30deg', 'margin-top':'10px','textAlign':'center'}),
                        html.Hr(),
                        html.H6('Wanna See More?', style={'textAlign':'center'}),
                        dbc.Row([
                            dbc.Col(linkedin_btn, lg=3, sm=10),
                            dbc.Col(git_hub_io, lg=3, sm=10),
                            dbc.Col(git_hub, lg=3, sm=10)
                        ], justify='center')
                    ])
                ])
            ])
        ], style={'height':'80vh', 'margin':'10px'}, md=2),
            
        #coluna 2
        dbc.Col([
            dbc.Row([
            dbc.Col([
                html.H6('Selecione o ativo:', style={'margin-top':'5px'}),
                dcc.Dropdown(df.Ticker.unique(),'PETR3.SA', style={'margin-top':'10px'}, id='drop1'),
            ]),
            dbc.Col([
                html.H6('Selecione o ativo:', style={'margin-top':'5px'}),
                dcc.Dropdown(df.Ticker.unique(),'PETR4.SA', style={'margin-top':'10px'}, id='drop2'),
            ]),
            ]),
            dbc.Row([
            dbc.Col([
                dcc.Graph(id='grafico1')
            ]),
            dbc.Col([
                dcc.Graph(id='grafico2')
            ]),
            ]),
            dbc.Row([
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Correlação entre os Ativos:", className="card-title"),
                        html.P(id='card1'),
                    ])
                )
            ]),
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Equação da Regressão:", className="card-title"),
                        html.P(id='card2'),
                    ])
                )
            ]),
            ], justify='center'),
            dbc.Row([
                    html.H1('Disclaimer', style={'margin-top':'5px'}),
                    html.P(html.I(disclaimer), style={'textAlign':'justify'})
            ])
        ], md=9)
    ])        
])
#callbacks


@app.callback(
    Output(component_id='grafico1', component_property='figure'),
    Output(component_id='grafico2', component_property='figure'),
    Output(component_id='card1', component_property='children'),
    Output(component_id='card2', component_property='children'),
    
    Input(component_id='drop1', component_property='value'),
    Input(component_id='drop2', component_property='value'),
    Input(component_id='data_inicial', component_property='date'),
    Input(component_id='data_final', component_property='date'))

def atts(stock1, stock2, data_inicial, data_final):
    #função para o grafico 1:
    df_stock1 = yf.download(stock1, start=data_inicial, end=data_final)['Adj Close']
    df_stock2 = yf.download(stock2, start=data_inicial, end=data_final)['Adj Close']
    
    ticker1 = stock1.split('.')[0]
    ticker2 = stock2.split('.')[0]

    df_stock1 = pd.DataFrame(df_stock1)
    df_stock1.reset_index(names=['Date'], inplace=True)
    df_stock2 = pd.DataFrame(df_stock2)
    df_stock2.reset_index(names=['Date'], inplace=True)

    df_zscore = pd.merge(df_stock1, df_stock2, on='Date')
    df_zscore.rename(columns={'Adj Close_x':ticker1, 'Adj Close_y':ticker2}, inplace=True)
        
    grafico1 = go.Figure()
    grafico1.add_trace(go.Scatter(x=df_zscore['Date'], y=df_zscore[ticker1], mode='lines', name=ticker1))
    grafico1.add_trace(go.Scatter(x=df_zscore['Date'], y=df_zscore[ticker2], mode='lines', name=ticker2))
    grafico1.update_layout(title_text=f'{ticker1}x{ticker2}')
    #função para o grafico 2:

        #Regressão Linear
    X = df_zscore[ticker1].values.reshape(-1,1)
    Y = df_zscore[ticker2].values.reshape(-1,1)
    reg = LinearRegression().fit(X, Y)
    y_hat = reg.predict(X)
    residual = Y - y_hat
    df_zscore['Residual'] = residual
    df_zscore['ResidualPad'] = (df_zscore['Residual'] - df_zscore['Residual'].mean()) / df_zscore['Residual'].std()
    
    grafico2 = go.Figure()
    grafico2.add_trace(go.Scatter(x=df_zscore['Date'], y=df_zscore['ResidualPad']))
    grafico2.add_hline(y=2, line_dash='dot')
    grafico2.add_hline(y=-2, line_dash='dot')
    grafico2.update_layout(title_text='Zscore')

    #Informações para os cards
    corr = np.corrcoef(df_zscore[ticker1], df_zscore[ticker2])
    card1 = f'A correlação entre os ativos é de {corr[0][1]:.4f}'

    a = reg.intercept_
    b = reg.coef_
    if b>0:
        card2 = f'{ticker1} = {a[0]:.4f}+{b[0][0]:.4f}x{ticker2}'
    else:
        card2 = f'{ticker1} = {a[0]:.4f}{b[0][0]:.4f}x{ticker2}'

    for grafico in [grafico1, grafico2]:
        grafico.update_layout(template='zephyr')

    return [grafico1, grafico2, card1, card2]
    


#server
if __name__ == '__main__':
    app.run_server(debug = False)
