import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from binance.client import Client
from dash import Dash, dcc, html, Input, Output

import pred

api_key = ''
api_secret = ''
apiClient = Client(api_key, api_secret)

app = Dash(external_stylesheets=[dbc.themes.MATERIA], title='Stock Price Prediction')

stock_codes = ['AAPL', 'AMZN', 'FB', 'GOOG', 'MSFT', 'TWTR']
prediction_methods = ['LSTM', 'RNN', 'XGBoost']

stock_code_dropdown = dcc.Dropdown(
    options=stock_codes, id="stock_code_dropdown",
    value=stock_codes[0],
    clearable=False,
    searchable=True,
    style={'width': '100px'}
)

prediction_method_dropdown = dcc.Dropdown(
    options=prediction_methods, id="prediction_method_dropdown",
    value=prediction_methods[0],
    clearable=False,
    searchable=True,
    style={'width': '100px'}
)

prediction_spec_dropdown = dcc.Dropdown(
    options=['Close', 'Price of Change'], id="prediction_spec_dropdown",
    value=[1, 2],
    clearable=False,
    searchable=True,
    style={'width': '240px'}
)

navbar = html.Div(
    children=[
        stock_code_dropdown,
        prediction_method_dropdown,
        prediction_spec_dropdown
    ],
    style={
        'display': 'flex',
        'flex-direction': 'row',
        'justify-content': 'space-evenly',
        'background-color': '#e3f2fd',
        'padding': '20px'
    }
)

@app.callback(
    [
        Output('graph-pred', 'figure'),
        Output("loading-output-stock-pred", "children")
    ],
    [
        Input('stock_code_dropdown', 'value'),
        Input('prediction_method_dropdown', 'value'),
        Input('prediction_spec_dropdown', 'value')
    ],
)
def update_stock_graph(stock_code, method, spec):
    return updateStockFigure(stock_code, method, spec), ''


app.layout = html.Div([
    navbar,
    html.Div([
        html.Div(
            [
                dcc.Loading(
                    id="loading-stock-pred",
                    type="default",
                    children=html.Div(
                        id="loading-output-stock-pred"),
                ),
                dcc.Graph(id="graph-pred"),
            ],
        ),
        dcc.Interval(
            id='graph-pred-update',
            interval=1000 * 60,
            n_intervals=0
        )
    ])
])


def updateStockFigure(stock_code, method, spec):
    train, actual, predicted = pred.get_predicted_price(
        stock_code, method, spec
    )

    trainTrace = go.Scatter(
        name="Huấn luyện",
        x=train.index,
        y=train['Close'],
    )
    actualTrace = go.Scatter(
        name="Thưc tế",
        x=actual.index,
        y=actual['Close'],
        line=dict(color='#f59e42')
    )
    predictedTrace = go.Scatter(
        name="Dự đoán",
        x=predicted.index,
        y=predicted["Close"],
    )
    traces = [trainTrace, actualTrace, predictedTrace]

    return go.Figure(data=traces, layout=go.Layout(
        title=go.layout.Title(
            text=f"Sử dụng phương pháp *{method}* để dự đoán cho mã *{stock_code}*"
        )
    ))


app.run_server()
