import dash
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output, State
from sklearn.externals import joblib

model = joblib.load('modelv1.pkl')

app = dash.Dash()
server = app.server

app.layout = html.Div([
    html.H2(children="Hello :D! Let's see if you are saying something nice or mean..."),
    html.Div([
        html.H3(children="Enter your thoughts:"),
        dcc.Textarea(
            id='input-text-area',
            placeholder="I feel...",
            style={'width': '50%', 'height': '100%'}
        ),
        html.Br(),
        html.Div([            
            html.Button(id='submit-button', n_clicks=0, children='Press me and let me guess',
                        style={'font-size': '14px', 'marginLeft': 10}),
            html.A(
                children=html.Button(children='Source in GitHub', style={'font-size': '14px', 'marginLeft': 10}),
                href='https://github.com/diabetichunny/Sentiment-Analyser',
                target='_blank'
            )
        ], style={'display': 'inline-block'}) 
    ]),
    html.Br(),
    html.Div([
        html.H2(id='response', style={'color': 'Blue'})
    ])
], className='six.columns')


@app.callback(Output('response', 'children'), [Input('submit-button', 'n_clicks')], [State('input-text-area', 'value')])
def predict_sentiment(_, text):
    if text is not None:
        pred = model.predict([text])[0]

        if pred == 0:
            return "Your comment was negative or too neutral. :|"
        return "Your comment was positive. :) <3"
    return "Waiting for your comment. :D"


app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

if __name__ == '__main__':
    app.run_server()
