import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from appMain import app, shM, socketio
import Models.utils as ut

layout = html.Div([
    html.H3('Update BDD'),
    html.Div(dcc.Input(id='input-on-submit', type='text')),
    html.Button('Update', id='submit-val', n_clicks=0),
    dcc.Checklist(
        id='check_duplicate',
        options=[{'label': 'Check duplicate', 'value': 'check_duplicate'}],
        value=[],
    ),
    dcc.Textarea(
        id='textarea',
        value='Outputs',
        style={'width': '100%', 'height': '40%'},
    ),
    dcc.Interval(id='updater', interval=500),
    dcc.Store(id='progress-info', data=None),

    html.Div(id='container-button-basic'),
    html.Br(),
    dcc.Link('Go to dashboard', href='/dashboard'),
    html.Br(),
    dcc.Link('Go to prediction', href='/prediction'),

    # Ajoutez le code JavaScript ici
    html.Script(src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.4.1/socket.io.min.js"),
    html.Script('''
        const socket = io.connect('http://' + document.domain + ':' + location.port);
        socket.on('update_textarea', function(message) {
            let textarea = document.getElementById('textarea');
            textarea.value = message;
        });
    ''')
])

@app.callback(
    Output('submit-val', 'n_clicks'),
    Input('submit-val', 'n_clicks'),
    Input('check_duplicate', 'value'),
    State('session-id', 'data')
)
def update_output(n, check_duplicate, session_data):
    if session_data is None:
        session_data = {'last_index': 0}

    last_index = session_data['last_index']
    nbTotalShares = shM.dfShares.shape[0]
    nbSharesUpdated = last_index
    print("avant boucle")

    for curShare in shM.dfShares.iloc[last_index:].itertuples():
        print(curShare)
        shM.updateShareCotations(curShare, checkDuplicate=check_duplicate)
        nbSharesUpdated += 1
        message = f"Nb shares cotation updated: {nbSharesUpdated}/{nbTotalShares}"
        socketio.emit('update_textarea', message)

    message = "Success: All shares cotation updated"
    socketio.emit('update_textarea', message)

    return n
