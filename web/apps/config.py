# config.py
import sys
import io
import dash  # Ajout de l'import manquant
from dash import dcc, html
from dash.dependencies import Input, Output, State
from app import app, shM, socketio
import Models.Shares as sm
import threading
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Example log to test
logging.warning("Logging is configured and ready to use.")

layout = html.Div([
    html.H3('Configuration Broker'),

    # Broker configuration form
    html.Div([
        html.Label('Broker Type'),
        dcc.Dropdown(
            id='broker_type',
            options=[
                {'label': 'Broker A', 'value': 'broker_a'},
                {'label': 'Broker B', 'value': 'broker_b'}
            ],
            value='broker_a'  # Default value
        ),
        html.Br(),
        html.Label('Username'),
        dcc.Input(id='broker_username', type='text', value=''),
        html.Br(),
        html.Label('Password'),
        dcc.Input(id='broker_password', type='password', value=''),
        html.Br(),
        html.Label('Host'),
        dcc.Input(id='broker_host', type='text', value='localhost'),
        html.Br(),
        html.Label('Port'),
        dcc.Input(id='broker_port', type='number', value=5432),
        html.Br(),
        html.Label('Database'),
        dcc.Input(id='broker_database', type='text', value='stocksprices'),
        html.Br(),
        html.Button('Save Configuration', id='save_config', n_clicks=0),
        html.Div(id='config_status', style={'color': 'green'})
    ], style={'width': '50%', 'margin': 'auto'}),

    html.Hr(),

    # Update stock data section
    html.H3('Update Stock Data'),
    html.Div([  # Container pour les boutons et la checkbox
        html.Div([  # Container pour les boutons
            html.Button(
                'Start Update', 
                id='start_update', 
                n_clicks=0,
                style={
                    'backgroundColor': '#4CAF50',
                    'color': 'white',
                    'padding': '10px 20px',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'marginRight': '10px',
                    'fontSize': '16px',
                    'transition': 'background-color 0.3s'
                },
                className='update-button'
            ),
            html.Button(
                'Stop Update', 
                id='stop_update', 
                n_clicks=0,
                style={
                    'backgroundColor': '#f44336',
                    'color': 'white',
                    'padding': '10px 20px',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'fontSize': '16px',
                    'transition': 'background-color 0.3s'
                },
                className='stop-button'
            ),
        ], style={'marginBottom': '10px'}),
        dcc.Checklist(
            id='check_duplicate',
            options=[{'label': 'Check duplicate', 'value': 'check_duplicate'}],
            value=[],
            style={'marginTop': '10px'}
        ),
    ], style={'width': '50%', 'margin': 'auto', 'textAlign': 'center'}),
    
    # Ajout d'un store pour le statut d'arrêt
    dcc.Store(id='stop_status', data=False),
    
    dcc.Interval(id='progress_interval', interval=1000, disabled=True),
    dcc.Store(id='update_state', data='idle'),

    # Progress bar personnalisée
    html.Div([
        html.Div(id='progress_bar', 
                children='0%',
                style={
                    'width': '0%',
                    'height': '30px',
                    'backgroundColor': '#4CAF50',
                    'textAlign': 'center',
                    'lineHeight': '30px',
                    'color': 'white',
                    'transition': 'width 0.5s ease-in-out'
                }),
    ], style={
        'width': '50%',
        'margin': '20px auto',
        'backgroundColor': '#ddd',
        'border': '1px solid #ccc',
        'borderRadius': '5px',
        'overflow': 'hidden'
    }),

    # Terminal output textarea avec style amélioré
    dcc.Textarea(
        id='terminal_output',
        value='',
        style={
            'width': '50%',  # Réduire la largeur
            'height': '300px',
            'margin': '20px auto',  # Centrer
            'display': 'block',  # Nécessaire pour que margin: auto fonctionne
            'backgroundColor': 'black',
            'color': '#00FF00',  # Vert "terminal"
            'fontFamily': 'monospace',
            'padding': '10px',
            'overflowY': 'scroll'
        }
    ),
])

# Ajouter cette variable globale au début du fichier, après les imports
stop_update_flag = False

# Callback to start the background update process
@app.callback(
    Output('progress_interval', 'disabled'),
    Output('update_state', 'data'),
    Output('start_update', 'style'),
    Output('stop_update', 'style'),
    Output('stop_status', 'data'),
    [Input('start_update', 'n_clicks'),
     Input('stop_update', 'n_clicks')],
    [State('check_duplicate', 'value'),
     State('update_state', 'data'),
     State('start_update', 'style'),
     State('stop_update', 'style')]
)
def handle_update_buttons(start_clicks, stop_clicks, check_duplicate, update_state, start_style, stop_style):
    global stop_update_flag
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return True, 'idle', start_style, stop_style, False
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'start_update' and start_clicks and update_state == 'idle':
        # Démarrer la mise à jour
        stop_update_flag = False  # Réinitialiser le flag
        socketio.start_background_task(update_shares_in_background, 'check_duplicate' in check_duplicate)
        
        # Désactiver le bouton Start et activer le bouton Stop
        start_style.update({'opacity': '0.6', 'cursor': 'not-allowed'})
        stop_style.update({'opacity': '1', 'cursor': 'pointer'})
        
        return False, 'running', start_style, stop_style, False
    
    elif button_id == 'stop_update' and stop_clicks and update_state == 'running':
        # Activer le flag d'arrêt
        stop_update_flag = True
        
        # Activer le bouton Start et désactiver le bouton Stop
        start_style.update({'opacity': '1', 'cursor': 'pointer'})
        stop_style.update({'opacity': '0.6', 'cursor': 'not-allowed'})
        
        return True, 'idle', start_style, stop_style, True
    
    return True, update_state, start_style, stop_style, False

@app.callback(
    Output('config_status', 'children'),
    Input('save_config', 'n_clicks'),
    State('broker_type', 'value'),
    State('broker_username', 'value'),
    State('broker_password', 'value'),
    State('broker_host', 'value'),
    State('broker_port', 'value'),
    State('broker_database', 'value')
)
def save_configuration(n_clicks, broker_type, username, password, host, port, database):
    if n_clicks > 0:
        logging.info(f"Configuration saved: {broker_type}, {username}, {password}, {host}, {port}, {database}")
        return "Configuration saved successfully!"
    return ""

def update_shares_in_background(check_duplicate):
    global stop_update_flag
    stop_update_flag = False  # Réinitialiser le flag au début
    
    try:
        socketio.emit('update_terminal', {'output': 'Starting background task...\n'}, namespace='/')
        
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        shM = sm.Shares(readOnlyThosetoUpdate=True)
        total_shares = len(shM.dfShares)
        updated_shares = 0

        for share in shM.dfShares.itertuples():
            try:
                # Vérifier si l'arrêt a été demandé
                if stop_update_flag:  # Utiliser la variable globale au lieu de app.get_asset_url
                    socketio.emit('update_terminal', {'output': 'Update process stopped by user\n'}, namespace='/')
                    break

                shM.updateShareCotations(share, checkDuplicate=check_duplicate)
                updated_shares += 1

                progress = float((updated_shares / total_shares) * 100)

                captured_output = sys.stdout.getvalue()
                if captured_output:
                    socketio.emit('update_terminal', {'output': captured_output}, namespace='/')
                sys.stdout.truncate(0)
                sys.stdout.seek(0)

                socketio.emit('update_progress', {'progress': progress}, namespace='/')

            except Exception as share_error:
                error_msg = f"Error processing share {share.symbol}: {str(share_error)}\n"
                socketio.emit('update_terminal', {'output': error_msg}, namespace='/')

        socketio.emit('update_terminal', {'output': 'Update process finished\n'}, namespace='/')

    except Exception as e:
        error_message = f"Error in background task: {str(e)}\n"
        socketio.emit('update_terminal', {'output': error_message}, namespace='/')

    finally:
        sys.stdout = old_stdout

# Modifier le callback de mise à jour de la barre de progression
@app.callback(
    Output('progress_bar', 'style'),
    Output('progress_bar', 'children'),
    [Input('progress_interval', 'n_intervals')],
    [State('progress_bar', 'style')]
)
def update_progress_bar(n, current_style):
    if not current_style:
        current_style = {}
    # La largeur sera mise à jour par Socket.IO
    return current_style, current_style.get('width', '0%')
