#update.py
import sys
import io
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from web.components.navigation import create_navigation  # Importer le composant de navigation
from web.app import app, shM, socketio
import Models.Shares as sm
import Models.utils as ut
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Variable globale pour le flag d'arrêt
stop_update_flag = False

layout = html.Div([
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
            'width': '50%',
            'height': '300px',
            'margin': '20px auto',
            'display': 'block',
            'backgroundColor': 'black',
            'color': '#00FF00',
            'fontFamily': 'monospace',
            'padding': '10px',
            'overflowY': 'scroll'
        }
    ),

    # Navigation standardisée avec séparateur
    create_navigation()
], style={'backgroundColor': 'black', 'minHeight': '100vh'})

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
        logging.info("No button clicked")
        return True, 'idle', start_style, stop_style, False
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    logging.info(f"Button clicked: {button_id}")
    
    if button_id == 'start_update' and start_clicks and update_state == 'idle':
        stop_update_flag = False
        logging.info("Starting background task...")
        socketio.start_background_task(update_shares_in_background, 'check_duplicate' in check_duplicate)
        
        start_style.update({'opacity': '0.6', 'cursor': 'not-allowed'})
        stop_style.update({'opacity': '1', 'cursor': 'pointer'})
        
        return False, 'running', start_style, stop_style, False
    
    elif button_id == 'stop_update' and stop_clicks and update_state == 'running':
        stop_update_flag = True
        logging.info("Stopping update process...")
        
        start_style.update({'opacity': '1', 'cursor': 'pointer'})
        stop_style.update({'opacity': '0.6', 'cursor': 'not-allowed'})
        
        return True, 'idle', start_style, stop_style, True
    
    return True, update_state, start_style, stop_style, False

def update_shares_in_background(check_duplicate):
    global stop_update_flag
    stop_update_flag = False
    
    try:
        socketio.emit('update_terminal', {'output': 'Starting background task...\n'}, namespace='/')
        
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        shM = sm.Shares(readOnlyThosetoUpdate=True)
        total_shares = len(shM.dfShares)
        updated_shares = 0

        for share in shM.dfShares.itertuples():
            try:
                if stop_update_flag:
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

@app.callback(
    Output('progress_bar', 'style'),
    Output('progress_bar', 'children'),
    [Input('progress_interval', 'n_intervals')],
    [State('progress_bar', 'style')]
)
def update_progress_bar(n, current_style):
    if not current_style:
        current_style = {}
    return current_style, current_style.get('width', '0%')
