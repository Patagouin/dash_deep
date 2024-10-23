import sys
import io
from dash import dcc, html
from dash.dependencies import Input, Output, State
from app import app, shM, socketio
import Models.Shares as sm
import threading
import logging

# Configure logging to display INFO level messages in the console
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format of the log messages
    datefmt='%Y-%m-%d %H:%M:%S'  # Format of the date in the log messages
)

# Example log to test
logging.warning("Logging is configured and ready to use.")

layout = html.Div([
    html.H3('Configuration Broker'),
    
    # Form for Broker type and credentials
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
    
    # Section for updating stock data
    html.H3('Update Stock Data'),
    html.Button('Start Update', id='start_update', n_clicks=0),
    dcc.Checklist(
        id='check_duplicate',
        options=[{'label': 'Check duplicate', 'value': 'check_duplicate'}],
        value=[],
    ),
    dcc.Interval(id='progress_interval', interval=1000, disabled=True),  # Interval initially disabled
    dcc.Store(id='progress_data', data={'progress': 0}),
    dcc.Store(id='update_state', data='idle'),  # Store to track update state
    
    # Use dcc.Loading to show a loading spinner during the update process
    dcc.Loading(
        id="loading_progress",
        type="default",
        children=[
            html.Div(id='progress_bar', style={'margin-top': '20px'})
        ]
    ),
    
    # Text area to display terminal output
    dcc.Textarea(
        id='terminal_output',
        value='',
        style={'width': '100%', 'height': '300px', 'overflowY': 'scroll'}
    ),

    # Add a script to listen for SocketIO events
    html.Script('''
        const socket = io.connect('http://' + document.domain + ':' + location.port);
        console.log("SocketIO connected:", socket.connected);  // Log the connection status

        // Listen for 'update_terminal' event
        socket.on('update_terminal', function(message) {
            console.log("Received terminal update:", message);
            let textarea = document.getElementById('terminal_output');
            textarea.value += message.output + "\\n";  // Append the new message to the textarea
        });

        // Listen for 'update_progress' event
        socket.on('update_progress', function(data) {
            console.log("Received progress update:", data);
            let progressBar = document.getElementById('progress_bar');
            progressBar.innerHTML = `Progress: ${data.progress}%`;  // Update the progress bar
        });
    ''')
])

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
        print(f"Configuration saved: {broker_type}, {username}, {password}, {host}, {port}, {database}")
        return "Configuration saved successfully!"
    return ""

def update_shares_in_background(check_duplicate, progress_data, terminal_output):
    try:
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        shM = sm.Shares(readOnlyThosetoUpdate=True)
        total_shares = len(shM.dfShares)
        updated_shares = 0

        for share in shM.dfShares.itertuples():               
            shM.updateShareCotations(share, checkDuplicate=check_duplicate)
            updated_shares += 1
            # Convertir en float pour éviter les problèmes de sérialisation
            progress = float((updated_shares / total_shares) * 100)
            progress_data['progress'] = progress
            
            # Capture the output from updateShareCotations

            captured_output = sys.stdout.getvalue()
            sys.stdout.truncate(0)
            sys.stdout.seek(0)

            # Emit the captured output to the client
            if captured_output:
                terminal_output += captured_output
                logging.warning(f"Emitting terminal output: {captured_output}")
                # S'assurer que captured_output est une chaîne de caractères
                socketio.emit('update_terminal', {'output': str(captured_output)})

            # Emit progress
            logging.warning(f"Emitting progress: {progress}")
            socketio.emit('update_progress', {'progress': progress})
        
        terminal_output += "Update complete!\n"
        logging.warning(f"Emitting terminal output: {terminal_output}")
        socketio.emit('update_terminal', {'output': str(terminal_output)})

    except Exception as e:
        terminal_output += f"Error: {str(e)}\n"
        logging.warning(f"Error occurred: {str(e)}")
        socketio.emit('update_terminal', {'output': str(terminal_output)})

    finally:
        # Restore stdout
        sys.stdout = old_stdout

@app.callback(
    Output('progress_bar', 'children'),
    Output('terminal_output', 'value'),
    Output('progress_data', 'data'),
    Output('start_update', 'n_clicks'),
    Output('progress_interval', 'disabled'),
    Output('update_state', 'data'),
    Input('start_update', 'n_clicks'),
    Input('progress_interval', 'n_intervals'),
    State('check_duplicate', 'value'),
    State('progress_data', 'data'),
    State('terminal_output', 'value'),
    State('update_state', 'data')
)
def update_stock_data(n_clicks, n_intervals, check_duplicate, progress_data, terminal_output, update_state):
    if n_clicks > 0 and update_state == 'idle':
        # Initialiser progress_data avec des types simples
        if progress_data is None:
            progress_data = {'progress': 0.0}
        thread = threading.Thread(
            target=update_shares_in_background, 
            args=('check_duplicate' in check_duplicate, progress_data, terminal_output or '')
        )
        thread.start()
        return "Updating...", terminal_output or '', progress_data, 0, False, 'running'
    
    if update_state == 'running':
        progress = progress_data.get('progress', 0.0) if progress_data else 0.0
        return f"Progress: {progress}%", terminal_output or '', progress_data, n_clicks, False, 'running'
    
    return "", terminal_output or '', progress_data or {'progress': 0.0}, n_clicks, True, 'idle'
