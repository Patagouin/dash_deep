#update.py
import sys
import io
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from web.components.navigation import create_navigation, create_page_help
from app import app, shM, socketio
import Models.Shares as sm
import Models.utils as ut
import logging
import os
import subprocess
from web.apps.config import load_config
from Models.SqlCom import SqlCom

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Variable globale pour le flag d'arrêt
stop_update_flag = False

# Styles communs
CARD_STYLE = {
    'backgroundColor': '#1a1a24',
    'padding': '24px',
    'borderRadius': '16px',
    'border': '1px solid rgba(148, 163, 184, 0.1)',
    'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.4)',
    'marginBottom': '20px'
}

BUTTON_PRIMARY = {
    'background': 'linear-gradient(135deg, #10b981 0%, #34d399 100%)',
    'color': 'white',
    'padding': '12px 24px',
    'border': 'none',
    'borderRadius': '10px',
    'cursor': 'pointer',
    'fontSize': '0.9375rem',
    'fontWeight': '600',
    'fontFamily': 'Outfit, sans-serif',
    'transition': 'all 0.25s ease',
    'boxShadow': '0 4px 6px -1px rgba(16, 185, 129, 0.3)'
}

BUTTON_DANGER = {
    'background': 'linear-gradient(135deg, #ef4444 0%, #f87171 100%)',
    'color': 'white',
    'padding': '12px 24px',
    'border': 'none',
    'borderRadius': '10px',
    'cursor': 'pointer',
    'fontSize': '0.9375rem',
    'fontWeight': '600',
    'fontFamily': 'Outfit, sans-serif',
    'transition': 'all 0.25s ease',
    'boxShadow': '0 4px 6px -1px rgba(239, 68, 68, 0.3)'
}

BUTTON_INFO = {
    'background': 'linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%)',
    'color': 'white',
    'padding': '12px 24px',
    'border': 'none',
    'borderRadius': '10px',
    'cursor': 'pointer',
    'fontSize': '0.9375rem',
    'fontWeight': '600',
    'fontFamily': 'Outfit, sans-serif',
    'transition': 'all 0.25s ease',
    'boxShadow': '0 4px 6px -1px rgba(59, 130, 246, 0.3)'
}

def get_db_backup_dir() -> str:
    """
    Retourne le dossier d'export de la BDD (CSV) configuré dans l'UI (page Config).

    Remarque:
    - On ne supporte plus d'autres chemins implicites (env var / data/DB / cwd).
    - Si `export_path` est absent, on force l'utilisateur à le configurer.
    """
    config_data = load_config()
    export_path = config_data.get("export_path", None)
    if not export_path:
        raise ValueError("export_path non configuré. Configure-le dans la page Config.")

    os.makedirs(export_path, exist_ok=True)
    if not os.path.isdir(export_path):
        raise FileNotFoundError(f"Dossier d'export introuvable: {export_path}")

    return export_path

def open_folder_in_os(path: str) -> None:
    if not path:
        raise ValueError("Chemin vide")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Dossier introuvable: {path}")

    if sys.platform.startswith("linux"):
        subprocess.Popen(["xdg-open", path])
        return
    if sys.platform.startswith("darwin"):
        subprocess.Popen(["open", path])
        return
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
        return

    raise RuntimeError(f"OS non supporté: {sys.platform}")

help_text = """
### Mise à jour des données

Cette page permet de télécharger et mettre à jour les cotations boursières depuis les sources de données.

#### Fonctionnalités
*   **Start Update** : Lance la mise à jour de toutes les actions configurées.
*   **Stop Update** : Arrête la mise à jour en cours.
*   **Export Database** : Exporte les données vers des fichiers CSV.

#### Options
*   **Check duplicate** : Vérifie les doublons avant insertion (plus lent mais plus sûr).
*   **Export after update** : Exporte automatiquement après la mise à jour.

#### Terminal
Le terminal affiche la progression en temps réel de la mise à jour.
"""

layout = html.Div([
    create_page_help("Aide Update", help_text),
    
    # En-tête de page
    html.Div([
        html.H3('Update Stock Data', style={
            'margin': '0',
            'textAlign': 'center'
        }),
        html.P('Mise à jour des cotations boursières', style={
            'textAlign': 'center',
            'color': '#94a3b8',
            'marginTop': '8px',
            'marginBottom': '0'
        })
    ], style={'marginBottom': '32px'}),
    
    # Carte des contrôles
    html.Div([
        # Titre de section
        html.Div([
            html.Span('⚡', style={'fontSize': '1.25rem'}),
            html.Span('Contrôles', style={
                'fontSize': '1rem',
                'fontWeight': '600',
                'color': '#a78bfa',
                'marginLeft': '8px'
            })
        ], style={'marginBottom': '20px', 'display': 'flex', 'alignItems': 'center'}),
        
        # Boutons d'action
        html.Div([
            html.Button(
                '▶ Start Update', 
                id='start_update', 
                n_clicks=0,
                style=BUTTON_PRIMARY,
                className='update-button'
            ),
            html.Button(
                '⏹ Stop Update', 
                id='stop_update', 
                n_clicks=0,
                style=BUTTON_DANGER,
                className='stop-button'
            ),
            html.Button(
                '📦 Export Database',
                id='export_db_now',
                n_clicks=0,
                style=BUTTON_INFO,
                className='export-button'
            ),
            html.Button(
                '📂 Ouvrir dossier sauvegarde BDD',
                id='open_backup_folder',
                n_clicks=0,
                style=BUTTON_INFO,
                className='open-folder-button'
            ),
        ], style={
            'display': 'flex',
            'gap': '12px',
            'flexWrap': 'wrap',
            'marginBottom': '20px'
        }),

        html.Div(
            id='open_backup_folder_status',
            style={
                'marginTop': '8px',
                'color': '#94a3b8',
                'fontSize': '0.875rem'
            }
        ),
        
        # Options (Checkboxes)
        html.Div([
            dcc.Checklist(
                id='check_duplicate',
                options=[{'label': ' Vérifier les doublons', 'value': 'check_duplicate'}],
                value=[],
                style={'marginBottom': '8px'},
                persistence=True, 
                persistence_type='session',
                labelStyle={'color': '#94a3b8', 'cursor': 'pointer'}
            ),
            dcc.Checklist(
                id='export_data',
                options=[{'label': ' Exporter après mise à jour', 'value': 'export_data'}],
                value=[],
                persistence=True, 
                persistence_type='session',
                labelStyle={'color': '#94a3b8', 'cursor': 'pointer'}
            ),
        ], style={
            'backgroundColor': '#12121a',
            'padding': '16px',
            'borderRadius': '10px',
            'border': '1px solid rgba(148, 163, 184, 0.1)'
        })
    ], style={
        **CARD_STYLE,
        'maxWidth': '600px',
        'margin': '0 auto 24px'
    }),
    
    # Stores et Interval
    dcc.Store(id='stop_status', data=False),
    dcc.Interval(id='progress_interval', interval=1000, disabled=True),
    dcc.Store(id='update_state', data='idle'),
    
    # Carte de progression
    html.Div([
        html.Div([
            html.Span('📊', style={'fontSize': '1.25rem'}),
            html.Span('Progression', style={
                'fontSize': '1rem',
                'fontWeight': '600',
                'color': '#a78bfa',
                'marginLeft': '8px'
            })
        ], style={'marginBottom': '16px', 'display': 'flex', 'alignItems': 'center'}),
        
        # Barre de progression
        html.Div([
            html.Div(
                id='progress_bar', 
                children='0%',
                style={
                    'width': '0%',
                    'height': '40px',
                    'background': 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%)',
                    'textAlign': 'center',
                    'lineHeight': '40px',
                    'color': 'white',
                    'fontWeight': '600',
                    'fontFamily': 'JetBrains Mono, monospace',
                    'fontSize': '0.875rem',
                    'transition': 'width 0.5s ease-in-out',
                    'borderRadius': '10px'
                }
            ),
        ], style={
            'backgroundColor': '#12121a',
            'borderRadius': '10px',
            'overflow': 'hidden',
            'border': '1px solid rgba(148, 163, 184, 0.1)'
        }),
    ], style={
        **CARD_STYLE,
        'maxWidth': '600px',
        'margin': '0 auto 24px'
    }),

    # Terminal output
    html.Div([
        html.Div([
            html.Span('💻', style={'fontSize': '1.25rem'}),
            html.Span('Terminal', style={
                'fontSize': '1rem',
                'fontWeight': '600',
                'color': '#a78bfa',
                'marginLeft': '8px'
            })
        ], style={'marginBottom': '16px', 'display': 'flex', 'alignItems': 'center'}),
        
        dcc.Textarea(
            id='terminal_output',
            value='En attente de commande...',
            style={
                'width': '100%',
                'height': '350px',
                'backgroundColor': '#0a0a0f',
                'color': '#10b981',
                'fontFamily': 'JetBrains Mono, monospace',
                'fontSize': '0.8125rem',
                'padding': '16px',
                'borderRadius': '10px',
                'border': '1px solid rgba(148, 163, 184, 0.1)',
                'resize': 'vertical',
                'lineHeight': '1.6'
            },
            persistence=True, 
            persistence_type='session'
        ),
    ], style={
        **CARD_STYLE,
        'maxWidth': '800px',
        'margin': '0 auto 24px'
    }),

    # Statut d'export
    html.Div(
        id='export_now_status', 
        style={
            'textAlign': 'center',
            'padding': '12px',
            'color': '#10b981',
            'fontWeight': '500'
        }
    ),

    # Spacer pour navigation
    html.Div(style={'height': '100px'}),

    # Navigation
    create_navigation()
], style={
    'backgroundColor': '#0a0a0f',
    'minHeight': '100vh',
    'padding': '24px'
})

@app.callback(
    Output('progress_interval', 'disabled'),
    Output('update_state', 'data'),
    Output('start_update', 'style'),
    Output('stop_update', 'style'),
    Output('stop_status', 'data'),
    [Input('start_update', 'n_clicks'),
     Input('stop_update', 'n_clicks')],
    [State('check_duplicate', 'value'),
     State('export_data', 'value'),
     State('update_state', 'data'),
     State('start_update', 'style'),
     State('stop_update', 'style')]
)
def handle_update_buttons(start_clicks, stop_clicks, check_duplicate, export_data, update_state, start_style, stop_style):
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
        socketio.start_background_task(update_shares_in_background, 'check_duplicate' in check_duplicate, 'export_data' in export_data)
        
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

def update_shares_in_background(check_duplicate, export_data):
    global stop_update_flag
    stop_update_flag = False
    
    try:
        socketio.emit('update_terminal', {'output': '🚀 Starting background task...\n'}, namespace='/')
        
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        shM = sm.Shares(readOnlyThosetoUpdate=True)
        total_shares = len(shM.dfShares)
        updated_shares = 0

        for share in shM.dfShares.itertuples():
            try:
                if stop_update_flag:
                    socketio.emit('update_terminal', {'output': '⏹ Update process stopped by user\n'}, namespace='/')
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
                error_msg = f"❌ Error processing share {share.symbol}: {str(share_error)}\n"
                socketio.emit('update_terminal', {'output': error_msg}, namespace='/')

        # Exporter les données si la case est cochée
        if export_data:
            export_path = get_db_backup_dir()
            from web.apps.config import export_database
            export_message = export_database(1, export_path)
            socketio.emit('update_terminal', {'output': f'📦 {export_message}\n'}, namespace='/')

        socketio.emit('update_terminal', {'output': '✅ Update process finished\n'}, namespace='/')
        stop_update_flag = True
    except Exception as e:
        error_message = f"❌ Error in background task: {str(e)}\n"
        socketio.emit('update_terminal', {'output': error_message}, namespace='/')

    finally:
        sys.stdout = old_stdout

@app.callback(
    Output('export_now_status', 'children'),
    Input('export_db_now', 'n_clicks')
)
def export_db_now(n_clicks):
    if not n_clicks:
        return ''
    try:
        export_path = get_db_backup_dir()

        sql_com = SqlCom(
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            sharesObj=None
        )

        export_message = sql_com.export_data_to_csv(export_path)
        socketio.emit('update_terminal', {'output': f'📦 {export_message}\n'}, namespace='/')
        return f'✅ {export_message}'
    except Exception as e:
        error_message = f"❌ Error during export: {str(e)}"
        socketio.emit('update_terminal', {'output': error_message + '\n'}, namespace='/')
        return error_message

@app.callback(
    Output('open_backup_folder_status', 'children'),
    Input('open_backup_folder', 'n_clicks')
)
def open_backup_folder(n_clicks):
    if not n_clicks:
        return ''
    try:
        backup_dir = get_db_backup_dir()
        open_folder_in_os(backup_dir)
        msg = f"📂 Dossier ouvert: {backup_dir}"
        socketio.emit('update_terminal', {'output': msg + '\n'}, namespace='/')
        return msg
    except Exception as e:
        msg = f"❌ Impossible d'ouvrir le dossier: {e}"
        socketio.emit('update_terminal', {'output': msg + '\n'}, namespace='/')
        return msg

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
