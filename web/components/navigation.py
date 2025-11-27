from dash import html, dcc, Input, Output
from app import app

def create_navigation():
    return html.Div([
        html.Hr(style={
            'width': '50%',
            'margin': '10px auto',
            'borderTop': '1px solid #666'
        }),
        html.Div([
            html.Div([
                html.A('Dashboard', href='/dashboard', id='nav-dashboard', className='nav-link', **{'data-page': 'dashboard'}),
                html.Span(' | ', className='nav-separator'),
                html.A('Visualisation', href='/visualisation', id='nav-visualisation', className='nav-link', **{'data-page': 'visualisation'}),
                html.Span(' | ', className='nav-separator'),
                html.A('Analyse', href='/analyse', id='nav-analyse', className='nav-link', **{'data-page': 'analyse'}),
                html.Span(' | ', className='nav-separator'),
                html.A('Prediction', href='/prediction', id='nav-prediction', className='nav-link', **{'data-page': 'prediction'}),
                html.Span(' | ', className='nav-separator'),
                html.A('Simulation', href='/simulation', id='nav-simulation', className='nav-link', **{'data-page': 'simulation'}),
                html.Span(' | ', className='nav-separator'),
                html.A('Playground', href='/playground', id='nav-playground', className='nav-link', **{'data-page': 'playground'}),
                html.Span(' | ', className='nav-separator'),
                html.A('Transaction', href='/transaction', id='nav-transaction', className='nav-link', **{'data-page': 'transaction'}),
                html.Span(' | ', className='nav-separator'),
                html.A('Update', href='/update', id='nav-update', className='nav-link', **{'data-page': 'update'}),
                html.Span(' | ', className='nav-separator'),
                html.A('Config', href='/config', id='nav-config', className='nav-link', **{'data-page': 'config'})
            ], className='nav-container')
        ])
    ], className='navigation-bar')

def create_page_help(title, content_markdown):
    """
    Crée un composant d'aide affiché en haut à droite de la page en utilisant le CSS Tooltip avec Toggle (Click).
    
    :param title: Titre de l'aide
    :param content_markdown: Contenu explicatif au format Markdown
    :return: Composant Dash html.Div
    """
    # Création d'un ID unique et sûr pour lier le label et la checkbox
    safe_id = "help-toggle-" + "".join([c if c.isalnum() else "_" for c in title])
    
    return html.Div([
        # Conteneur tooltip CSS
        html.Div([
            # Checkbox invisible qui contrôle l'état affiché/caché
            dcc.Input(type='checkbox', id=safe_id, className='help-toggle-checkbox'),
            
            # Label qui agit comme le bouton déclencheur (l'icône ?)
            html.Label(
                "?", 
                htmlFor=safe_id,
                className='help-tooltip-icon',
                style={
                    'width': '30px',
                    'height': '30px',
                    'borderRadius': '50%',
                    'backgroundColor': '#4CAF50',
                    'color': 'white',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'fontWeight': 'bold',
                    'fontSize': '18px',
                    'boxShadow': '0 2px 5px rgba(0,0,0,0.3)',
                    'cursor': 'pointer', # Indique qu'on peut cliquer
                    'userSelect': 'none' # Évite de sélectionner le texte du ?
                }
            ),
            # Contenu du tooltip (caché par défaut en CSS, affiché quand checkbox checked)
            html.Div([
                html.Span(title, className='help-tooltip-title'),
                dcc.Markdown(content_markdown)
            ], className='help-tooltip-text')
        ], className='help-tooltip-container')
    ], style={
        'position': 'fixed', # Fixed pour rester en place même au scroll
        'top': '80px',       # En dessous de la barre de titre/nav potentielle
        'right': '20px',
        'zIndex': '9998'
    })

# Callback unique pour mettre à jour les classes CSS
@app.callback(
    [Output('nav-dashboard', 'className'),
     Output('nav-visualisation', 'className'),
     Output('nav-analyse', 'className'),
     Output('nav-prediction', 'className'),
     Output('nav-simulation', 'className'),
     Output('nav-playground', 'className'),
     Output('nav-update', 'className'),
     Output('nav-config', 'className'),
     Output('nav-transaction', 'className')],
    [Input('url', 'pathname')]
)
def update_nav_style(pathname):
    # Classes de base et active
    base_class = 'nav-link'
    active_class = 'nav-link active'
    
    # Initialiser toutes les classes comme inactives
    classes = [base_class] * 9
    
    # Mettre à jour la classe du lien actif
    if pathname:
        pathname = pathname.lstrip('/')
        if pathname == '' or pathname == 'dashboard':
            classes[0] = active_class
        elif pathname == 'visualisation':
            classes[1] = active_class
        elif pathname == 'analyse':
            classes[2] = active_class
        elif pathname == 'prediction':
            classes[3] = active_class
        elif pathname == 'simulation':
            classes[4] = active_class
        elif pathname == 'playground':
            classes[5] = active_class
        elif pathname == 'update':
            classes[6] = active_class
        elif pathname == 'config':
            classes[7] = active_class
        elif pathname == 'transaction':
            classes[8] = active_class
    
    return classes
