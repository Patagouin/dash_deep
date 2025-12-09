from dash import html, dcc, Input, Output
from app import app

# Définition des icônes (emoji) pour chaque page
NAV_ICONS = {
    'dashboard': '📊',
    'visualisation': '📈',
    'analyse': '📉',
    'prediction': '🤖',
    'simulation': '🧪',
    'playground': '🧩',
    'transaction': '💸',
    'update': '🔄',
    'config': '⚙️'
}


def create_nav_item(page_id, label, href, icon_key):
    """Crée un élément de navigation avec une icône."""
    icon = NAV_ICONS.get(icon_key)

    children = []
    if icon:
        children.append(
            html.Span(
                icon,
                className='nav-icon',
                style={'marginRight': '6px'}
            )
        )

    children.append(html.Span(label, className='nav-label'))

    return html.A(
        children,
        href=href,
        id=f'nav-{page_id}',
        className='nav-link',
        **{'data-page': page_id}
    )

def create_navigation():
    """Crée la barre de navigation principale avec un design moderne."""
    nav_items = [
        ('dashboard', 'Dashboard', '/dashboard', 'dashboard'),
        ('visualisation', 'Visualisation', '/visualisation', 'visualisation'),
        ('analyse', 'Analyse', '/analyse', 'analyse'),
        ('prediction', 'Prédiction', '/prediction', 'prediction'),
        ('simulation', 'Simulation', '/simulation', 'simulation'),
        ('playground', 'Playground', '/playground', 'playground'),
        ('transaction', 'Transaction', '/transaction', 'transaction'),
        ('update', 'Update', '/update', 'update'),
        ('config', 'Config', '/config', 'config'),
    ]
    
    nav_elements = []
    for i, (page_id, label, href, icon) in enumerate(nav_items):
        nav_elements.append(create_nav_item(page_id, label, href, icon))
        
    return html.Div([
        html.Div([
            html.Div(nav_elements, className='nav-container')
        ], className='nav-inner')
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
        html.Div([
            # Checkbox invisible qui contrôle l'état affiché/caché
            dcc.Input(type='checkbox', id=safe_id, className='help-toggle-checkbox'),
            
            # Label qui agit comme le bouton déclencheur (l'icône ?)
            html.Label(
                "?", 
                htmlFor=safe_id,
                className='help-tooltip-icon',
                style={
                    'width': '40px',
                    'height': '40px',
                    'borderRadius': '50%',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'fontWeight': '600',
                    'fontSize': '20px',
                    'cursor': 'pointer',
                    'userSelect': 'none'
                }
            ),
            # Contenu du tooltip (caché par défaut en CSS, affiché quand checkbox checked)
            html.Div([
                html.Span(title, className='help-tooltip-title'),
                dcc.Markdown(content_markdown)
            ], className='help-tooltip-text')
        ], className='help-tooltip-container')
    ], style={
        'position': 'fixed',
        'top': '20px',
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
