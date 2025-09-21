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
                html.A('Analyse', href='/analyse', id='nav-analyse', className='nav-link', **{'data-page': 'analyse'}),
                html.Span(' | ', className='nav-separator'),
                html.A('Prediction', href='/prediction', id='nav-prediction', className='nav-link', **{'data-page': 'prediction'}),
                html.Span(' | ', className='nav-separator'),
                html.A('Update', href='/update', id='nav-update', className='nav-link', **{'data-page': 'update'}),
                html.Span(' | ', className='nav-separator'),
                html.A('Config', href='/config', id='nav-config', className='nav-link', **{'data-page': 'config'}),
                html.Span(' | ', className='nav-separator'),
                html.A('Transaction', href='/transaction', id='nav-transaction', className='nav-link', **{'data-page': 'transaction'})
            ], className='nav-container')
        ])
    ], className='navigation-bar')

# Callback unique pour mettre à jour les classes CSS
@app.callback(
    [Output('nav-dashboard', 'className'),
     Output('nav-analyse', 'className'),
     Output('nav-prediction', 'className'),
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
    classes = [base_class] * 6
    
    # Mettre à jour la classe du lien actif
    if pathname:
        pathname = pathname.lstrip('/')
        if pathname == '' or pathname == 'dashboard':
            classes[0] = active_class
        elif pathname == 'analyse':
            classes[1] = active_class
        elif pathname == 'prediction':
            classes[2] = active_class
        elif pathname == 'update':
            classes[3] = active_class
        elif pathname == 'config':
            classes[4] = active_class
        elif pathname == 'transaction':
            classes[5] = active_class
    
    return classes
