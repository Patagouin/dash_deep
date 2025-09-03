from dash import dcc, html
import plotly.graph_objs as go
import datetime  # Import datetime

# Create the figure directly
fig = go.FigureWidget()

def get_visualization_layout():
    return html.Div([
        html.H4('Visualisation des Prédictions', style={
            'marginBottom': '0px',  # Less gap for the second subtitle
            'padding': '20px',
            'color': '#FF8C00'
        }),
        
        # Performance Metrics Section
        html.Div([
            html.H5('Métriques de Performance', style={
                'color': '#4CAF50',
                'marginBottom': '10px'
            }),
            html.Div(id='performance-metrics', style={
                'backgroundColor': '#2E2E2E',
                'padding': '15px',
                'borderRadius': '8px',
                'marginBottom': '20px',
                'fontFamily': 'monospace'
            })
        ], style={
            'marginBottom': '20px',
            'padding': '0 20px'
        }),
        
        # Visualization controls
        html.Div([
            # Date picker
            html.Div([
                html.Label('Période'),
                dcc.DatePickerRange(
                    id='date_picker_range',
                    display_format='DD/MM/YY',
                    start_date=datetime.datetime.now()-datetime.timedelta(days=7),
                    end_date=datetime.datetime.now()
                ),
            ], style={'marginBottom': '20px'}),
            
            # Radio items and Checkbox on the same line
            html.Div([
                dcc.RadioItems(
                    id='data_type_selector',
                    options=[
                        {'label': 'All Data', 'value': 'all'},
                        {'label': 'Main Hours', 'value': 'main'}
                    ],
                    value='all',
                    inline=True,
                    labelStyle={'marginRight': '20px'},
                    style={'color': '#4CAF50'}
                ),
                dcc.Checklist(
                    id='normalize_checkbox',
                    options=[{'label': 'Normalize', 'value': 'normalize'}],
                    value=[],
                    inline=True,
                    className='custom-checkbox',
                    labelStyle={
                        'color': '#4CAF50',
                        'display': 'flex',
                        'alignItems': 'center',
                        'cursor': 'pointer',
                        'padding': '5px 10px',
                        'borderRadius': '4px',
                        'transition': 'background-color 0.3s',
                        'backgroundColor': 'rgba(76, 175, 80, 0.1)',
                        'border': '1px solid #4CAF50'
                    }
                )
            ], style={
                'display': 'flex',
                'justifyContent': 'center',
                'alignItems': 'center',
                'marginBottom': '20px',
                'backgroundColor': '#2E2E2E',
                'padding': '20px',
                'borderRadius': '8px',
                'gap': '50px'  # Space between elements
            }),

            # Graph container
            html.Div([
                dcc.Graph(
                    id='stock_graph',
                    figure=fig,
                    config={'scrollZoom': True},
                    style={
                        'backgroundColor': 'black',
                        'height': '50vh'
                    }
                )
            ], style={
                'width': '100%',
                'backgroundColor': 'black',
                'padding': '20px 0'
            }),
        ], style={
            'padding': '20px'
        }),
    ], style={
        'width': '100%',
        'backgroundColor': '#1E1E1E',
        'borderRadius': '8px',
        'marginBottom': '20px'
    }) 