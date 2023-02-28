import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from dash.exceptions import PreventUpdate

from appMain import app, shM

import Models.utils as ut


layout = html.Div([
    html.H3('Update BDD'),
    html.Div(dcc.Input(id='input-on-submit', type='text')),
    html.Button('Update', id='submit-val', n_clicks=0),
    dcc.Checklist(id='check_duplicate', value=['Check duplicate']),
    dcc.Textarea(
    id='textarea',
    value='Outputs',
    style={'width': '100%', 'height': '40%'},
    ),
    dcc.Interval(id='updater', interval=5000),
    html.Div(id='container-button-basic'),
    dcc.Link('Go to dashboard', href='/dashboard')
    #dcc.Interval(id="interval", interval=500)
])


@app.callback(
    Output('submit-val', 'value'),
    Input('submit-val', 'n_clicks'),
    Input('check_duplicate', 'value')
)
def update_output(n, value):
    if (n > 0):
        nbTotalShares =  shM.listShares.shape[0]
        nbSharesUpdated = 0
        for curShare in shM.listShares.itertuples():
            shM.updateShareCotations(curShare, checkDuplicate=False)
            nbSharesUpdated += 1
            ut.logOperation(f"Nb shares cotation updated: {nbSharesUpdated}/{nbTotalShares}")

        ut.logOperation("Success: All shares cotation updated")

        return "Updated"
    return "Update"

