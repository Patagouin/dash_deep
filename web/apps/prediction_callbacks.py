from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import random
from app import app, shM
import datetime
from dash import html

# Callback to update shares based on selected sector
@app.callback(
    Output('prediction_dropdown', 'options'),
    Input('sector_dropdown', 'value')
)
def update_shares_based_on_sector(selected_sector):
    if selected_sector == 'Non défini':
        # Filter the shares where sector is NaN
        filtered_shares = shM.dfShares[shM.dfShares['sector'].isna()]
    elif selected_sector:
        # Filter the shares based on the selected sector
        filtered_shares = shM.dfShares[shM.dfShares['sector'] == selected_sector]
    else:
        # If no sector is selected, use all shares
        filtered_shares = shM.dfShares

    # Sort the filtered shares by the 'symbol' column
    sorted_shares = filtered_shares.sort_values(by='symbol')

    # Generate the options for the dropdown, including the "All" option
    return [{'label': 'All', 'value': 'All'}] + [
        {'label': '{}'.format(stock.symbol), 'value': stock.symbol} 
        for stock in sorted_shares.itertuples()
    ]

# Callback to update the training days slider based on selected shares
@app.callback(
    Output('training_days_slider', 'max'),
    Output('training_days_slider', 'marks'),
    Input('prediction_dropdown', 'value')
)
def update_days_slider(selected_symbols):
    if not selected_symbols:
        return 365, {
            7: '7j',
            30: '1m',
            90: '3m',
            180: '6m',
            365: '1a'
        }
    
    # Calculate the maximum number of days available for the selected shares
    max_days = 0
    for symbol in selected_symbols:
        dfShares = shM.getRowsDfByKeysValues(['symbol'], [symbol])
        if not dfShares.empty:
            dfData = shM.getListDfDataFromDfShares(dfShares)[0]
            if not dfData.empty:
                days_available = (dfData.index.max() - dfData.index.min()).days
                max_days = max(max_days, days_available)
    
    # Round up to the nearest multiple of 30
    max_days = min(((max_days + 29) // 30) * 30, 365)
    
    # Create dynamic marks
    marks = {
        7: '7j',
        30: '1m',
    }
    
    if max_days >= 90:
        marks[90] = '3m'
    if max_days >= 180:
        marks[180] = '6m'
    if max_days >= 365:
        marks[365] = '1a'
    
    return max_days, marks

# Callback for the stock graph
@app.callback(
    Output('stock_graph', 'figure'),
    [Input('prediction_dropdown', 'value'),
     Input('date_picker_range', 'start_date'),
     Input('date_picker_range', 'end_date'),
     Input('data_type_selector', 'value'),
     Input('normalize_checkbox', 'value'),
     Input('stock_graph', 'relayoutData')]
)
def display_stock_graph(values, start_date, end_date, data_type, normalize_value, relayoutData):
    if values is None or len(values) == 0:
        print("No values selected.")
        fig = go.FigureWidget()
        fig.layout.template = 'plotly_dark'
        return fig
    
    # Conversion of start and end dates
    if end_date is not None:
        dateLast = datetime.datetime.fromisoformat(end_date)
    else:
        dateLast = datetime.datetime.now()

    if start_date is not None:
        dateBegin = datetime.datetime.fromisoformat(start_date)
    else:
        dateBegin = dateLast - datetime.timedelta(days=7)

    # Create an empty figure with a dark theme
    fig = go.FigureWidget()
    fig.layout.template = 'plotly_dark'
    fig.layout.hovermode = "x unified"
    fig.update_layout(
        hoverlabel=dict(
            bgcolor='#FFFFFF',
            font=dict(color='#000000'),
            bordercolor='#FFFFFF'
        ),
        legend=dict(
            font=dict(color='#FFFFFF'),
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='#FFFFFF'
        )
    )
    fig.layout.title = 'Shares quots'
    
    # Retrieve data for the selected shares
    dfShares = shM.getRowsDfByKeysValues(['symbol'] * len(values), values, op='|')

    if dfShares.empty:
        print("No data found for the selected symbols.")
        return fig

    # Create a set to track already added symbols
    added_symbols = set()

    # Retrieve data for each share within the selected date range
    listDfData = shM.getListDfDataFromDfShares(dfShares, dateBegin, dateLast)

    # Add data to the graph
    for i, dfData in enumerate(listDfData):
        if not dfData.empty and values[i] not in added_symbols:
            added_symbols.add(values[i])

            # Get the open and close market times for the current stock
            open_time = dfShares.iloc[i].openMarketTime
            close_time = dfShares.iloc[i].closeMarketTime

            # Filter data based on the selected data type
            if data_type == 'main':
                dfData = dfData.between_time(open_time.strftime('%H:%M'), close_time.strftime('%H:%M'))

            # Normalize if necessary
            toNormalize = 'normalize' in normalize_value
            if toNormalize and dfData['openPrice'][0] > 0:
                dfData['openPrice'] /= dfData['openPrice'][0]

            # Convert data to JSON-compatible format
            x_values = dfData.index.astype(str).tolist()
            y_values = dfData['openPrice'].astype(float).tolist()

            fig.add_scatter(
                name=values[i],
                x=x_values,
                y=y_values,
                hovertemplate='%{y:.2f}<extra></extra>'
            )

    # Apply the previous zoom level if available
    if relayoutData and 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData:
        fig.update_layout(
            xaxis_range=[relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']]
        )

    return fig

# Callback for training the model
@app.callback(
    [Output('training-results', 'children'),
     Output('accuracy_graph', 'figure')],
    [Input('train_button', 'n_clicks')],
    [State('prediction_dropdown', 'value'),
     State('training_days_slider', 'value'),
     State('train_test_ratio_slider', 'value'),
     State('look_back_x', 'value'),
     State('stride_x', 'value'),
     State('nb_y', 'value'),
     State('nb_units', 'value'),
     State('layers', 'value'),
     State('learning_rate', 'value'),
     State('loss_function', 'value')]
)
def train_model(n_clicks, selected_symbols, training_days, train_test_ratio, 
                look_back_x, stride_x, nb_y, nb_units, layers, learning_rate, loss_function):
    if n_clicks is None or n_clicks == 0:
        return "", go.Figure()

    # Simulate training process (replace with actual training logic)
    epochs = 10
    train_accuracy = []
    val_accuracy = []
    avg_percentage_diff = 0
    correct_direction_percentage = 0

    # Simulate training and validation accuracy over epochs
    for epoch in range(epochs):
        train_acc = random.uniform(0.7, 0.9)  # Simulated training accuracy
        val_acc = random.uniform(0.6, 0.85)  # Simulated validation accuracy
        train_accuracy.append(train_acc)
        val_accuracy.append(val_acc)

    # Simulate average percentage difference and correct directional predictions
    avg_percentage_diff = random.uniform(1, 5)  # Simulated percentage difference
    correct_direction_percentage = random.uniform(70, 90)  # Simulated correct direction percentage

    # Create the accuracy graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(epochs)), y=train_accuracy, mode='lines', name='Training Accuracy'))
    fig.add_trace(go.Scatter(x=list(range(epochs)), y=val_accuracy, mode='lines', name='Validation Accuracy'))
    fig.update_layout(
        title='Training and Validation Accuracy',
        xaxis_title='Epoch',
        yaxis_title='Accuracy',
        template='plotly_dark'
    )

    # Display the results
    results = [
        html.P(f"Pourcentage moyen de différence: {avg_percentage_diff:.2f}%"),
        html.P(f"Pourcentage de prédictions correctes: {correct_direction_percentage:.2f}%")
    ]

    return results, fig

# Callback to display model metrics
@app.callback(
    Output('model_metrics', 'children'),
    [Input('train_button', 'n_clicks')],
    [State('prediction_dropdown', 'value'),
     State('training_days_slider', 'value'),
     State('train_test_ratio_slider', 'value'),
     State('look_back_x', 'value'),
     State('stride_x', 'value'),
     State('nb_y', 'value'),
     State('nb_units', 'value'),
     State('layers', 'value'),
     State('learning_rate', 'value'),
     State('loss_function', 'value')]
)
def display_model_metrics(n_clicks, selected_symbols, training_days, train_test_ratio, 
                          look_back_x, stride_x, nb_y, nb_units, layers, learning_rate, loss_function):
    if n_clicks is None or n_clicks == 0:
        return "Aucun résultat disponible. Veuillez entraîner le modèle."

    # Logic to calculate model metrics
    # For now, we simulate some metrics
    metrics = {
        'Perte finale': 0.025,
        'Exactitude': '85%',
        'Précision': '80%',
        'Rappel': '75%'
    }

    # Format metrics for display
    metrics_display = [
        html.P(f"{key}: {value}") for key, value in metrics.items()
    ]

    return metrics_display