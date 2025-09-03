from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import random
from app import app, shM
import datetime
from dash import html, dash_table
import time
import logging
import numpy as np
import pandas as pd
from Models import prediction_utils as pred_ut
from tensorflow.keras.callbacks import Callback
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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

# Callback to update date pickers based on graph zoom
@app.callback(
    [Output('date_picker_range', 'start_date'),
     Output('date_picker_range', 'end_date')],
    [Input('stock_graph', 'relayoutData')],
    [State('date_picker_range', 'start_date'),
     State('date_picker_range', 'end_date')]
)
def update_date_pickers(relayoutData, current_start_date, current_end_date):
    if relayoutData is None:
        return current_start_date, current_end_date
        
    if 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData:
        try:
            new_start = datetime.datetime.strptime(relayoutData['xaxis.range[0]'].split('.')[0], '%Y-%m-%d %H:%M:%S')
            new_end = datetime.datetime.strptime(relayoutData['xaxis.range[1]'].split('.')[0], '%Y-%m-%d %H:%M:%S')
            return new_start.strftime('%Y-%m-%d'), new_end.strftime('%Y-%m-%d')
        except Exception as e:
            print(f"Error parsing dates: {e}")
            return current_start_date, current_end_date
            
    if 'autosize' in relayoutData:
        return current_start_date, current_end_date
            
    return current_start_date, current_end_date

# Callback for updating performance metrics
@app.callback(
    Output('performance-metrics', 'children'),
    [Input('stock_graph', 'relayoutData'),
     Input('prediction_dropdown', 'value'),
     Input('date_picker_range', 'start_date'),
     Input('date_picker_range', 'end_date'),
     Input('data_type_selector', 'value'),
     Input('normalize_checkbox', 'value')]
)
def update_performance_metrics(*args):
    # Create a list to store performance metrics
    metrics = []
    
    # Get the latest performance metrics from the logs
    if hasattr(update_performance_metrics, 'latest_metrics'):
        metrics = [
            html.Div([
                html.Strong("Dernière mise à jour du graphique:"),
                html.Pre(update_performance_metrics.latest_metrics)
            ])
        ]
    
    return metrics

def reduce_data_points(x_values, y_values, max_points=1000):
    """Réduit le nombre de points de données pour l'affichage."""
    # S'assurer que nous avons des données à traiter
    if not x_values or not y_values or len(x_values) == 0 or len(y_values) == 0:
        logging.warning("No data points to reduce")
        return [], []
        
    if len(x_values) <= max_points:
        return x_values, y_values
        
    # Calculer l'intervalle d'échantillonnage
    step = len(x_values) // max_points
    
    # Réduire les données en prenant min/max dans chaque intervalle pour préserver les variations
    reduced_x = []
    reduced_y = []
    
    for i in range(0, len(x_values), step):
        chunk_x = x_values[i:i + step]
        chunk_y = y_values[i:i + step]
        
        # Ajouter le point min et max de l'intervalle
        if len(chunk_y) > 0:
            min_idx = np.argmin(chunk_y)
            max_idx = np.argmax(chunk_y)
            
            if min_idx <= max_idx:
                reduced_x.extend([chunk_x[min_idx], chunk_x[max_idx]])
                reduced_y.extend([chunk_y[min_idx], chunk_y[max_idx]])
            else:
                reduced_x.extend([chunk_x[max_idx], chunk_x[min_idx]])
                reduced_y.extend([chunk_y[max_idx], chunk_y[min_idx]])
    
    return reduced_x, reduced_y

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
    start_time = time.time()
    
    if values is None or len(values) == 0:
        print("No values selected.")
        fig = go.FigureWidget()
        fig.layout.template = 'plotly_dark'
        return fig
    
    # Handle zoom events from mouse wheel
    if relayoutData and ('xaxis.range' in relayoutData or ('xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData)):
        try:
            zoom_start_time = time.time()
            if 'xaxis.range' in relayoutData:
                dateBegin = datetime.datetime.strptime(relayoutData['xaxis.range'][0].split('.')[0], '%Y-%m-%d %H:%M:%S')
                dateLast = datetime.datetime.strptime(relayoutData['xaxis.range'][1].split('.')[0], '%Y-%m-%d %H:%M:%S')
            else:
                dateBegin = datetime.datetime.strptime(relayoutData['xaxis.range[0]'].split('.')[0], '%Y-%m-%d %H:%M:%S')
                dateLast = datetime.datetime.strptime(relayoutData['xaxis.range[1]'].split('.')[0], '%Y-%m-%d %H:%M:%S')
            zoom_parse_time = time.time() - zoom_start_time
            logging.info(f"Zoom date parsing took: {zoom_parse_time:.3f} seconds")
        except Exception as e:
            print(f"Error parsing dates: {e}")
            # If parsing fails, use the date picker values
            if end_date is not None:
                dateLast = datetime.datetime.fromisoformat(end_date)
            else:
                dateLast = datetime.datetime.now()

            if start_date is not None:
                dateBegin = datetime.datetime.fromisoformat(start_date)
            else:
                dateBegin = dateLast - datetime.timedelta(days=7)
    else:
        # Use date picker values if no zoom event
        if end_date is not None:
            dateLast = datetime.datetime.fromisoformat(end_date)
        else:
            dateLast = datetime.datetime.now()

        if start_date is not None:
            dateBegin = datetime.datetime.fromisoformat(start_date)
        else:
            dateBegin = dateLast - datetime.timedelta(days=7)
    
    logging.info(f"Date range: {dateBegin} to {dateLast}")

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
    db_start_time = time.time()
    dfShares = shM.getRowsDfByKeysValues(['symbol'] * len(values), values, op='|')
    db_query_time = time.time() - db_start_time
    logging.info(f"Database shares query took: {db_query_time:.3f} seconds")
    logging.info(f"dfShares shape: {dfShares.shape}")

    if dfShares.empty:
        logging.error("No data found for the selected symbols.")
        return fig

    # Create a set to track already added symbols
    added_symbols = set()

    # Retrieve data for each share within the selected date range
    data_start_time = time.time()
    listDfData = shM.getListDfDataFromDfShares(dfShares, dateBegin, dateLast)
    data_query_time = time.time() - data_start_time
    logging.info(f"Data retrieval took: {data_query_time:.3f} seconds")
    logging.info(f"Number of dataframes in listDfData: {len(listDfData)}")
    
    # Log if dataframes are empty
    for i, df in enumerate(listDfData):
        symbol = values[i] if i < len(values) else "unknown"
        logging.info(f"DataFrame {i} ({symbol}) is empty: {df.empty}")
        if not df.empty:
            logging.info(f"DataFrame {i} shape: {df.shape}")
            logging.info(f"DataFrame {i} columns: {df.columns.tolist()}")
            logging.info(f"DataFrame {i} first few rows: {df.head(2)}")

    # Add data to the graph
    plot_start_time = time.time()
    for i, dfData in enumerate(listDfData):
        if not dfData.empty and i < len(values) and values[i] not in added_symbols:
            added_symbols.add(values[i])
            logging.info(f"Processing data for {values[i]}")

            # Get the open and close market times for the current stock
            open_time = dfShares.iloc[i].openMarketTime
            close_time = dfShares.iloc[i].closeMarketTime
            logging.info(f"Market hours for {values[i]}: {open_time} to {close_time}")

            # Filter data based on the selected data type
            original_len = len(dfData)
            if data_type == 'main':
                try:
                    dfData = dfData.between_time(open_time.strftime('%H:%M'), close_time.strftime('%H:%M'))
                    logging.info(f"Filtered by market hours: {original_len} -> {len(dfData)} rows")
                except Exception as e:
                    logging.error(f"Error filtering by market hours: {e}")

            # Normalize if necessary
            toNormalize = 'normalize' in normalize_value
            if toNormalize and len(dfData) > 0 and 'openPrice' in dfData.columns and dfData['openPrice'].iloc[0] > 0:
                dfData['openPrice'] /= dfData['openPrice'].iloc[0]
                logging.info(f"Data normalized")

            # Convert data to JSON-compatible format
            try:
                x_values = dfData.index.astype(str).tolist()
                y_values = dfData['openPrice'].astype(float).tolist()
                logging.info(f"Converted data for graph: {len(x_values)} points")
                
                # Réduire le nombre de points si nécessaire
                x_reduced, y_reduced = reduce_data_points(x_values, y_values)
                logging.info(f"Reduced data points: {len(x_values)} -> {len(x_reduced)}")
                
                # Check if we have points to display
                if len(x_reduced) > 0 and len(y_reduced) > 0:
                    fig.add_scatter(
                        name=values[i],
                        x=x_reduced,
                        y=y_reduced,
                        hovertemplate='%{y:.2f}<extra></extra>'
                    )
                    logging.info(f"Added scatter trace for {values[i]} with {len(x_reduced)} points")
                else:
                    logging.warning(f"No points to display for {values[i]} after reduction")
            except Exception as e:
                logging.error(f"Error adding scatter for {values[i]}: {e}")
    
    plot_time = time.time() - plot_start_time
    logging.info(f"Plotting data took: {plot_time:.3f} seconds")
    logging.info(f"Number of traces in figure: {len(fig.data)}")

    # Preserve zoom level if it exists
    if relayoutData and 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData:
        fig.update_layout(
            xaxis_range=[relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']]
        )

    total_time = time.time() - start_time
    
    # Format performance metrics
    metrics_text = f"""
Temps total: {total_time:.3f}s
Détail des opérations:
- Requête base de données: {db_query_time:.3f}s ({(db_query_time/total_time)*100:.1f}%)
- Récupération données: {data_query_time:.3f}s ({(data_query_time/total_time)*100:.1f}%)
- Création graphique: {plot_time:.3f}s ({(plot_time/total_time)*100:.1f}%)
"""
    # Store metrics for the performance display callback
    update_performance_metrics.latest_metrics = metrics_text

    # Log metrics
    logging.info(f"Total graph update took: {total_time:.3f} seconds")
    logging.info(f"Performance breakdown:")
    logging.info(f"- Database shares query: {db_query_time:.3f}s ({(db_query_time/total_time)*100:.1f}%)")
    logging.info(f"- Data retrieval: {data_query_time:.3f}s ({(data_query_time/total_time)*100:.1f}%)")
    logging.info(f"- Plotting: {plot_time:.3f}s ({(plot_time/total_time)*100:.1f}%)")

    return fig

# Custom Keras callback to update Dash UI
class DashProgressCallback(Callback):
    def __init__(self, set_progress_func):
        super().__init__()
        self.set_progress = set_progress_func
        self.trials_results = []
        self.fig = go.Figure()
        self.fig.update_layout(
            title='Training and Validation Accuracy (Live)',
            xaxis_title='Epoch',
            yaxis_title='Accuracy',
            template='plotly_dark'
        )

    def on_trial_end(self, trial):
        trial_data = {
            'trial_id': trial.trial_id,
            'hyperparameters': str(trial.hyperparameters.values),
            'score': trial.score,
            'status': trial.status
        }
        self.trials_results.append(trial_data)
        
        results_df = pd.DataFrame(self.trials_results)
        results_df.rename(columns={
            'trial_id': 'Essai', 'hyperparameters': 'Hyperparamètres',
            'score': 'Score', 'status': 'Statut'
        }, inplace=True)

        table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in results_df.columns],
            data=results_df.to_dict('records'),
            style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto'},
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
            style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},
        )
        self.set_progress((table, self.fig))

# Combined callback for training and displaying metrics
@app.callback(
    [Output('training-results', 'children'),
     Output('model_metrics', 'children'),
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
     State('loss_function', 'value')],
    background=True,
    progress=[
        Output('model_metrics', 'children'),
        Output('accuracy_graph', 'figure')
    ],
    running=[
        (Output("train_button", "disabled"), True, False),
    ],
    prevent_initial_call=True,
)
def train_and_display_progress(set_progress, n_clicks, selected_symbols, training_days, 
                               train_test_ratio, look_back_x, stride_x, nb_y, 
                               nb_units, layers, learning_rate, loss_function):
    if not selected_symbols:
        return html.P("Veuillez sélectionner au moins une action."), "Sélectionnez une action.", go.Figure()

    symbol = selected_symbols[0]
    share_series = shM.getRowsDfByKeysValues(['symbol'], [symbol])
    if share_series.empty:
        return html.P(f"Aucune donnée pour {symbol}"), f"Erreur: {symbol} non trouvé.", go.Figure()
    shareObj = share_series.iloc[0]

    hps = {
        'layers': [layers], 'nb_units': [nb_units], 'learning_rate': [learning_rate],
        'loss': loss_function, 'max_trials': 5, 'executions_per_trial': 1,
        'directory': 'tuner_results_ui', 'project_name': f'pred_{symbol}_ui',
        'patience': 3, 'epochs': 15
    }
    
    data_info = {
        'look_back_x': look_back_x, 'stride_x': stride_x, 'nb_y': nb_y,
        'features': ['cotation', 'volume'], 'return_type': 'yield',
        'nb_days_to_take_dataset': training_days, 'percent_train_test': train_test_ratio,
        'shareObj': shareObj
    }

    try:
        dash_callback = DashProgressCallback(set_progress)
        best_model, best_hps, tuner = shM.train_share_model(shareObj, data_info, hps, callbacks=[dash_callback])
        
        final_results_df = pred_ut.tuner_results_to_dataframe(tuner)
        final_table = dash_table.DataTable(
            data=final_results_df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in final_results_df.columns],
            style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto'},
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
            style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},
        )

        history = best_model.fit(tuner.trainX, tuner.trainY, epochs=hps['epochs'], validation_data=(tuner.testX, tuner.testY))
        final_fig = go.Figure()
        final_fig.add_trace(go.Scatter(x=history.epoch, y=history.history['directional_accuracy'], name='Training Accuracy'))
        final_fig.add_trace(go.Scatter(x=history.epoch, y=history.history['val_directional_accuracy'], name='Validation Accuracy'))
        final_fig.update_layout(title=f'Résultats finaux pour {symbol}', template='plotly_dark')

        status_message = f"Entraînement terminé. Meilleur score: {tuner.oracle.get_best_trials(1)[0].score:.4f}"
        
        return status_message, final_table, final_fig

    except Exception as e:
        error_msg = html.Div([
            html.H5("Erreur lors de l'entraînement"),
            html.Pre(str(e)),
            html.Pre(traceback.format_exc())
        ])
        return error_msg, None, go.Figure()