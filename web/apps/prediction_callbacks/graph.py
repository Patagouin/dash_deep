from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from app import app, shM
import datetime
from dash import html
import time
import logging
import numpy as np
import dash


@app.callback(
    Output('performance-metrics', 'children'),
    [Input('stock_graph', 'relayoutData'),
     Input('train_share_list', 'value'),
     Input('date_picker_range', 'start_date'),
     Input('date_picker_range', 'end_date'),
     Input('data_type_selector', 'value'),
     Input('normalize_checkbox', 'value')]
)
def update_performance_metrics(*args):
    metrics = []
    if hasattr(update_performance_metrics, 'latest_metrics'):
        metrics = [
            html.Div([
                html.Strong("Dernière mise à jour du graphique:"),
                html.Pre(update_performance_metrics.latest_metrics)
            ])
        ]
    return metrics


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


def reduce_data_points(x_values, y_values, max_points=1000):
    if not x_values or not y_values or len(x_values) == 0 or len(y_values) == 0:
        logging.warning("No data points to reduce")
        return [], []

    if len(x_values) <= max_points:
        return x_values, y_values

    step = len(x_values) // max_points

    reduced_x = []
    reduced_y = []

    for i in range(0, len(x_values), step):
        chunk_x = x_values[i:i + step]
        chunk_y = y_values[i:i + step]

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


@app.callback(
    Output('stock_graph', 'figure'),
    [Input('train_share_list', 'value'),
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

    # Déterminer le déclencheur pour éviter que relayoutData n'écrase les dates choisies
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'] if ctx and ctx.triggered else None
    use_relayout = False
    if trigger_id and trigger_id.startswith('stock_graph.relayoutData') and relayoutData:
        if 'xaxis.range' in relayoutData or ('xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData):
            use_relayout = True

    if use_relayout:
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
            if end_date is not None:
                dateLast = datetime.datetime.fromisoformat(end_date)
            else:
                dateLast = datetime.datetime.now()

            if start_date is not None:
                dateBegin = datetime.datetime.fromisoformat(start_date)
            else:
                dateBegin = dateLast - datetime.timedelta(days=7)
    else:
        if end_date is not None:
            dateLast = datetime.datetime.fromisoformat(end_date)
        else:
            dateLast = datetime.datetime.now()

        if start_date is not None:
            dateBegin = datetime.datetime.fromisoformat(start_date)
        else:
            dateBegin = dateLast - datetime.timedelta(days=7)

    logging.info(f"Date range: {dateBegin} to {dateLast}")

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

    db_start_time = time.time()
    dfShares = shM.getRowsDfByKeysValues(['symbol'] * len(values), values, op='|')
    db_query_time = time.time() - db_start_time
    logging.info(f"Database shares query took: {db_query_time:.3f} seconds")
    logging.info(f"dfShares shape: {dfShares.shape}")

    if dfShares.empty:
        logging.error("No data found for the selected symbols.")
        return fig

    added_symbols = set()

    data_start_time = time.time()
    listDfData = shM.getListDfDataFromDfShares(dfShares, dateBegin, dateLast)
    data_query_time = time.time() - data_start_time
    logging.info(f"Data retrieval took: {data_query_time:.3f} seconds")
    logging.info(f"Number of dataframes in listDfData: {len(listDfData)}")

    for i, df in enumerate(listDfData):
        symbol = values[i] if i < len(values) else "unknown"
        logging.info(f"DataFrame {i} ({symbol}) is empty: {df.empty}")
        if not df.empty:
            logging.info(f"DataFrame {i} shape: {df.shape}")
            logging.info(f"DataFrame {i} columns: {df.columns.tolist()}")
            logging.info(f"DataFrame {i} first few rows: {df.head(2)}")

    plot_start_time = time.time()
    for i, dfData in enumerate(listDfData):
        if not dfData.empty and i < len(values) and values[i] not in added_symbols:
            added_symbols.add(values[i])
            logging.info(f"Processing data for {values[i]}")

            open_time = dfShares.iloc[i].openMarketTime
            close_time = dfShares.iloc[i].closeMarketTime
            logging.info(f"Market hours for {values[i]}: {open_time} to {close_time}")

            original_len = len(dfData)
            if data_type == 'main':
                try:
                    dfData = dfData.between_time(open_time.strftime('%H:%M'), close_time.strftime('%H:%M'))
                    logging.info(f"Filtered by market hours: {original_len} -> {len(dfData)} rows")
                except Exception as e:
                    logging.error(f"Error filtering by market hours: {e}")

            toNormalize = 'normalize' in normalize_value
            if toNormalize and len(dfData) > 0 and 'openPrice' in dfData.columns and dfData['openPrice'].iloc[0] > 0:
                dfData['openPrice'] /= dfData['openPrice'].iloc[0]

            try:
                x_values = dfData.index.astype(str).tolist()
                y_values = dfData['openPrice'].astype(float).tolist()

                x_reduced, y_reduced = reduce_data_points(x_values, y_values)

                if len(x_reduced) > 0 and len(y_reduced) > 0:
                    fig.add_scatter(
                        name=values[i],
                        x=x_reduced,
                        y=y_reduced,
                        hovertemplate='%{y:.2f}<extra></extra>'
                    )
                else:
                    logging.warning(f"No points to display for {values[i]} after reduction")
            except Exception as e:
                logging.error(f"Error adding scatter for {values[i]}: {e}")

    plot_time = time.time() - plot_start_time
    logging.info(f"Plotting data took: {plot_time:.3f} seconds")
    logging.info(f"Number of traces in figure: {len(fig.data)}")

    if relayoutData and 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData:
        fig.update_layout(
            xaxis_range=[relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']]
        )

    total_time = time.time() - start_time

    metrics_text = f"""
Temps total: {total_time:.3f}s
Détail des opérations:
- Requête base de données: {db_query_time:.3f}s ({(db_query_time/total_time)*100:.1f}%)
- Récupération données: {data_query_time:.3f}s ({(data_query_time/total_time)*100:.1f}%)
- Création graphique: {plot_time:.3f}s ({(plot_time/total_time)*100:.1f}%)
"""
    update_performance_metrics.latest_metrics = metrics_text

    logging.info(f"Total graph update took: {total_time:.3f} seconds")

    return fig


