
## File Descriptions

### Root Directory

#### `__init__.py`
*   **Purpose**: Standard Python file marking the root directory as a Python package.
*   **Content**: Empty.
*   **Dependencies**: None.

#### `test_models.py`
*   **Purpose**: Main script for developing, training, tuning, and evaluating stock price prediction models (LSTMs). Appears to be a comprehensive workflow, possibly originating from a Jupyter Notebook.
*   **Key Functionalities**:
    *   Data loading and extensive preprocessing (scaling, creating sequences for LSTMs).
    *   Defines LSTM model architecture using Keras, often with custom metrics like `directional_accuracy`.
    *   Hyperparameter tuning using Keras Tuner (RandomSearch, custom Hyperband).
    *   Training models with early stopping.
    *   Extensive visualization of data, tuning results, and model predictions using Plotly and Matplotlib.
    *   Compares model performance against baselines.
*   **Dependencies**: `sys`, `math`, `random`, `datetime`, `time`, `numpy`, `pandas`, `matplotlib`, `plotly`, `scikit-learn`, `tensorflow`, `keras`, `keras-tuner`, `bayes_opt`, and custom modules from the `Models` directory (`SqlCom`, `Shares`, `utils`).
*   **Note**: Contains several functions (e.g., `create_X_Y`, `build_model_closure`, `directional_accuracy`) that are also present in `Models/prediction_utils.py`, suggesting a potential refactoring opportunity or that `test_models.py` is a high-level script utilizing these refined utilities.

### `Models/` Directory

#### `Models/__init__.py`
*   **Purpose**: Standard Python file marking the `Models` directory as a Python package.
*   **Content**: Empty.
*   **Dependencies**: None.

#### `Models/Broker.py`
*   **Purpose**: Defines a `Broker` class to abstract interactions with a trading broker's API, currently hardcoded for Trading 212.
*   **Key Functionalities**:
    *   Manages API key and base URL for the broker.
    *   Calculates trading fees based on a predefined fee structure.
    *   Places market orders (`place_order`).
    *   Cancels orders (`cancel_order`).
    *   Retrieves order status (`get_order_status`).
*   **Dependencies**: `typing`, `requests`, `Models.Order`.

#### `Models/lstm.py`
*   **Purpose**: Provides basic functions for creating and training LSTM models. It seems to be a simpler or possibly earlier version compared to the model-building logic in `prediction_utils.py` or `test_models.py`.
*   **Key Functionalities**:
    *   `create_lstm_model`: Defines a simple Keras Sequential LSTM model.
    *   `train_lstm_model`: Trains a given Keras model.
    *   `compute_lstm`: Example workflow for preprocessing, training, and predicting (notes a dependency on a `preprocess_data` function to be defined).
*   **Dependencies**: `numpy`, `tensorflow.keras`.
*   **Note**: The `Shares.py` module imports `Models.lstm as ls` and calls `ls.test_lstm`, indicating this file might contain or previously contained a more complete `test_lstm` function used for per-share model training.

#### `Models/Order.py`
*   **Purpose**: Defines an `Order` class to represent and manage details of a trading order.
*   **Key Functionalities**:
    *   Stores order attributes (time, price, type, quantity, symbol, market, broker, status, filled details).
    *   Calculates order amount and fees (delegating fee calculation to the `Broker` object).
    *   Methods to update order status and check if filled, cancelled, or rejected.
*   **Dependencies**: `typing`, `datetime`, `Models.Broker`.

#### `Models/prediction_utils.py`
*   **Purpose**: Contains utility functions specifically for the prediction model workflow, including data preparation, custom Keras components, model building, and tuner result processing.
*   **Key Functionalities**:
    *   `create_X_Y`, `split_dataset_train_test`, `create_train_test`: Data preparation for LSTM models.
    *   `get_and_clean_data`: Fetches and cleans data.
    *   `directional_accuracy`: Custom Keras metric.
    *   `CustomHyperband`, `CustomRandomSearch`: Custom Keras Tuner classes.
    *   `build_model_closure`: Defines Keras LSTM model architecture for hyperparameter tuning (two versions present).
    *   `tuner_results_to_dataframe`: Processes Keras Tuner trial results.
    *   Outlined functions for training and selecting the best model (`train_and_select_best_model`, `train_with_hyperband`).
*   **Dependencies**: `numpy`, `pandas`, `tensorflow`, `keras`, `sklearn.preprocessing`, `datetime`, `Models.utils` (as `ut`), `math`, `random`, `keras_tuner`.
*   **Note**: This file has significant overlap with `test_models.py`, suggesting it's a refactored module for model-related utilities.

#### `Models/Shares.py`
*   **Purpose**: Defines the `Shares` class, a central component for managing stock information, quotation data, and orchestrating model training per share.
*   **Key Functionalities**:
    *   Initializes by connecting to the database (via `SqlCom`) and loading share information.
    *   Manages fetching, adding, and updating company information and historical price data (cotations) in the database.
    *   `updateAllSharesModels`: Iterates through shares, prepares data, triggers LSTM model training (using `Models.lstm.test_lstm`), and saves the trained models to the database.
    *   Provides various methods for retrieving share data and filtered lists of shares.
    *   Calculates and updates various statistics and metadata for shares (e.g., market times, data coverage).
*   **Dependencies**: `numpy`, `pandas`, `csv`, `datetime`, `re`, `dotenv`, `os`, `Models.SqlCom`, `Models.utils`, `Models.lstm`.

#### `Models/SqlCom.py`
*   **Purpose**: Defines the `SqlCom` class, which handles all direct communication with the PostgreSQL database.
*   **Key Functionalities**:
    *   Manages database connection (`psycopg2`).
    *   Saves and retrieves stock quotation data (`sharesPricesQuots` table).
    *   Saves and retrieves company information (`sharesInfos` table), including dynamic schema modification.
    *   Downloads data from Yahoo Finance (delegating to `Models.utils`) and stores it.
    *   Computes and stores various metadata about shares.
    *   Saves trained machine learning models (binary and scores) to the database.
    *   Exports data to CSV.
*   **Dependencies**: `psycopg2`, `csv`, `os`, `Models.utils`, `datetime`, `pandas`, `numpy`, `pytz`, `re`, `subprocess`.

#### `Models/trading212.py`
*   **Purpose**: Provides functions to directly interact with the Trading 212 API for trading operations.
*   **Key Functionalities**:
    *   `get_account_info`: Fetches account details.
    *   `buy_stock`: Places buy orders.
    *   `sell_stock`: Places sell orders.
*   **Dependencies**: `requests`, `logging`.

#### `Models/utils.py`
*   **Purpose**: A collection of general-purpose utility functions used throughout the project.
*   **Key Functionalities**:
    *   Logging operations and managing log file sizes (`data/logFile_*.log`).
    *   Downloading historical stock data and company info from Yahoo Finance (`yfinance` library).
    *   Extensive Pandas DataFrame manipulation (splitting, assembling, filling missing values, normalization, reindexing, interpolation).
    *   Date and time utilities, including timezone conversions.
    *   Helper for saving DataFrames to files.
    *   Correlation analysis with time shifts (`temporalComparison`).
    *   A custom sorted dictionary class (`dicoSortedValue`).
    *   Functions for Z-score normalization and other calculations (`getPotential`).
*   **Dependencies**: `datetime`, `matplotlib.pyplot`, `pandas`, `yfinance`, `pytz`, `numpy`, `collections.OrderedDict`.

#### `Models/Visualizer.py`
*   **Purpose**: Provides functions for creating visualizations using `matplotlib`.
*   **Key Functionalities**:
    *   `display`: Simple scatter plot.
    *   `potentialDisplay`: Plots a series along with "potential" points calculated by `utils.getPotential`.
    *   `displayQuots`: Plots quotation data for a share.
*   **Dependencies**: `numpy`, `matplotlib.pyplot`, `Models.utils` (as `ut`), `Models.Shares` (as `sm`).

#### `Models/Wallet.py`
*   **Purpose**: Defines a `Wallet` class to simulate or track a trading portfolio's cash, investments, and performance.
*   **Key Functionalities**:
    *   Manages initial and current cash amounts.
    *   Tracks orders, total profit, and total fees.
    *   `add_order`: Updates cash and fees when an order is placed.
    *   `close_order`: Updates cash and profit when an order is closed.
    *   Provides methods to get income, current amount, total invested, fees, and profit.
*   **Dependencies**: `typing`, `Models.Order`, `Models.Broker`.

## Implicit Directories

### `data/`
*   **Purpose**: Stores log files generated by `Models/utils.py`. Log files are named `logFile_{DayName}.log`.

## Overall Workflow Summary

1.  **Data Management (`SqlCom`, `Shares`, `utils`):**
    *   Stock symbols and company information are fetched (from Yahoo Finance via `utils`) and stored/updated in a PostgreSQL database (`SqlCom`, managed by `Shares`).
    *   Historical price data (cotations) is similarly downloaded and stored.
    *   Various metadata (market times, data quality metrics) are computed and saved.
2.  **Model Development & Training (`test_models.py`, `prediction_utils.py`, `Models/lstm.py`):**
    *   Data is retrieved and extensively preprocessed for time-series modeling.
    *   LSTM models are defined, with a focus on predicting price direction (`directional_accuracy`).
    *   Hyperparameter tuning is performed using Keras Tuner.
    *   The best models are trained and can be saved to the database (via `Shares` and `SqlCom`).
3.  **Trading Simulation/Execution (`Broker`, `Order`, `Wallet`, `trading212.py`):**
    *   A `Broker` class (interfacing with Trading 212 API via `trading212.py` and direct `requests` calls) handles order placement and status checks.
    *   `Order` objects represent individual trades.
    *   A `Wallet` class tracks portfolio value, profits, and fees.
4.  **Visualization (`Visualizer.py`, `test_models.py`):**
    *   Data, model performance, and "potential" metrics are visualized using `matplotlib` and `plotly`.

The project is structured to separate concerns: data persistence (`SqlCom`), stock data management (`Shares`), general utilities (`utils`), model-specific logic (`prediction_utils`, `lstm`), broker interaction (`Broker`, `trading212`), and high-level scripting/experimentation (`test_models.py`).