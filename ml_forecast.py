#!/usr/bin/env python3
"""
ML-based return forecaster for TimeFolio portfolio system.
Provides LSTM and XGBoost implementations for weekly return predictions.

Features:
- Explicit train/predict date boundary to prevent look-ahead bias
- Enhanced feature set including price, volume, volatility and market factors
- Support for both LSTM and XGBoost models
"""
import os
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy.stats import skew, kurtosis

# ML libraries
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)


class MLForecaster:
    """ML-based forecaster base class."""
    
    def __init__(self, model_dir: str = 'models', lookback_periods: int = 12):
        """
        Initialize the ML forecaster.
        
        Args:
            model_dir: Directory to save/load trained models
            lookback_periods: Number of historical periods to use for prediction
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.lookback_periods = lookback_periods
        self.scalers = {}
        self.models = {}
        self.training_end_dates = {}  # Store the last date used for training each model
        
    def _prepare_features(self, data: pd.DataFrame, ticker: str, 
                            training_mode: bool = True, 
                            cutoff_date: Optional[pd.Timestamp] = None) -> Tuple[np.ndarray, np.ndarray, Optional[pd.Timestamp]]:
        """
        Create enhanced features and targets for a single ticker.
        
        Args:
            data: DataFrame containing returns and additional features (volume, etc.)
            ticker: The ticker to prepare features for
            training_mode: Whether to prepare data for training (True) or prediction (False)
            cutoff_date: Optional date to separate training from validation/prediction data
            
        Returns:
            X: Features array
            y: Target array (or None if prediction mode)
            last_date: Last date used in the dataset (for tracking training boundaries)
        """
        # Ensure we have a DataFrame with a datetime index for proper date handling
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning(f"Data for {ticker} does not have DatetimeIndex, attempting conversion")
            try:
                data.index = pd.to_datetime(data.index)
            except:
                logger.error(f"Could not convert index to datetime for {ticker}")
                return None, None, None
        
        # Apply cutoff date if provided (to prevent look-ahead bias)
        if cutoff_date is not None:
            if training_mode:
                # For training, use data up to cutoff_date
                mask = data.index <= cutoff_date
                if mask.sum() < self.lookback_periods + 10:  # Ensure minimum data
                    logger.warning(f"Insufficient data for {ticker} up to {cutoff_date}")
                    return None, None, None
                data = data.loc[mask]
            else:
                # For prediction, verify we're not using future data relative to last training
                if ticker in self.training_end_dates:
                    last_train_date = self.training_end_dates[ticker]
                    if cutoff_date <= last_train_date:
                        logger.warning(f"Prediction date {cutoff_date} is not after training cutoff {last_train_date} for {ticker}")
                
        # Store the last date in the dataset for training boundary tracking
        last_date = data.index[-1] if not data.empty else None
        
        # Extract base returns series and create extended features
        if ticker not in data.columns and f"{ticker}_return" in data.columns:
            returns_col = f"{ticker}_return"
        else:
            returns_col = ticker
            
        # Check if we have the necessary data
        if returns_col not in data.columns:
            logger.error(f"Return data for {ticker} not found in dataset")
            return None, None, None
            
        # Extract returns and create base series
        returns = data[returns_col].dropna()
        if len(returns) <= self.lookback_periods:
            logger.warning(f"Insufficient return data for {ticker}")
            return None, None, None
            
        # Create feature DataFrame with various derived metrics
        features = pd.DataFrame(index=returns.index)
        
        # 1. Basic return features
        features['return'] = returns
        
        # 2. Add volatility features (if we have enough data)
        if len(returns) >= 10:
            features['volatility'] = returns.rolling(10).std()
            features['volatility_20d'] = returns.rolling(20).std()
            
        # 3. Add momentum features
        features['momentum_5d'] = returns.rolling(5).sum()
        features['momentum_20d'] = returns.rolling(20).sum()
        
        # 4. Add volume features if available
        vol_col = f"{ticker}_volume" if f"{ticker}_volume" in data.columns else None
        if vol_col and vol_col in data.columns:
            volume = data[vol_col].dropna()
            if len(volume) > 0:
                features['volume'] = volume
                features['volume_change'] = volume.pct_change()
                features['volume_ratio_5d'] = volume / volume.rolling(5).mean()
                # Volume-price relationship
                features['vol_ret_corr'] = returns.rolling(10).corr(volume.pct_change())
        
        # 5. Add market relative features if available
        market_col = 'KS11_return' if 'KS11_return' in data.columns else None
        if market_col and market_col in data.columns:
            market_ret = data[market_col].dropna()
            if len(market_ret) > 0:
                features['market_return'] = market_ret
                # Calculate beta over 20-day rolling window if enough data
                if len(market_ret) >= 20:
                    features['beta_20d'] = returns.rolling(20).cov(market_ret) / market_ret.rolling(20).var()
                # Calculate relative strength
                features['rel_strength'] = returns - market_ret
        
        # 6. Add statistical moments to capture distribution characteristics
        if len(returns) >= 20:
            features['skewness'] = returns.rolling(20).apply(lambda x: skew(x) if len(x.dropna()) > 5 else np.nan, raw=True)
            features['kurtosis'] = returns.rolling(20).apply(lambda x: kurtosis(x) if len(x.dropna()) > 5 else np.nan, raw=True)
        
        # Fill any NaNs with 0 after creating all features
        features = features.fillna(0)
        
        # Create sequences for ML models
        X, y = [], []
        
        # Convert DataFrame to numpy for sequence creation
        feature_array = features.values
        n_features = feature_array.shape[1]  # Number of feature columns
        
        for i in range(len(feature_array) - self.lookback_periods):
            if training_mode:
                # For training, we need both X and y
                X.append(feature_array[i:i+self.lookback_periods])
                y.append(feature_array[i+self.lookback_periods, 0])  # First column is 'return'
            else:
                # For prediction, we only need the last sequence
                if i == len(feature_array) - self.lookback_periods - 1:
                    X.append(feature_array[i:i+self.lookback_periods])
        
        X = np.array(X)
        y = np.array(y) if training_mode else None
        
        # Scale features - each feature type separately
        if X.shape[0] > 0:
            if ticker not in self.scalers:
                self.scalers[ticker] = StandardScaler()
                # Reshape to 2D for scaling each feature independently
                reshaped = X.reshape(-1, n_features) 
                scaled = self.scalers[ticker].fit_transform(reshaped)
                X_scaled = scaled.reshape(X.shape)
            else:
                reshaped = X.reshape(-1, n_features)
                scaled = self.scalers[ticker].transform(reshaped)
                X_scaled = scaled.reshape(X.shape)
                
            return X_scaled, y, last_date
        
        return None, None, None
    
    def save_models(self):
        """Save all models and scalers to disk."""
        # Save scalers
        with open(self.model_dir / 'scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
            
        # Save training end dates to enforce temporal boundaries
        with open(self.model_dir / 'training_end_dates.pkl', 'wb') as f:
            pickle.dump(self.training_end_dates, f)
        
        # Specific model saving implemented in subclasses
        
    def load_models(self) -> bool:
        """
        Load models and scalers from disk.
        
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        scaler_path = self.model_dir / 'scalers.pkl'
        date_path = self.model_dir / 'training_end_dates.pkl'
        
        success = False
        
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scalers = pickle.load(f)
            success = True
                
        if date_path.exists():
            with open(date_path, 'rb') as f:
                self.training_end_dates = pickle.load(f)
            logger.info(f"Loaded training boundary dates for {len(self.training_end_dates)} models")
        else:
            logger.warning("No training boundary dates found, look-ahead prevention may be limited")
            
        return success
        return False
    
    def train(self, returns: pd.DataFrame) -> Dict[str, Dict]:
        """
        Train models for all tickers.
        
        Args:
            returns: DataFrame of historical returns
            
        Returns:
            Dict of model metrics by ticker
        """
        raise NotImplementedError("Subclasses must implement train method")
    
    def forecast_returns(self, returns: pd.DataFrame) -> pd.Series:
        """
        Generate return forecasts for all tickers.
        
        Args:
            returns: DataFrame of historical returns
            
        Returns:
            Series of predicted returns
        """
        raise NotImplementedError("Subclasses must implement forecast_returns method")


class LSTMForecaster(MLForecaster):
    """LSTM-based return forecaster."""
    
    def __init__(self, model_dir: str = 'models/lstm', lookback_periods: int = 12,
                 lstm_units: int = 50, dropout_rate: float = 0.1, epochs: int = 100):
        """
        Initialize LSTM forecaster.
        
        Args:
            model_dir: Directory to save/load trained models
            lookback_periods: Number of historical periods to use for prediction
            lstm_units: Number of LSTM units in the model
            dropout_rate: Dropout rate for regularization
            epochs: Maximum number of training epochs
        """
        super().__init__(model_dir, lookback_periods)
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        
    def _build_model(self, input_shape: Tuple) -> Sequential:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input features
            
        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential([
            LSTM(self.lstm_units, activation='tanh', input_shape=input_shape, 
                 return_sequences=True),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units // 2, activation='tanh'),
            Dropout(self.dropout_rate),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train(self, returns: pd.DataFrame) -> Dict[str, Dict]:
        """
        Train LSTM models for all tickers.
        
        Args:
            returns: DataFrame of historical returns
            
        Returns:
            Dict of model metrics by ticker
        """
        tickers = returns.columns
        metrics = {}
        
        for ticker in tickers:
            logger.info(f"Training LSTM model for {ticker}")
            try:
                X, y = self._prepare_features(returns, ticker)
                
                # Create time series cross-validation splits
                tscv = TimeSeriesSplit(n_splits=5)
                last_split = None
                
                for train_idx, test_idx in tscv.split(X):
                    last_split = (train_idx, test_idx)
                
                if last_split is None:
                    logger.warning(f"Not enough data to train model for {ticker}")
                    continue
                    
                train_idx, test_idx = last_split
                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]
                
                # Build and train model
                model = self._build_model((X_train.shape[1], 1))
                
                # Setup callbacks
                callbacks = [
                    EarlyStopping(patience=10, restore_best_weights=True),
                    ModelCheckpoint(
                        self.model_dir / f"{ticker}_best_model.h5",
                        save_best_only=True,
                        monitor='val_loss'
                    )
                ]
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=self.epochs,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Evaluate model
                y_pred = model.predict(X_test).flatten()
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                metrics[ticker] = {
                    'mse': mse,
                    'r2': r2,
                    'val_loss': min(history.history['val_loss'])
                }
                
                # Save the model
                self.models[ticker] = model
                
            except Exception as e:
                logger.error(f"Error training model for {ticker}: {e}")
                
        self.save_models()
        return metrics
    
    def save_models(self):
        """Save LSTM models and scalers to disk."""
        super().save_models()
        for ticker, model in self.models.items():
            try:
                model.save(self.model_dir / f"{ticker}_model.h5")
            except Exception as e:
                logger.error(f"Error saving model for {ticker}: {e}")
    
    def load_models(self) -> bool:
        """
        Load LSTM models and scalers from disk.
        
        Returns:
            bool: True if at least some models loaded successfully, False otherwise
        """
        if not super().load_models():
            return False
            
        success = False
        for file in self.model_dir.glob("*_model.h5"):
            ticker = file.name.split("_model.h5")[0]
            try:
                self.models[ticker] = load_model(file)
                success = True
            except Exception as e:
                logger.error(f"Error loading model for {ticker}: {e}")
                
        return success
    
    def forecast_returns(self, returns: pd.DataFrame) -> pd.Series:
        """
        Generate LSTM return forecasts for all tickers.
        
        Args:
            returns: DataFrame of historical returns
            
        Returns:
            Series of predicted returns
        """
        predictions = {}
        
        # Load models if not already loaded
        if not self.models:
            if not self.load_models():
                logger.warning("No models found, falling back to historical mean")
                return returns.mean()
        
        for ticker in returns.columns:
            if ticker not in self.models:
                logger.warning(f"No model for {ticker}, using historical mean")
                predictions[ticker] = returns[ticker].mean()
                continue
                
            try:
                # Prepare most recent data point for prediction
                data = returns[ticker].dropna().values
                
                if len(data) < self.lookback_periods:
                    logger.warning(f"Not enough data for {ticker}, using historical mean")
                    predictions[ticker] = returns[ticker].mean()
                    continue
                
                # Take the most recent lookback_periods data points
                X = data[-self.lookback_periods:].reshape(1, self.lookback_periods, 1)
                
                # Scale features
                X_scaled = self.scalers[ticker].transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
                
                # Make prediction
                pred = self.models[ticker].predict(X_scaled, verbose=0)[0][0]
                predictions[ticker] = pred
                
            except Exception as e:
                logger.error(f"Error forecasting for {ticker}: {e}")
                predictions[ticker] = returns[ticker].mean()
                
        return pd.Series(predictions)


class XGBoostForecaster(MLForecaster):
    """XGBoost-based return forecaster."""
    
    def __init__(self, model_dir: str = 'models/xgb', lookback_periods: int = 12, 
                 n_estimators: int = 100, max_depth: int = 3, learning_rate: float = 0.1):
        """
        Initialize XGBoost forecaster.
        
        Args:
            model_dir: Directory to save/load trained models
            lookback_periods: Number of historical periods to use for prediction
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
        """
        super().__init__(model_dir, lookback_periods)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        
    def _build_model(self) -> xgb.XGBRegressor:
        """
        Build XGBoost model.
        
        Returns:
            XGBoost regressor model
        """
        return xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective='reg:squarederror',
            random_state=42
        )
    
    def train(self, returns: pd.DataFrame) -> Dict[str, Dict]:
        """
        Train XGBoost models for all tickers.
        
        Args:
            returns: DataFrame of historical returns
            
        Returns:
            Dict of model metrics by ticker
        """
        tickers = returns.columns
        metrics = {}
        
        for ticker in tickers:
            logger.info(f"Training XGBoost model for {ticker}")
            try:
                X, y = self._prepare_features(returns, ticker)
                X = X.reshape(X.shape[0], -1)  # Flatten the features for XGBoost
                
                # Create time series cross-validation splits
                tscv = TimeSeriesSplit(n_splits=5)
                last_split = None
                
                for train_idx, test_idx in tscv.split(X):
                    last_split = (train_idx, test_idx)
                
                if last_split is None:
                    logger.warning(f"Not enough data to train model for {ticker}")
                    continue
                    
                train_idx, test_idx = last_split
                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]
                
                # Train model
                model = self._build_model()
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    early_stopping_rounds=10,
                    verbose=0
                )
                
                # Evaluate model
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                metrics[ticker] = {
                    'mse': mse,
                    'r2': r2,
                    'best_iteration': model.get_booster().best_iteration
                }
                
                # Save model
                self.models[ticker] = model
                
            except Exception as e:
                logger.error(f"Error training model for {ticker}: {e}")
                
        self.save_models()
        return metrics
    
    def save_models(self):
        """Save XGBoost models and scalers to disk."""
        super().save_models()
        for ticker, model in self.models.items():
            try:
                model.save_model(str(self.model_dir / f"{ticker}_model.json"))
            except Exception as e:
                logger.error(f"Error saving model for {ticker}: {e}")
    
    def load_models(self) -> bool:
        """
        Load XGBoost models and scalers from disk.
        
        Returns:
            bool: True if at least some models loaded successfully, False otherwise
        """
        if not super().load_models():
            return False
            
        success = False
        for file in self.model_dir.glob("*_model.json"):
            ticker = file.name.split("_model.json")[0]
            try:
                model = self._build_model()
                model.load_model(str(file))
                self.models[ticker] = model
                success = True
            except Exception as e:
                logger.error(f"Error loading model for {ticker}: {e}")
                
        return success
    
    def forecast_returns(self, returns: pd.DataFrame) -> pd.Series:
        """
        Generate XGBoost return forecasts for all tickers.
        
        Args:
            returns: DataFrame of historical returns
            
        Returns:
            Series of predicted returns
        """
        predictions = {}
        
        # Load models if not already loaded
        if not self.models:
            if not self.load_models():
                logger.warning("No models found, falling back to historical mean")
                return returns.mean()
        
        for ticker in returns.columns:
            if ticker not in self.models:
                logger.warning(f"No model for {ticker}, using historical mean")
                predictions[ticker] = returns[ticker].mean()
                continue
                
            try:
                # Prepare most recent data point for prediction
                data = returns[ticker].dropna().values
                
                if len(data) < self.lookback_periods:
                    logger.warning(f"Not enough data for {ticker}, using historical mean")
                    predictions[ticker] = returns[ticker].mean()
                    continue
                
                # Take the most recent lookback_periods data points
                X = data[-self.lookback_periods:].reshape(1, -1)
                
                # Scale features
                X_scaled = self.scalers[ticker].transform(X)
                
                # Make prediction
                pred = self.models[ticker].predict(X_scaled)[0]
                predictions[ticker] = pred
                
            except Exception as e:
                logger.error(f"Error forecasting for {ticker}: {e}")
                predictions[ticker] = returns[ticker].mean()
                
        return pd.Series(predictions)
    
    def save_models(self):
        """Save ML models and scalers to disk."""
        for ticker, model in self.models.items():
            try:
                model.save(str(self.model_dir / f"{ticker}_model.joblib"))
            except Exception as e:
                logger.error(f"Error saving model for {ticker}: {e}")
        
        for ticker, scaler in self.scalers.items():
            try:
                joblib.dump(scaler, str(self.model_dir / f"{ticker}_scaler.joblib"))
            except Exception as e:
                logger.error(f"Error saving scaler for {ticker}: {e}")
    
    def load_models(self) -> bool:
        """
        Load ML models and scalers from disk.
        
        Returns:
            bool: True if at least some models loaded successfully, False otherwise
        """
        success = False
        for file in self.model_dir.glob("*_model.joblib"):
            ticker = file.name.split("_model.joblib")[0]
            try:
                model = joblib.load(str(file))
                self.models[ticker] = model
                success = True
            except Exception as e:
                logger.error(f"Error loading model for {ticker}: {e}")
                
        for file in self.model_dir.glob("*_scaler.joblib"):
            ticker = file.name.split("_scaler.joblib")[0]
            try:
                scaler = joblib.load(str(file))
                self.scalers[ticker] = scaler
                success = True
            except Exception as e:
                logger.error(f"Error loading scaler for {ticker}: {e}")
                
        return success


class LSTMForecaster(MLForecaster):
    """LSTM-based return forecaster."""
    
    def __init__(self, model_dir: str = 'models/lstm', lookback_periods: int = 12):
        """
        Initialize LSTM forecaster.
        
        Args:
            model_dir: Directory to save/load trained models
            lookback_periods: Number of historical periods to use for prediction
        """
        super().__init__(model_dir, lookback_periods)
        
    def _build_model(self) -> tf.keras.Model:
        """
        Build LSTM model.
        
        Returns:
            LSTM model instance
        """
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, input_shape=(self.lookback_periods, 1)),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model


class XGBoostForecaster(MLForecaster):
    """XGBoost-based return forecaster."""
    
    def __init__(self, model_dir: str = 'models/xgb', lookback_periods: int = 12, 
                 n_estimators: int = 100, max_depth: int = 3, learning_rate: float = 0.1):
        """
        Initialize XGBoost forecaster.
        
        Args:
            model_dir: Directory to save/load trained models
            lookback_periods: Number of historical periods to use for prediction
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
        """
        super().__init__(model_dir, lookback_periods)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        
    def _build_model(self) -> xgb.XGBRegressor:
        """
        Build XGBoost model.
        
        Returns:
            XGBoost regressor model
        """
        return xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective='reg:squarederror',
            random_state=42
        )


def forecast_returns(data: pd.DataFrame, model_type: str = 'xgboost', 
                    use_cached: bool = True, train_if_missing: bool = True,
                    prediction_date: Optional[Union[str, datetime, pd.Timestamp]] = None):
    """
    Generate return forecasts using the specified model type.
    
    Args:
        data: DataFrame containing returns and other features (volume, etc.)
        model_type: Type of model to use ('lstm' or 'xgboost')
        use_cached: Whether to use cached models if available
        train_if_missing: Whether to train models if not found
        prediction_date: Explicit date for prediction (prevents look-ahead bias)
        
    Returns:
        Series of predicted returns
    """
    # Standardize prediction date handling to enforce time boundaries
    if prediction_date is None:
        prediction_date = datetime.now().date()
        logger.warning(f"No prediction date provided, using current date: {prediction_date}")
    elif isinstance(prediction_date, str):
        prediction_date = pd.to_datetime(prediction_date).date()
    elif isinstance(prediction_date, datetime):
        prediction_date = prediction_date.date()
    elif isinstance(prediction_date, pd.Timestamp):
        prediction_date = prediction_date.date()
        
    # Create appropriate forecaster
    if model_type.lower() == 'lstm':
        forecaster = LSTMForecaster()
    elif model_type.lower() == 'xgboost':
        forecaster = XGBoostForecaster()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'lstm' or 'xgboost'.")
    
    # Try to load saved models
    models_loaded = False
    if use_cached:
        models_loaded = forecaster.load_models()
    
    # Check if we need to train new models
    # Determine training cutoff to be strictly before prediction date
    training_cutoff = pd.Timestamp(prediction_date) - pd.Timedelta(days=1)
    
    # Train if needed
    if not models_loaded and train_if_missing:
        logger.info(f"Training new {model_type} models with data up to {training_cutoff}")
        forecaster.train(data, cutoff_date=training_cutoff)
        forecaster.save_models()
    elif not models_loaded:
        raise ValueError("No saved models found and train_if_missing=False")
    
    # Generate forecasts using data up to prediction date
    logger.info(f"Generating forecasts for {prediction_date} using {model_type}")
    predictions = forecaster.forecast_returns(data, prediction_date=pd.Timestamp(prediction_date))
    
    # Add metadata about the forecast
    if isinstance(predictions, pd.Series):
        predictions.attrs['model_type'] = model_type
        predictions.attrs['prediction_date'] = prediction_date
        predictions.attrs['generated_at'] = datetime.now()
    
    return predictions


if __name__ == "__main__":
    # Simple test code
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ML return forecaster")
    parser.add_argument("--model", choices=["lstm", "xgboost"], default="xgboost")
    parser.add_argument("--train", action="store_true", help="Force training new models")
    args = parser.parse_args()
    
    # Create dummy data for testing
    from portfolio_optimizer import Config, DataManager
    
    cfg = Config("config.yaml")
    dm = DataManager(cfg)
    
    # Fetch some historical returns
    returns = dm.get_returns(
        start=datetime.now() - timedelta(days=365*2),
        end=datetime.now().strftime("%Y-%m-%d")
    )
    
    # Generate forecasts
    preds = forecast_returns(
        returns, 
        model_type=args.model,
        use_cached=not args.train,
        train_if_missing=True
    )
    
    print("Forecasted Returns:")
    print(preds.sort_values(ascending=False).head(10))
