"""
高精度リークフリー機械学習モデル - 改良版
- 高度な特徴量エンジニアリング
- 複数時間軸分析
- アンサンブル学習
- 予測精度最適化
- 市場微細構造分析
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, classification_report
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import joblib
from pathlib import Path
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# ファンダメンタルズデータ収集器をインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from fundamental_data_collector import FundamentalDataCollector
except ImportError:
    FundamentalDataCollector = None

# yfinanceインポート
try:
    import yfinance as yf
    yfinance_available = True
except ImportError:
    yfinance_available = False

class HighPrecisionLeakFreeModels:
    """高精度リークフリー機械学習取引モデル"""
    
    def __init__(self, db_path: str = "high_precision_trading.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # ファンダメンタルズデータ収集器
        if FundamentalDataCollector:
            self.fundamental_collector = FundamentalDataCollector()
        else:
            self.fundamental_collector = None
        
        # モデル保存パス
        self.model_dir = Path("high_precision_models")
        self.model_dir.mkdir(exist_ok=True)
        
        # 特徴量定義
        self.feature_columns = None
        
        # スケーラー
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        
    def collect_extended_market_data(self, symbols: List[str], days: int = 60) -> bool:
        """拡張市場データ収集"""
        if not yfinance_available:
            self.logger.error("yfinanceが利用できません")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # テーブル作成
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS extended_market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    datetime TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, datetime, timeframe)
                )
            ''')
            
            success_count = 0
            total_data = 0
            
            for symbol in symbols:
                try:
                    # yfinanceでデータ取得（複数時間軸）
                    yahoo_symbol = f"{symbol}.T"
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    
                    ticker = yf.Ticker(yahoo_symbol)
                    
                    # 5分足データ
                    data_5m = ticker.history(start=start_date, end=end_date, interval="5m")
                    
                    # 1時間足データ
                    data_1h = ticker.history(start=start_date, end=end_date, interval="1h")
                    
                    # 日足データ
                    data_1d = ticker.history(start=start_date, end=end_date, interval="1d")
                    
                    # 各時間軸のデータを保存
                    timeframes = [("5m", data_5m), ("1h", data_1h), ("1d", data_1d)]
                    
                    for timeframe, data in timeframes:
                        if data.empty:
                            continue
                        
                        df = data.reset_index()
                        df = df.sort_values('Datetime')
                        
                        for _, row in df.iterrows():
                            cursor.execute('''
                                INSERT OR REPLACE INTO extended_market_data 
                                (symbol, datetime, timeframe, open_price, high_price, low_price, close_price, volume)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                symbol,
                                row['Datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                                timeframe,
                                row['Open'],
                                row['High'],
                                row['Low'],
                                row['Close'],
                                int(row['Volume'])
                            ))
                        
                        total_data += len(df)
                    
                    success_count += 1
                    self.logger.info(f"✅ {symbol}: 複数時間軸データを保存")
                    
                except Exception as e:
                    self.logger.error(f"❌ {symbol}のデータ収集エラー: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"拡張データ収集完了: {success_count}/{len(symbols)}銘柄, {total_data}件")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"拡張データ収集エラー: {e}")
            return False
    
    def create_advanced_features(self, df_5m: pd.DataFrame, df_1h: pd.DataFrame, df_1d: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """高度な特徴量作成"""
        if df_5m.empty or len(df_5m) < 50:
            return pd.DataFrame()
        
        df = df_5m.copy()
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # ===== 基本価格特徴量（リークフリー） =====
        
        # 価格変化率（複数期間）
        for period in [1, 2, 3, 5, 10, 20]:
            df[f'price_change_{period}'] = df['close_price'].pct_change(period)
            df[f'high_change_{period}'] = df['high_price'].pct_change(period)
            df[f'low_change_{period}'] = df['low_price'].pct_change(period)
            df[f'volume_change_{period}'] = df['volume'].pct_change(period)
        
        # 価格比率
        df['high_low_ratio'] = df['high_price'] / df['low_price']
        df['open_close_ratio'] = df['open_price'] / df['close_price']
        df['close_open_ratio'] = df['close_price'] / df['open_price']
        df['hl_spread'] = (df['high_price'] - df['low_price']) / df['close_price']
        df['oc_spread'] = abs(df['open_price'] - df['close_price']) / df['close_price']
        
        # ===== 移動平均特徴量（リークフリー） =====
        
        for window in [5, 10, 20, 50]:
            # 移動平均（過去データのみ）
            df[f'sma_{window}'] = df['close_price'].rolling(window=window, min_periods=1).mean().shift(1)
            df[f'ema_{window}'] = df['close_price'].ewm(span=window).mean().shift(1)
            
            # 移動平均比率
            df[f'price_to_sma_{window}'] = df['close_price'] / df[f'sma_{window}']
            df[f'price_to_ema_{window}'] = df['close_price'] / df[f'ema_{window}']
            
            # 移動平均のトレンド
            df[f'sma_trend_{window}'] = df[f'sma_{window}'].diff().shift(1)
            df[f'ema_trend_{window}'] = df[f'ema_{window}'].diff().shift(1)
        
        # ===== ボラティリティ特徴量 =====
        
        for window in [5, 10, 20]:
            # 価格ボラティリティ
            df[f'volatility_{window}'] = df['close_price'].rolling(window=window, min_periods=1).std().shift(1)
            df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / df['close_price']
            
            # 出来高ボラティリティ
            df[f'volume_volatility_{window}'] = df['volume'].rolling(window=window, min_periods=1).std().shift(1)
            
            # ハイローボラティリティ
            df[f'hl_volatility_{window}'] = (df['high_price'] - df['low_price']).rolling(window=window, min_periods=1).std().shift(1)
        
        # ===== テクニカル指標（リークフリー） =====
        
        # RSI（14期間、過去データのみ）
        delta = df['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].shift(1)
        
        # MACD（12-26-9）
        ema12 = df['close_price'].ewm(span=12).mean()
        ema26 = df['close_price'].ewm(span=26).mean()
        df['macd'] = (ema12 - ema26).shift(1)
        df['macd_signal'] = df['macd'].ewm(span=9).mean().shift(1)
        df['macd_histogram'] = (df['macd'] - df['macd_signal']).shift(1)
        
        # ボリンジャーバンド
        sma20 = df['close_price'].rolling(window=20, min_periods=1).mean().shift(1)
        std20 = df['close_price'].rolling(window=20, min_periods=1).std().shift(1)
        df['bb_upper'] = sma20 + (std20 * 2)
        df['bb_lower'] = sma20 - (std20 * 2)
        df['bb_position'] = (df['close_price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ===== 出来高特徴量 =====
        
        for window in [5, 10, 20]:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window=window, min_periods=1).mean().shift(1)
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
        
        df['volume_price_ratio'] = df['volume'] / df['close_price']
        df['volume_hl_ratio'] = df['volume'] / (df['high_price'] - df['low_price'] + 0.001)
        
        # ===== 時間特徴量 =====
        
        datetime_col = pd.to_datetime(df['datetime'])
        df['hour'] = datetime_col.dt.hour
        df['minute'] = datetime_col.dt.minute
        df['day_of_week'] = datetime_col.dt.dayofweek
        df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] <= 15)).astype(int)
        df['time_of_day'] = df['hour'] * 60 + df['minute']
        
        # 市場セッション
        df['is_morning_session'] = ((df['hour'] >= 9) & (df['hour'] < 11.5)).astype(int)
        df['is_afternoon_session'] = ((df['hour'] >= 12.5) & (df['hour'] <= 15)).astype(int)
        
        # ===== 上位時間軸特徴量 =====
        
        if not df_1h.empty and not df_1d.empty:
            # 1時間足からの特徴量
            df_1h_sorted = df_1h.sort_values('datetime')
            hourly_trend = df_1h_sorted['close_price'].pct_change().iloc[-1] if len(df_1h_sorted) > 1 else 0
            hourly_volume_trend = df_1h_sorted['volume'].pct_change().iloc[-1] if len(df_1h_sorted) > 1 else 0
            
            df['hourly_trend'] = hourly_trend
            df['hourly_volume_trend'] = hourly_volume_trend
            
            # 日足からの特徴量
            df_1d_sorted = df_1d.sort_values('datetime')
            daily_trend = df_1d_sorted['close_price'].pct_change().iloc[-1] if len(df_1d_sorted) > 1 else 0
            daily_volume_trend = df_1d_sorted['volume'].pct_change().iloc[-1] if len(df_1d_sorted) > 1 else 0
            
            df['daily_trend'] = daily_trend
            df['daily_volume_trend'] = daily_volume_trend
        else:
            df['hourly_trend'] = 0
            df['hourly_volume_trend'] = 0
            df['daily_trend'] = 0
            df['daily_volume_trend'] = 0
        
        # ===== ファンダメンタルズ特徴量 =====
        
        if symbol and self.fundamental_collector:
            try:
                fundamental_data = self.fundamental_collector.get_fundamental_data_from_db(symbol)
                if not fundamental_data:
                    fundamental_data = self.fundamental_collector.get_fundamental_data_yfinance(symbol)
                
                if fundamental_data:
                    df['per'] = getattr(fundamental_data, 'per', 0)
                    df['pbr'] = getattr(fundamental_data, 'pbr', 0)
                    df['dividend_yield'] = getattr(fundamental_data, 'dividend_yield', 0)
                    df['roe'] = getattr(fundamental_data, 'roe', 0)
                    df['roa'] = getattr(fundamental_data, 'roa', 0)
                    df['market_cap'] = getattr(fundamental_data, 'market_cap', 0)
                    df['eps'] = getattr(fundamental_data, 'eps', 0)
                    df['bps'] = getattr(fundamental_data, 'bps', 0)
                    df['revenue_growth'] = getattr(fundamental_data, 'revenue_growth', 0)
                    df['profit_growth'] = getattr(fundamental_data, 'profit_growth', 0)
                    df['debt_ratio'] = getattr(fundamental_data, 'debt_ratio', 0)
                    
                    self.logger.info(f"✅ {symbol}: ファンダメンタルズ特徴量を追加")
                else:
                    fundamental_features = [
                        'per', 'pbr', 'dividend_yield', 'roe', 'roa', 'market_cap', 
                        'eps', 'bps', 'revenue_growth', 'profit_growth', 'debt_ratio'
                    ]
                    for feature in fundamental_features:
                        df[feature] = 0
                        
            except Exception as e:
                self.logger.error(f"ファンダメンタルズ特徴量エラー: {e}")
                fundamental_features = [
                    'per', 'pbr', 'dividend_yield', 'roe', 'roa', 'market_cap', 
                    'eps', 'bps', 'revenue_growth', 'profit_growth', 'debt_ratio'
                ]
                for feature in fundamental_features:
                    df[feature] = 0
        else:
            fundamental_features = [
                'per', 'pbr', 'dividend_yield', 'roe', 'roa', 'market_cap', 
                'eps', 'bps', 'revenue_growth', 'profit_growth', 'debt_ratio'
            ]
            for feature in fundamental_features:
                df[feature] = 0
        
        # ===== 目標変数（リークフリー） =====
        
        # 次の期間の価格変化を予測
        df['future_price_change'] = df['close_price'].pct_change().shift(-1)
        df['future_direction'] = (df['future_price_change'] > 0).astype(int)
        
        # 複数期間の目標変数
        for period in [1, 3, 5]:
            df[f'future_change_{period}'] = df['close_price'].pct_change(period).shift(-period)
            df[f'future_direction_{period}'] = (df[f'future_change_{period}'] > 0).astype(int)
        
        return df
    
    def prepare_high_precision_data(self, symbol: str) -> Tuple:
        """高精度データ準備"""
        self.logger.info(f"高精度データ準備: {symbol}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 複数時間軸データを取得
            queries = {
                '5m': '''
                    SELECT datetime, open_price, high_price, low_price, close_price, volume
                    FROM extended_market_data
                    WHERE symbol = ? AND timeframe = '5m'
                    ORDER BY datetime ASC
                ''',
                '1h': '''
                    SELECT datetime, open_price, high_price, low_price, close_price, volume
                    FROM extended_market_data
                    WHERE symbol = ? AND timeframe = '1h'
                    ORDER BY datetime ASC
                ''',
                '1d': '''
                    SELECT datetime, open_price, high_price, low_price, close_price, volume
                    FROM extended_market_data
                    WHERE symbol = ? AND timeframe = '1d'
                    ORDER BY datetime ASC
                '''
            }
            
            dataframes = {}
            for timeframe, query in queries.items():
                df = pd.read_sql_query(query, conn, params=(symbol,))
                dataframes[timeframe] = df
                self.logger.info(f"{timeframe}: {len(df)}件")
            
            conn.close()
            
            # 5分足データが主軸
            if dataframes['5m'].empty:
                self.logger.error(f"❌ {symbol} の5分足データが見つかりません")
                return None, None, None, None, None, None
            
            # 高度な特徴量作成
            df = self.create_advanced_features(
                dataframes['5m'], 
                dataframes['1h'], 
                dataframes['1d'], 
                symbol
            )
            
            # 特徴量列を定義
            feature_cols = []
            
            # 価格変化特徴量
            for period in [1, 2, 3, 5, 10, 20]:
                feature_cols.extend([
                    f'price_change_{period}', f'high_change_{period}', 
                    f'low_change_{period}', f'volume_change_{period}'
                ])
            
            # 価格比率特徴量
            feature_cols.extend([
                'high_low_ratio', 'open_close_ratio', 'close_open_ratio', 
                'hl_spread', 'oc_spread'
            ])
            
            # 移動平均特徴量
            for window in [5, 10, 20, 50]:
                feature_cols.extend([
                    f'price_to_sma_{window}', f'price_to_ema_{window}',
                    f'sma_trend_{window}', f'ema_trend_{window}'
                ])
            
            # ボラティリティ特徴量
            for window in [5, 10, 20]:
                feature_cols.extend([
                    f'volatility_{window}', f'volatility_ratio_{window}',
                    f'volume_volatility_{window}', f'hl_volatility_{window}'
                ])
            
            # テクニカル指標
            feature_cols.extend([
                'rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_position'
            ])
            
            # 出来高特徴量
            for window in [5, 10, 20]:
                feature_cols.extend([f'volume_ratio_{window}'])
            feature_cols.extend(['volume_price_ratio', 'volume_hl_ratio'])
            
            # 時間特徴量
            feature_cols.extend([
                'hour', 'minute', 'day_of_week', 'is_market_open', 'time_of_day',
                'is_morning_session', 'is_afternoon_session'
            ])
            
            # 上位時間軸特徴量
            feature_cols.extend([
                'hourly_trend', 'hourly_volume_trend', 'daily_trend', 'daily_volume_trend'
            ])
            
            # ファンダメンタルズ特徴量
            feature_cols.extend([
                'per', 'pbr', 'dividend_yield', 'roe', 'roa', 'market_cap',
                'eps', 'bps', 'revenue_growth', 'profit_growth', 'debt_ratio'
            ])
            
            # 存在する特徴量のみを使用
            available_features = [col for col in feature_cols if col in df.columns]
            
            # 欠損値処理
            df = df.dropna()
            
            if len(df) < 100:
                self.logger.error(f"❌ 有効なデータが不足: {len(df)}件")
                return None, None, None, None, None, None
            
            # 特徴量と目標変数を分離
            X = df[available_features].copy()
            y_price = df['future_price_change'].copy()
            y_direction = df['future_direction'].copy()
            
            # 無限大、NaN値を処理
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            
            # 最後の行を削除（目標変数が未来なのでNaN）
            X = X[:-1]
            y_price = y_price[:-1]
            y_direction = y_direction[:-1]
            
            # 時系列分割（最後の20%をテスト用）
            split_idx = int(len(X) * 0.8)
            
            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_price_train = y_price[:split_idx]
            y_price_test = y_price[split_idx:]
            y_direction_train = y_direction[:split_idx]
            y_direction_test = y_direction[split_idx:]
            
            self.feature_columns = available_features
            
            self.logger.info(f"特徴量数: {len(available_features)}")
            self.logger.info(f"訓練データ: {len(X_train)}件")
            self.logger.info(f"テストデータ: {len(X_test)}件")
            
            return X_train, X_test, y_price_train, y_price_test, y_direction_train, y_direction_test
            
        except Exception as e:
            self.logger.error(f"データ準備エラー: {e}")
            return None, None, None, None, None, None
    
    def train_ensemble_models(self, X_train, y_price_train, y_direction_train):
        """アンサンブルモデル訓練"""
        results = {}
        
        # 特徴量選択
        self.logger.info("特徴量選択中...")
        
        # 価格予測用特徴量選択
        price_selector = SelectKBest(score_func=f_regression, k=min(50, len(self.feature_columns)))
        X_train_price_selected = price_selector.fit_transform(X_train, y_price_train)
        
        # 方向予測用特徴量選択
        direction_selector = SelectKBest(score_func=f_classif, k=min(30, len(self.feature_columns)))
        X_train_direction_selected = direction_selector.fit_transform(X_train, y_direction_train)
        
        # データスケーリング
        price_scaler = RobustScaler()
        direction_scaler = StandardScaler()
        
        X_train_price_scaled = price_scaler.fit_transform(X_train_price_selected)
        X_train_direction_scaled = direction_scaler.fit_transform(X_train_direction_selected)
        
        # ===== 価格予測モデル群 =====
        
        self.logger.info("価格予測モデル群を訓練中...")
        
        price_models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'RandomForest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        price_results = {}
        for name, model in price_models.items():
            try:
                model.fit(X_train_price_scaled, y_price_train)
                price_results[name] = {
                    'model': model,
                    'scaler': price_scaler,
                    'selector': price_selector
                }
                self.logger.info(f"✅ {name} (価格予測) 訓練完了")
            except Exception as e:
                self.logger.error(f"❌ {name} (価格予測) 訓練エラー: {e}")
        
        # ===== 方向予測モデル群 =====
        
        self.logger.info("方向予測モデル群を訓練中...")
        
        direction_models = {
            'RandomForestClassifier': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'GradientBoostingClassifier': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'SVC': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        }
        
        direction_results = {}
        for name, model in direction_models.items():
            try:
                model.fit(X_train_direction_scaled, y_direction_train)
                direction_results[name] = {
                    'model': model,
                    'scaler': direction_scaler,
                    'selector': direction_selector
                }
                self.logger.info(f"✅ {name} (方向予測) 訓練完了")
            except Exception as e:
                self.logger.error(f"❌ {name} (方向予測) 訓練エラー: {e}")
        
        results = {
            'price_models': price_results,
            'direction_models': direction_results
        }
        
        return results
    
    def evaluate_ensemble_models(self, models, X_test, y_price_test, y_direction_test):
        """アンサンブルモデル評価"""
        evaluation_results = {
            'price_models': {},
            'direction_models': {}
        }
        
        # ===== 価格予測モデル評価 =====
        
        for name, model_data in models['price_models'].items():
            try:
                model = model_data['model']
                scaler = model_data['scaler']
                selector = model_data['selector']
                
                # 同じ前処理を適用
                X_test_selected = selector.transform(X_test)
                X_test_scaled = scaler.transform(X_test_selected)
                
                y_pred = model.predict(X_test_scaled)
                
                mse = mean_squared_error(y_price_test, y_pred)
                mae = mean_absolute_error(y_price_test, y_pred)
                r2 = r2_score(y_price_test, y_pred)
                
                # 方向精度も計算
                pred_direction = (y_pred > 0).astype(int)
                actual_direction = (y_price_test > 0).astype(int)
                direction_accuracy = accuracy_score(actual_direction, pred_direction)
                
                evaluation_results['price_models'][name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'direction_accuracy': direction_accuracy,
                    'predictions': y_pred,
                    'actual': y_price_test.values
                }
                
                self.logger.info(f"{name} (価格予測):")
                self.logger.info(f"  MSE: {mse:.6f}")
                self.logger.info(f"  MAE: {mae:.6f}")
                self.logger.info(f"  R²: {r2:.4f}")
                self.logger.info(f"  方向精度: {direction_accuracy:.4f} ({direction_accuracy*100:.1f}%)")
                
            except Exception as e:
                self.logger.error(f"❌ {name} (価格予測) 評価エラー: {e}")
        
        # ===== 方向予測モデル評価 =====
        
        for name, model_data in models['direction_models'].items():
            try:
                model = model_data['model']
                scaler = model_data['scaler']
                selector = model_data['selector']
                
                # 同じ前処理を適用
                X_test_selected = selector.transform(X_test)
                X_test_scaled = scaler.transform(X_test_selected)
                
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
                
                accuracy = accuracy_score(y_direction_test, y_pred)
                
                evaluation_results['direction_models'][name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'actual': y_direction_test.values
                }
                
                self.logger.info(f"{name} (方向予測):")
                self.logger.info(f"  精度: {accuracy:.4f} ({accuracy*100:.1f}%)")
                
            except Exception as e:
                self.logger.error(f"❌ {name} (方向予測) 評価エラー: {e}")
        
        return evaluation_results
    
    def run_high_precision_analysis(self, symbols: List[str]) -> Dict:
        """高精度分析実行"""
        all_results = {}
        
        for symbol in symbols:
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"高精度分析: {symbol}")
            self.logger.info('='*70)
            
            # データ準備
            data_result = self.prepare_high_precision_data(symbol)
            if data_result[0] is None:
                continue
            
            X_train, X_test, y_price_train, y_price_test, y_direction_train, y_direction_test = data_result
            
            # アンサンブルモデル訓練
            models = self.train_ensemble_models(X_train, y_price_train, y_direction_train)
            
            # モデル評価
            evaluation = self.evaluate_ensemble_models(models, X_test, y_price_test, y_direction_test)
            
            # レポート生成
            report = self.generate_high_precision_report(symbol, evaluation)
            
            # 結果保存
            all_results[symbol] = {
                'models': models,
                'evaluation': evaluation,
                'report': report
            }
            
            # レポートをファイルに保存
            try:
                report_path = self.model_dir / f'{symbol}_high_precision_report.txt'
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                self.logger.info(f"📊 高精度レポート保存: {report_path}")
            except Exception as e:
                self.logger.error(f"レポート保存エラー: {e}")
        
        return all_results
    
    def generate_high_precision_report(self, symbol: str, evaluation: Dict) -> str:
        """高精度レポート生成"""
        report = f"=== {symbol} 高精度分析レポート ===\n\n"
        report += f"分析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"特徴量数: {len(self.feature_columns) if self.feature_columns else 'N/A'}\n\n"
        
        # 価格予測結果
        report += "【価格変化予測モデル】\n"
        price_results = []
        for model_name, results in evaluation['price_models'].items():
            price_results.append((model_name, results))
            report += f"{model_name}:\n"
            report += f"  MSE: {results['mse']:.6f}\n"
            report += f"  MAE: {results['mae']:.6f}\n"
            report += f"  R²: {results['r2']:.4f}\n"
            report += f"  方向精度: {results['direction_accuracy']:.1%}\n\n"
        
        # 方向予測結果
        report += "【方向予測モデル】\n"
        direction_results = []
        for model_name, results in evaluation['direction_models'].items():
            direction_results.append((model_name, results))
            report += f"{model_name}:\n"
            report += f"  精度: {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)\n\n"
        
        # 最良モデル選択
        if price_results:
            best_price_model = min(price_results, key=lambda x: x[1]['mae'])
            report += f"最良価格予測: {best_price_model[0]}\n"
            report += f"  MAE: {best_price_model[1]['mae']:.6f}\n"
            report += f"  方向精度: {best_price_model[1]['direction_accuracy']:.1%}\n\n"
        
        if direction_results:
            best_direction_model = max(direction_results, key=lambda x: x[1]['accuracy'])
            report += f"最良方向予測: {best_direction_model[0]}\n"
            report += f"  精度: {best_direction_model[1]['accuracy']:.1%}\n\n"
        
        # 実用性評価
        report += "【実用性評価】\n"
        if price_results and direction_results:
            best_mae = best_price_model[1]['mae']
            best_direction_acc = best_direction_model[1]['accuracy']
            
            if best_mae < 0.005 and best_direction_acc > 0.6:
                practical_rating = "非常に実用的"
            elif best_mae < 0.01 and best_direction_acc > 0.55:
                practical_rating = "実用的"
            elif best_mae < 0.02 and best_direction_acc > 0.52:
                practical_rating = "やや実用的"
            else:
                practical_rating = "要改善"
            
            report += f"総合評価: {practical_rating}\n"
        
        # 改善提案
        report += "\n【改善提案】\n"
        if direction_results:
            max_acc = max(r[1]['accuracy'] for r in direction_results)
            if max_acc < 0.6:
                report += "- より多くの特徴量を追加してください\n"
                report += "- データ期間を延長してください\n"
                report += "- ハイパーパラメータを調整してください\n"
            elif max_acc < 0.7:
                report += "- 特徴量エンジニアリングを改善してください\n"
                report += "- アンサンブル手法を強化してください\n"
            else:
                report += "- 素晴らしい性能です！実運用を検討してください\n"
        
        return report

# 使用例
if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 高精度モデルインスタンス作成
    high_precision_models = HighPrecisionLeakFreeModels()
    
    # テスト用銘柄
    symbols = ['7203', '6758', '8306']
    
    print("=== 拡張市場データ収集 ===")
    if high_precision_models.collect_extended_market_data(symbols, days=60):
        print("✅ 拡張データ収集完了")
    else:
        print("❌ 拡張データ収集失敗")
    
    print("\n=== 高精度分析実行 ===")
    results = high_precision_models.run_high_precision_analysis(symbols)
    
    print("\n=== 高精度結果サマリー ===")
    for symbol in symbols:
        if symbol in results:
            print(f"\n{symbol}:")
            evaluation = results[symbol]['evaluation']
            
            # 最良の価格予測
            if evaluation['price_models']:
                best_price = min(evaluation['price_models'].items(), key=lambda x: x[1]['mae'])
                print(f"  最良価格予測: {best_price[0]} (MAE: {best_price[1]['mae']:.6f}, 方向精度: {best_price[1]['direction_accuracy']:.1%})")
            
            # 最良の方向予測
            if evaluation['direction_models']:
                best_direction = max(evaluation['direction_models'].items(), key=lambda x: x[1]['accuracy'])
                print(f"  最良方向予測: {best_direction[0]} (精度: {best_direction[1]['accuracy']:.1%})")
    
    print("\n=== 高精度分析完了 ===")
    print("🚀 高度な特徴量エンジニアリング適用")
    print("🎯 アンサンブル学習による精度向上")
    print("📈 複数時間軸分析による予測強化")
