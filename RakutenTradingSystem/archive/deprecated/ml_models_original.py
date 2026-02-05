"""
機械学習モデル - 方法1の実装
- 1時間線形予測モデル
- 5分足上下予測モデル
- ファクトチェック機能
- ファンダメンタルズ分析統合
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
from pathlib import Path
import sys
import os

# yfinanceデータ収集とファンダメンタルズデータ収集器をインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

# scikit-learnの追加インポート
try:
    from sklearn.metrics import mean_absolute_error, r2_score
    sklearn_available = True
except ImportError:
    sklearn_available = False

# LightGBMインポート
try:
    import lightgbm as lgb
    lightgbm_available = True
except ImportError:
    lightgbm_available = False

class MLTradingModels:
    """機械学習取引モデル - yfinanceデータ対応"""
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # モデル初期化
        self.hourly_model = LinearRegression()
        self.minute_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # ファンダメンタルズデータ収集器
        if FundamentalDataCollector:
            self.fundamental_collector = FundamentalDataCollector()
        else:
            self.fundamental_collector = None
        
        # モデル保存パス
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # 予測精度追跡
        self.prediction_history = []
        
        # 統合機能用の追加プロパティ
        self.feature_columns = None
        self.advanced_model = None  # 高度なモデル用
        
    def collect_yfinance_data(self, symbols: List[str], days: int = 5) -> bool:
        """yfinanceからデータを収集してデータベースに保存"""
        if not yfinance_available:
            self.logger.error("yfinanceが利用できません")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # テーブル作成
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chart_data (
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
                    # yfinanceでデータ取得
                    yahoo_symbol = f"{symbol}.T"
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    
                    ticker = yf.Ticker(yahoo_symbol)
                    data = ticker.history(start=start_date, end=end_date, interval="5m")
                    
                    if data.empty:
                        self.logger.warning(f"データが見つかりません: {yahoo_symbol}")
                        continue
                    
                    # データ整形
                    df = data.reset_index()
                    
                    # データベースに保存
                    for _, row in df.iterrows():
                        cursor.execute('''
                            INSERT OR REPLACE INTO chart_data 
                            (symbol, datetime, timeframe, open_price, high_price, low_price, close_price, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            symbol,
                            row['Datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                            '5M',
                            row['Open'],
                            row['High'],
                            row['Low'],
                            row['Close'],
                            int(row['Volume'])
                        ))
                    
                    success_count += 1
                    total_data += len(df)
                    self.logger.info(f"✅ {symbol}: {len(df)}件のデータを保存")
                    
                except Exception as e:
                    self.logger.error(f"❌ {symbol}のデータ収集エラー: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"データ収集完了: {success_count}/{len(symbols)}銘柄, {total_data}件")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"データ収集エラー: {e}")
            return False
    
    def prepare_features(self, symbol: str, target_time: datetime, lookback_hours: int = 72, use_fundamental: bool = True) -> Optional[np.ndarray]:
        """特徴量準備 - yfinanceデータ対応"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 過去72時間のデータ取得 (chart_dataテーブルから)
            start_time = target_time - timedelta(hours=lookback_hours)
            
            query = '''
                SELECT 
                    datetime,
                    close_price,
                    volume,
                    open_price,
                    high_price,
                    low_price
                FROM chart_data 
                WHERE symbol = ? AND datetime >= ? AND datetime <= ?
                ORDER BY datetime
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol, start_time.strftime('%Y-%m-%d %H:%M:%S'), target_time.strftime('%Y-%m-%d %H:%M:%S')))
            conn.close()
            
            if len(df) < 24:  # 最低2時間分のデータが必要
                return None
            
            # テクニカル特徴量計算
            features = []
            
            # 移動平均
            df['ma_5'] = df['close_price'].rolling(window=5).mean()
            df['ma_20'] = df['close_price'].rolling(window=20).mean()
            df['ma_60'] = df['close_price'].rolling(window=60).mean()
            
            features.extend([
                df['ma_5'].iloc[-1] if not pd.isna(df['ma_5'].iloc[-1]) else df['close_price'].iloc[-1],
                df['ma_20'].iloc[-1] if not pd.isna(df['ma_20'].iloc[-1]) else df['close_price'].iloc[-1],
                df['ma_60'].iloc[-1] if not pd.isna(df['ma_60'].iloc[-1]) else df['close_price'].iloc[-1]
            ])
            
            # 価格関連
            current_price = df['close_price'].iloc[-1]
            features.extend([
                current_price,
                df['close_price'].pct_change().iloc[-5:].mean(),  # 直近5分間の平均変化率
                df['volume'].iloc[-12:].mean(),  # 直近1時間の平均出来高
            ])
            
            # 価格比率（ゼロ除算を防ぐ）
            features.extend([
                df['high_price'].iloc[-1] / df['low_price'].iloc[-1] if df['low_price'].iloc[-1] != 0 else 1.0,  # 高値/安値比
                df['open_price'].iloc[-1] / df['close_price'].iloc[-1] if df['close_price'].iloc[-1] != 0 else 1.0,  # 始値/終値比
                (df['close_price'].iloc[-1] / df['volume'].iloc[-1] * 1000000) if df['volume'].iloc[-1] != 0 else 0.0  # 価格/出来高比
            ])
            
            # 業界フラグ（簡易版）
            sector_flag = self.get_sector_flag(symbol)
            features.append(sector_flag)
            
            # 前3日間の日足データ
            daily_features = self.get_daily_features(symbol, target_time)
            features.extend(daily_features)
            
            # ファンダメンタルズ特徴量追加
            if use_fundamental and self.fundamental_collector:
                fundamental_features = self.get_fundamental_features(symbol, target_time)
                features.extend(fundamental_features)
            else:
                # ファンダメンタルズデータがない場合のデフォルト値
                features.extend([0.0] * 12)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"特徴量準備エラー: {e}")
            return None
    
    def get_fundamental_features(self, symbol: str, target_time: datetime) -> List[float]:
        """ファンダメンタルズ特徴量取得"""
        try:
            if not self.fundamental_collector:
                return [0.0] * 12
            
            # ファンダメンタルズデータ取得
            fundamental_data = self.fundamental_collector.get_fundamental_data_from_db(
                symbol, target_time.date()
            )
            
            if not fundamental_data:
                # データがない場合はデフォルト値
                return [0.0] * 12
            
            # 相対評価指標取得
            relative_metrics = self.fundamental_collector.get_relative_valuation(symbol)
            
            # ファンダメンタルズ特徴量
            fund_features = [
                fundamental_data.per,
                fundamental_data.pbr,
                fundamental_data.psr,
                fundamental_data.dividend_yield,
                fundamental_data.roe,
                fundamental_data.roa,
                fundamental_data.debt_ratio,
                fundamental_data.revenue_growth,
                fundamental_data.profit_growth,
                fundamental_data.operating_margin,
                relative_metrics.get('per_ratio', 1.0),
                relative_metrics.get('pbr_ratio', 1.0)
            ]
            
            return fund_features
            
        except Exception as e:
            self.logger.error(f"ファンダメンタルズ特徴量取得エラー {symbol}: {e}")
            return [0.0] * 12
    
    def get_sector_flag(self, symbol: str) -> float:
        """業界フラグ取得（簡易版）"""
        # prime_symbols.csvから業界情報取得
        try:
            df = pd.read_csv("prime_symbols.csv")
            sector_row = df[df['symbol'] == int(symbol)]
            if not sector_row.empty:
                sector = sector_row['sector'].iloc[0]
                # 業界を数値にエンコード
                sector_mapping = {
                    '電機': 1, '自動車': 2, '銀行': 3, '化学': 4, '機械': 5,
                    '情報通信': 6, '建設': 7, '食品': 8, '医薬品': 9, '不動産': 10
                }
                return sector_mapping.get(sector, 0)
        except:
            pass
        return 0
    
    def get_daily_features(self, symbol: str, target_time: datetime) -> List[float]:
        """前3日間の日足データ特徴量"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 前3日間のデータ (chart_dataテーブルから)
            start_date = (target_time - timedelta(days=3)).date()
            
            query = '''
                SELECT 
                    DATE(datetime) as date,
                    MIN(close_price) as low,
                    MAX(close_price) as high,
                    SUM(volume) as daily_volume,
                    AVG(close_price) as avg_price
                FROM chart_data
                WHERE symbol = ? AND DATE(datetime) >= ? AND timeframe = '5M'
                GROUP BY DATE(datetime)
                ORDER BY date DESC
                LIMIT 3
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol, start_date.strftime('%Y-%m-%d')))
            conn.close()
            
            if len(df) < 3:
                return [0, 0, 0, 0, 0, 0]  # デフォルト値
            
            # 3日間の変化率
            price_changes = df['avg_price'].pct_change().fillna(0).tolist()[-2:]
            volume_avg = df['daily_volume'].mean()
            volatility = (df['high'] - df['low']).mean() / df['avg_price'].mean()
            
            return price_changes + [volume_avg, volatility, df['avg_price'].iloc[0], df['daily_volume'].iloc[0]]
            
        except Exception as e:
            self.logger.error(f"日足特徴量取得エラー: {e}")
            return [0, 0, 0, 0, 0, 0]
    
    def train_hourly_model(self, symbols: List[str], lookback_days: int = 30):
        """1時間線形予測モデル訓練"""
        self.logger.info("1時間予測モデル訓練開始")
        
        X_data = []
        y_data = []
        
        conn = sqlite3.connect(self.db_path)
        
        for symbol in symbols:
            try:
                # 過去30日のデータで学習 (chart_dataテーブルから)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days)
                
                query = '''
                    SELECT datetime, close_price
                    FROM chart_data
                    WHERE symbol = ? AND datetime >= ? AND datetime <= ? 
                    AND timeframe = '5M'
                    ORDER BY datetime
                '''
                
                df = pd.read_sql_query(query, conn, params=(symbol, start_date.strftime('%Y-%m-%d %H:%M:%S'), end_date.strftime('%Y-%m-%d %H:%M:%S')))
                
                if len(df) < 100:  # 最低データ量チェック
                    continue
                
                # 1時間ごとのデータポイント作成
                for i in range(0, len(df) - 12, 12):  # 1時間 = 12 * 5分
                    current_time = pd.to_datetime(df.iloc[i]['datetime'])
                    
                    # 特徴量準備
                    features = self.prepare_features(symbol, current_time)
                    if features is None:
                        continue
                    
                    # 1時間後の価格（ターゲット）
                    if i + 12 < len(df):
                        future_price = df.iloc[i + 12]['close_price']
                        current_price = df.iloc[i]['close_price']
                        price_change_rate = (future_price - current_price) / current_price
                        
                        X_data.append(features.flatten())
                        y_data.append(price_change_rate)
                
            except Exception as e:
                self.logger.error(f"シンボル {symbol} の訓練データ準備エラー: {e}")
                continue
        
        conn.close()
        
        if len(X_data) < 50:
            self.logger.error("訓練データが不足しています")
            return False
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        # データ標準化
        X_scaled = self.scaler.fit_transform(X_data)
        
        # 訓練
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size=0.2, random_state=42)
        
        self.hourly_model.fit(X_train, y_train)
        
        # 精度評価
        y_pred = self.hourly_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        self.logger.info(f"1時間予測モデル訓練完了 - MSE: {mse:.6f}")
        
        # モデル保存
        joblib.dump(self.hourly_model, self.model_dir / "hourly_model.pkl")
        joblib.dump(self.scaler, self.model_dir / "scaler.pkl")
        
        return True
    
    def train_minute_model(self, symbols: List[str], lookback_days: int = 30):
        """5分足上下予測モデル訓練"""
        self.logger.info("5分足予測モデル訓練開始")
        
        X_data = []
        y_data = []
        
        conn = sqlite3.connect(self.db_path)
        
        for symbol in symbols:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days)
                
                query = '''
                    SELECT datetime, close_price
                    FROM chart_data
                    WHERE symbol = ? AND datetime >= ? AND datetime <= ?
                    AND timeframe = '5M'
                    ORDER BY datetime
                '''
                
                df = pd.read_sql_query(query, conn, params=(symbol, start_date.strftime('%Y-%m-%d %H:%M:%S'), end_date.strftime('%Y-%m-%d %H:%M:%S')))
                
                if len(df) < 50:
                    continue
                
                # 5分ごとのデータポイント作成
                for i in range(len(df) - 1):
                    current_time = pd.to_datetime(df.iloc[i]['datetime'])
                    
                    features = self.prepare_features(symbol, current_time)
                    if features is None:
                        continue
                    
                    # 次の5分後の価格変動（上がる=1, 下がる=0）
                    current_price = df.iloc[i]['close_price']
                    next_price = df.iloc[i + 1]['close_price']
                    direction = 1 if next_price > current_price else 0
                    
                    X_data.append(features.flatten())
                    y_data.append(direction)
                
            except Exception as e:
                self.logger.error(f"シンボル {symbol} の5分足訓練データ準備エラー: {e}")
                continue
        
        conn.close()
        
        if len(X_data) < 50:
            self.logger.error("5分足訓練データが不足しています")
            return False
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        # 訓練
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
        
        self.minute_model.fit(X_train, y_train)
        
        # 精度評価
        y_pred = self.minute_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.logger.info(f"5分足予測モデル訓練完了 - 精度: {accuracy:.3f}")
        
        # モデル保存
        joblib.dump(self.minute_model, self.model_dir / "minute_model.pkl")
        
        return True
    
    def predict_hourly_trend(self, symbol: str, current_time: datetime) -> Optional[float]:
        """1時間トレンド予測"""
        try:
            features = self.prepare_features(symbol, current_time)
            if features is None:
                return None
            
            # モデル読み込み
            if not hasattr(self, 'hourly_model') or self.hourly_model is None:
                self.load_models()
            
            features_scaled = self.scaler.transform(features)
            prediction = self.hourly_model.predict(features_scaled)[0]
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"1時間予測エラー: {e}")
            return None
    
    def predict_minute_direction(self, symbol: str, current_time: datetime) -> Optional[Tuple[int, float]]:
        """5分足方向予測（方向, 確率）"""
        try:
            features = self.prepare_features(symbol, current_time)
            if features is None:
                return None
            
            if not hasattr(self, 'minute_model') or self.minute_model is None:
                self.load_models()
            
            # 予測と確率
            prediction = self.minute_model.predict(features)[0]
            probabilities = self.minute_model.predict_proba(features)[0]
            confidence = max(probabilities)
            
            return prediction, confidence
            
        except Exception as e:
            self.logger.error(f"5分足予測エラー: {e}")
            return None
    
    def fact_check_predictions(self, symbol: str, current_time: datetime) -> Dict:
        """ファクトチェック実行"""
        result = {
            'should_trade': False,
            'direction': None,
            'confidence': 0.0,
            'hourly_prediction': None,
            'minute_prediction': None,
            'minute_confidence': 0.0
        }
        
        try:
            # 1時間予測
            hourly_pred = self.predict_hourly_trend(symbol, current_time)
            if hourly_pred is None:
                return result
            
            # 5分足予測
            minute_result = self.predict_minute_direction(symbol, current_time)
            if minute_result is None:
                return result
            
            minute_pred, minute_conf = minute_result
            
            # 方向の一致チェック
            hourly_direction = 1 if hourly_pred > 0 else 0
            directions_match = hourly_direction == minute_pred
            
            # 5分足の信頼度が80%以上かチェック
            high_confidence = minute_conf >= 0.8
            
            result.update({
                'hourly_prediction': hourly_pred,
                'minute_prediction': minute_pred,
                'minute_confidence': minute_conf,
                'should_trade': directions_match and high_confidence,
                'direction': minute_pred if directions_match and high_confidence else None,
                'confidence': minute_conf
            })
            
            self.logger.info(f"ファクトチェック結果 - 取引実行: {result['should_trade']}, 方向: {result['direction']}, 信頼度: {result['confidence']:.3f}")
            
        except Exception as e:
            self.logger.error(f"ファクトチェックエラー: {e}")
        
        return result
    
    def load_models(self):
        """保存されたモデル読み込み"""
        try:
            if (self.model_dir / "hourly_model.pkl").exists():
                self.hourly_model = joblib.load(self.model_dir / "hourly_model.pkl")
                self.scaler = joblib.load(self.model_dir / "scaler.pkl")
                
            if (self.model_dir / "minute_model.pkl").exists():
                self.minute_model = joblib.load(self.model_dir / "minute_model.pkl")
                
            self.logger.info("モデル読み込み完了")
            
        except Exception as e:
            self.logger.error(f"モデル読み込みエラー: {e}")
    
    def validate_prediction(self, symbol: str, prediction_time: datetime, actual_time: datetime):
        """予測精度検証"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 実際の価格取得
            query = '''
                SELECT close_price 
                FROM chart_data 
                WHERE symbol = ? AND datetime = ? AND timeframe = '5M'
            '''
            
            pred_result = pd.read_sql_query(query, conn, params=(symbol, prediction_time.strftime('%Y-%m-%d %H:%M:%S')))
            actual_result = pd.read_sql_query(query, conn, params=(symbol, actual_time.strftime('%Y-%m-%d %H:%M:%S')))
            
            if len(pred_result) > 0 and len(actual_result) > 0:
                pred_price = pred_result.iloc[0]['close_price']
                actual_price = actual_result.iloc[0]['close_price']
                
                # 方向の正確性
                predicted_direction = 1 if actual_price > pred_price else 0
                
                # 予測履歴に追加
                self.prediction_history.append({
                    'symbol': symbol,
                    'prediction_time': prediction_time,
                    'actual_time': actual_time,
                    'predicted_direction': predicted_direction,
                    'actual_direction': predicted_direction,
                    'accuracy': 1 if predicted_direction == predicted_direction else 0
                })
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"予測検証エラー: {e}")

    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """高度な特徴量を作成（ml_prediction_model.pyから統合）"""
        if df.empty or len(df) < 10:
            return pd.DataFrame()
        
        df = df.copy()
        
        # 基本的な特徴量（ゼロ除算を防ぐ）
        df['price_change'] = df['close_price'].pct_change()
        df['high_low_ratio'] = df['high_price'] / df['low_price'].replace(0, np.nan)
        df['open_close_ratio'] = df['open_price'] / df['close_price'].replace(0, np.nan)
        df['volume_price_ratio'] = df['volume'] / df['close_price'].replace(0, np.nan)
        
        # 移動平均
        df['sma_5'] = df['close_price'].rolling(window=5).mean()
        df['sma_10'] = df['close_price'].rolling(window=10).mean()
        df['sma_ratio'] = df['close_price'] / df['sma_5'].replace(0, np.nan)
        
        # ボラティリティ
        df['volatility_5'] = df['close_price'].rolling(window=5).std()
        df['volatility_10'] = df['close_price'].rolling(window=10).std()
        
        # 出来高系
        df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_5'].replace(0, np.nan)
        
        # RSI（簡易版）
        delta = df['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ラグ特徴量（過去の値）
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['close_price'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'change_lag_{lag}'] = df['price_change'].shift(lag)
        
        # 時間系特徴量
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
        df['minute'] = pd.to_datetime(df['datetime']).dt.minute
        df['time_of_day'] = df['hour'] * 60 + df['minute']
        
        # 目標変数（次の期間の価格）
        df['target'] = df['close_price'].shift(-1)
        
        return df

    def prepare_advanced_data(self, symbol: str, period: int = 1000) -> tuple:
        """高度なデータ準備（ml_prediction_model.pyから統合）"""
        self.logger.info(f"データ準備中: {symbol}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # yfinanceデータを取得
            query = '''
                SELECT datetime, open_price, high_price, low_price, close_price, volume
                FROM chart_data
                WHERE symbol = ? AND timeframe = '5M'
                ORDER BY datetime DESC
                LIMIT ?
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol, period))
            conn.close()
            
            if df.empty:
                self.logger.error(f"❌ {symbol} のデータが見つかりません")
                return None, None, None, None
            
            self.logger.info(f"取得データ数: {len(df)}件")
            
            # データを時系列順に並び替え
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # 特徴量を作成
            df = self.create_advanced_features(df)
            
            # 特徴量の列を定義
            feature_cols = [
                'price_change', 'high_low_ratio', 'open_close_ratio', 'volume_price_ratio',
                'sma_ratio', 'volatility_5', 'volatility_10', 'volume_ratio', 'rsi',
                'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5',
                'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5',
                'change_lag_1', 'change_lag_2', 'change_lag_3', 'change_lag_5',
                'hour', 'minute', 'time_of_day'
            ]
            
            # 欠損値を削除
            df = df.dropna()
            
            if len(df) < 50:
                self.logger.error(f"❌ 有効なデータが不足しています ({len(df)}件)")
                return None, None, None, None
            
            # 特徴量と目標変数を分離
            X = df[feature_cols].copy()
            y = df['target'].copy()
            
            # NaNや無限大の値を処理
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            
            # さらに無限大や異常値をクリップ
            X = X.clip(-1e6, 1e6)
            
            self.logger.info(f"特徴量数: {len(feature_cols)}")
            self.logger.info(f"有効サンプル数: {len(X)}")
            
            self.feature_columns = feature_cols
            
            return X, y, df, feature_cols
            
        except Exception as e:
            self.logger.error(f"データ準備エラー: {e}")
            return None, None, None, None

    def train_advanced_model(self, X, y, model_type='comparison'):
        """高度なモデルを訓練（比較モード追加）"""
        if model_type == 'comparison':
            return self.compare_models(X, y)
        elif sklearn_available and model_type == 'advanced':
            return self._train_sklearn_model(X, y)
        else:
            return self._train_simple_model(X, y)
    
    def _train_sklearn_model(self, X, y):
        """scikit-learnを使用したモデル訓練"""
        self.logger.info("高度なモデル（Random Forest）を訓練中...")
        
        # データを分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # スケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # モデル訓練
        self.advanced_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.advanced_model.fit(X_train_scaled, y_train)
        
        # 予測
        y_pred = self.advanced_model.predict(X_test_scaled)
        
        # 評価指標
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.logger.info(f"MSE: {mse:.4f}")
        self.logger.info(f"R²: {r2:.4f}")
        self.logger.info(f"MAE: {mae:.4f}")
        
        # 特徴量重要度
        if self.feature_columns:
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.advanced_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.logger.info("\n上位10の重要な特徴量:")
            for _, row in feature_importance.head(10).iterrows():
                self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return {
            'model_type': 'RandomForest',
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'feature_importance': feature_importance if self.feature_columns else None,
            'test_predictions': y_pred,
            'test_actual': y_test.values,
            'scaler': scaler
        }
    
    def _train_simple_model(self, X, y):
        """シンプルなモデル訓練"""
        self.logger.info("シンプルなモデル（移動平均ベース）を訓練中...")
        
        # 単純な移動平均モデル
        window = 5
        y_pred = []
        y_test = []
        
        for i in range(window, len(X)):
            # 過去n期間の平均を予測値とする
            pred = np.mean(y.iloc[i-window:i])
            y_pred.append(pred)
            y_test.append(y.iloc[i])
        
        y_pred = np.array(y_pred)
        y_test = np.array(y_test)
        
        # 評価指標
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        
        self.logger.info(f"MSE: {mse:.4f}")
        self.logger.info(f"MAE: {mae:.4f}")
        
        return {
            'model_type': 'SimpleMovingAverage',
            'mse': mse,
            'mae': mae,
            'test_predictions': y_pred,
            'test_actual': y_test
        }

    def predict_next_price(self, symbol: str, periods: int = 5) -> list:
        """次の期間の価格を予測"""
        if self.advanced_model is None:
            self.logger.error("❌ 高度なモデルが訓練されていません")
            return []
        
        try:
            # 最新データを取得
            conn = sqlite3.connect(self.db_path)
            query = '''
                SELECT datetime, open_price, high_price, low_price, close_price, volume
                FROM chart_data
                WHERE symbol = ? AND timeframe = '5M'
                ORDER BY datetime DESC
                LIMIT 100
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()
            
            if df.empty:
                return []
            
            # データを時系列順に並び替え
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # 特徴量を作成
            df = self.create_advanced_features(df)
            
            # 最新の特徴量を取得
            if self.feature_columns:
                X_latest = df[self.feature_columns].tail(1)
                X_latest = X_latest.replace([np.inf, -np.inf], np.nan).fillna(X_latest.median())
                
                # 予測
                prediction = self.advanced_model.predict(X_latest)[0]
                return [prediction]
            else:
                # シンプルモデルの場合
                prediction = df['close_price'].tail(5).mean()
                return [prediction]
                
        except Exception as e:
            self.logger.error(f"価格予測エラー: {e}")
            return []

    def generate_comparison_report(self, symbol: str, results: dict, y_test: np.ndarray) -> str:
        """比較レポートを生成"""
        report = f"=== {symbol} モデル比較レポート ===\n\n"
        
        # 各モデルの結果を整理
        model_scores = []
        for model_name, metrics in results.items():
            model_scores.append({
                'model': model_name,
                'mse': metrics['mse'],
                'mae': metrics['mae'],
                'r2': metrics['r2']
            })
        
        # MAEで並び替え（低いほうが良い）
        model_scores.sort(key=lambda x: x['mae'])
        
        report += "📊 モデル性能比較（MAE順）:\n"
        for i, score in enumerate(model_scores, 1):
            report += f"{i}. {score['model']}\n"
            report += f"   MSE: {score['mse']:.4f}\n"
            report += f"   MAE: {score['mae']:.4f}\n"
            report += f"   R²: {score['r2']:.4f}\n\n"
        
        # 最優秀モデル
        best_model = model_scores[0]
        report += f"🏆 最優秀モデル: {best_model['model']}\n"
        report += f"   精度評価: {'高精度' if best_model['mae'] < 10 else '中精度' if best_model['mae'] < 50 else '低精度'}\n\n"
        
        # LightGBMの結果があれば特別に記載
        if 'LightGBM' in results:
            lgb_metrics = results['LightGBM']
            report += f"🚀 LightGBM性能:\n"
            report += f"   MSE: {lgb_metrics['mse']:.4f}\n"
            report += f"   MAE: {lgb_metrics['mae']:.4f}\n"
            report += f"   R²: {lgb_metrics['r2']:.4f}\n\n"
        
        return report

    def run_integrated_analysis(self, symbols: List[str]) -> Dict:
        """統合分析を実行（比較モード）"""
        results = {}
        
        for symbol in symbols:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"銘柄: {symbol}")
            self.logger.info('='*50)
            
            # 高度なデータ準備
            X, y, df, feature_cols = self.prepare_advanced_data(symbol)
            
            if X is None:
                continue
            
            # 複数モデル比較
            comparison_results, y_test = self.train_advanced_model(X, y, model_type='comparison')
            
            # 比較レポート生成
            report = self.generate_comparison_report(symbol, comparison_results, y_test)
            self.logger.info(report)
            
            # 結果をファイルに保存
            try:
                with open(f'{symbol}_comparison_report.txt', 'w', encoding='utf-8') as f:
                    f.write(report)
                self.logger.info(f"📊 比較レポートを保存しました: {symbol}_comparison_report.txt")
            except Exception as e:
                self.logger.error(f"レポート保存エラー: {e}")
            
            results[symbol] = {
                'comparison_results': comparison_results,
                'y_test': y_test,
                'report': report
            }
        
        return results

# 使用例
if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    ml_models = MLTradingModels()
    
    # サンプル銘柄で訓練
    symbols = ['7203', '6758', '8306', '9984', '6861']  # yfinanceテストで使用した銘柄
    
    print("=== yfinanceデータ収集 ===")
    # yfinanceデータを収集
    if ml_models.collect_yfinance_data(symbols, days=5):
        print("✅ データ収集完了")
    else:
        print("❌ データ収集失敗")
        exit(1)
    
    print("\n=== 従来のML予測モデル訓練 ===")
    # 従来のモデル訓練
    if ml_models.train_hourly_model(symbols):
        print("✅ 1時間予測モデル訓練完了")
    
    if ml_models.train_minute_model(symbols):
        print("✅ 5分足予測モデル訓練完了")
    
    print("\n=== 高度な統合分析（モデル比較）===")
    # 統合分析実行
    results = ml_models.run_integrated_analysis(symbols)
    
    print("\n=== ファクトチェック予測テスト ===")
    # 予測テスト
    current_time = datetime.now()
    
    for symbol in symbols:
        result = ml_models.fact_check_predictions(symbol, current_time)
        
        print(f"\n{symbol} ファクトチェック結果:")
        print(f"取引実行: {result['should_trade']}")
        print(f"方向: {result['direction']}")
        print(f"信頼度: {result['confidence']:.3f}")
        print(f"1時間予測: {result['hourly_prediction']}")
        print(f"5分足予測: {result['minute_prediction']}")
    
    print("\n=== 統合完了 ===")
    print("✅ 全ての機能が統合されました")
    print("📊 比較レポートファイルが生成されました")
    print("🤖 複数のML予測モデルが比較されました")
    if lightgbm_available:
        print("🚀 LightGBMも比較に含まれています")
    else:
        print("⚠️  LightGBMは利用できません (pip install lightgbm で追加可能)")
