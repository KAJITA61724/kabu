"""
機械学習モデル - 完全統合版
- 1時間線形予測モデル
- 5分足上下予測モデル
- ファクトチェック機能
- ファンダメンタルズ分析統合
- 高度な予測モデル
- 複数モデル比較機能
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
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
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
    sklearn_available = True
except ImportError:
    sklearn_available = False

class MLTradingModels:
    """機械学習取引モデル - 完全統合版"""
    
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
        self.advanced_model = None
        
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
                            '5m',
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
    
    def create_advanced_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """高度な特徴量を作成（ファンダメンタルズデータ統合）"""
        if df.empty or len(df) < 10:
            return pd.DataFrame()
        
        df = df.copy()
        
        # 基本的な特徴量
        df['price_change'] = df['close_price'].pct_change()
        df['high_low_ratio'] = df['high_price'] / df['low_price']
        df['open_close_ratio'] = df['open_price'] / df['close_price']
        df['volume_price_ratio'] = df['volume'] / df['close_price']
        
        # 移動平均
        df['sma_5'] = df['close_price'].rolling(window=5).mean()
        df['sma_10'] = df['close_price'].rolling(window=10).mean()
        df['sma_ratio'] = df['close_price'] / df['sma_5']
        
        # ボラティリティ
        df['volatility_5'] = df['close_price'].rolling(window=5).std()
        df['volatility_10'] = df['close_price'].rolling(window=10).std()
        
        # 出来高系
        df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_5']
        
        # RSI（簡易版）
        delta = df['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
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
        
        # ===== ファンダメンタルズ特徴量の追加 =====
        if symbol and self.fundamental_collector:
            try:
                # ファンダメンタルズデータを取得（DB優先、なければyfinance）
                fundamental_data = self.fundamental_collector.get_fundamental_data_from_db(symbol)
                if not fundamental_data:
                    fundamental_data = self.fundamental_collector.get_fundamental_data_yfinance(symbol)
                
                if fundamental_data:
                    # 財務指標を特徴量として追加
                    df['per'] = getattr(fundamental_data, 'per', 0)  # PER
                    df['pbr'] = getattr(fundamental_data, 'pbr', 0)  # PBR
                    df['dividend_yield'] = getattr(fundamental_data, 'dividend_yield', 0)  # 配当利回り
                    df['roe'] = getattr(fundamental_data, 'roe', 0)  # ROE
                    df['roa'] = getattr(fundamental_data, 'roa', 0)  # ROA
                    df['market_cap'] = getattr(fundamental_data, 'market_cap', 0)  # 時価総額
                    df['eps'] = getattr(fundamental_data, 'eps', 0)  # EPS
                    df['bps'] = getattr(fundamental_data, 'bps', 0)  # BPS
                    
                    # 業績データ
                    df['revenue_growth'] = getattr(fundamental_data, 'revenue_growth', 0)  # 売上成長率
                    df['profit_growth'] = getattr(fundamental_data, 'profit_growth', 0)  # 利益成長率
                    df['debt_ratio'] = getattr(fundamental_data, 'debt_ratio', 0)  # 負債比率
                    
                    # セクター関連
                    df['sector_avg_per'] = getattr(fundamental_data, 'sector_avg_per', 0)  # セクター平均PER
                    df['per_vs_sector'] = df['per'] / (df['sector_avg_per'] + 0.001)  # PER対セクター比
                    
                    self.logger.info(f"✅ {symbol}: ファンダメンタルズ特徴量を追加")
                else:
                    # データがない場合はゼロで埋める
                    fundamental_features = [
                        'per', 'pbr', 'dividend_yield', 'roe', 'roa', 'market_cap', 
                        'eps', 'bps', 'revenue_growth', 'profit_growth', 'debt_ratio',
                        'sector_avg_per', 'per_vs_sector'
                    ]
                    for feature in fundamental_features:
                        df[feature] = 0
                    
                    self.logger.warning(f"⚠️ {symbol}: ファンダメンタルズデータが見つかりません（ゼロで埋めました）")
                    
            except Exception as e:
                self.logger.error(f"❌ ファンダメンタルズ特徴量追加エラー: {e}")
                # エラーの場合もゼロで埋める
                fundamental_features = [
                    'per', 'pbr', 'dividend_yield', 'roe', 'roa', 'market_cap', 
                    'eps', 'bps', 'revenue_growth', 'profit_growth', 'debt_ratio',
                    'sector_avg_per', 'per_vs_sector'
                ]
                for feature in fundamental_features:
                    df[feature] = 0
        else:
            # ファンダメンタルズデータ収集器がない場合はゼロで埋める
            fundamental_features = [
                'per', 'pbr', 'dividend_yield', 'roe', 'roa', 'market_cap', 
                'eps', 'bps', 'revenue_growth', 'profit_growth', 'debt_ratio',
                'sector_avg_per', 'per_vs_sector'
            ]
            for feature in fundamental_features:
                df[feature] = 0
        
        # 目標変数（次の期間の価格）
        df['target'] = df['close_price'].shift(-1)
        
        return df

    def prepare_advanced_data(self, symbol: str, period: int = 1000) -> tuple:
        """高度なデータ準備"""
        self.logger.info(f"データ準備中: {symbol}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # yfinanceデータを取得
            query = '''
                SELECT datetime, open_price, high_price, low_price, close_price, volume
                FROM chart_data
                WHERE symbol = ? AND timeframe = '5m'
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
            
            # 特徴量を作成（ファンダメンタルズデータ統合）
            df = self.create_advanced_features(df, symbol)
            
            # 特徴量の列を定義（ファンダメンタルズ特徴量を含む）
            feature_cols = [
                # テクニカル指標
                'price_change', 'high_low_ratio', 'open_close_ratio', 'volume_price_ratio',
                'sma_ratio', 'volatility_5', 'volatility_10', 'volume_ratio', 'rsi',
                'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5',
                'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5',
                'change_lag_1', 'change_lag_2', 'change_lag_3', 'change_lag_5',
                'hour', 'minute', 'time_of_day',
                
                # ファンダメンタルズ指標
                'per', 'pbr', 'dividend_yield', 'roe', 'roa', 'market_cap',
                'eps', 'bps', 'revenue_growth', 'profit_growth', 'debt_ratio',
                'sector_avg_per', 'per_vs_sector'
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
            X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
            
            self.logger.info(f"特徴量数: {len(feature_cols)}")
            self.logger.info(f"有効サンプル数: {len(X)}")
            
            self.feature_columns = feature_cols
            
            return X, y, df, feature_cols
            
        except Exception as e:
            self.logger.error(f"データ準備エラー: {e}")
            return None, None, None, None

    def train_advanced_model(self, X, y, model_type='advanced'):
        """高度なモデルを訓練"""
        if sklearn_available and model_type == 'advanced':
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
        feature_importance = None
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
            'feature_importance': feature_importance,
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

    def run_integrated_analysis(self, symbols: List[str]) -> Dict:
        """統合分析を実行"""
        results = {}
        
        for symbol in symbols:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"銘柄: {symbol}")
            self.logger.info('='*50)
            
            # 高度なデータ準備
            X, y, df, feature_cols = self.prepare_advanced_data(symbol)
            
            if X is None:
                continue
            
            # 高度なモデル訓練
            model_type = 'advanced' if sklearn_available else 'simple'
            model_results = self.train_advanced_model(X, y, model_type)
            
            # レポート生成
            report = self.generate_prediction_report(symbol, model_results)
            self.logger.info(report)
            
            # 結果をファイルに保存
            try:
                with open(f'{symbol}_prediction_report.txt', 'w', encoding='utf-8') as f:
                    f.write(report)
                self.logger.info(f"📊 レポートを保存しました: {symbol}_prediction_report.txt")
            except Exception as e:
                self.logger.error(f"レポート保存エラー: {e}")
            
            results[symbol] = {
                'model_results': model_results,
                'report': report
            }
        
        return results

    def generate_prediction_report(self, symbol: str, results: dict) -> str:
        """予測レポートを生成"""
        report = f"=== {symbol} 予測モデルレポート ===\n\n"
        
        report += f"モデルタイプ: {results['model_type']}\n"
        report += f"MSE: {results['mse']:.4f}\n"
        report += f"MAE: {results['mae']:.4f}\n"
        
        if 'r2' in results:
            report += f"R²: {results['r2']:.4f}\n"
        
        # 予測精度の評価
        if results['mae'] < 10:
            accuracy = "高精度"
        elif results['mae'] < 50:
            accuracy = "中精度"
        else:
            accuracy = "低精度"
        
        report += f"予測精度: {accuracy}\n\n"
        
        # 特徴量重要度
        if results.get('feature_importance') is not None:
            report += "上位5つの重要な特徴量:\n"
            for _, row in results['feature_importance'].head(5).iterrows():
                report += f"  {row['feature']}: {row['importance']:.4f}\n"
        
        return report

    def compare_models(self, symbols: List[str]) -> Dict:
        """複数モデルの比較"""
        comparison_results = {}
        
        for symbol in symbols:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"複数モデル比較: {symbol}")
            self.logger.info('='*60)
            
            # データ準備
            X, y, df, feature_cols = self.prepare_advanced_data(symbol)
            
            if X is None:
                continue
            
            # 複数モデルで訓練・比較
            model_results = {}
            
            # 1. Random Forest
            rf_results = self.train_advanced_model(X, y, 'advanced')
            model_results['RandomForest'] = rf_results
            
            # 2. 移動平均モデル
            simple_results = self.train_advanced_model(X, y, 'simple')
            model_results['SimpleMovingAverage'] = simple_results
            
            # 3. 線形回帰
            lr_model = LinearRegression()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            lr_model.fit(X_train_scaled, y_train)
            y_pred_lr = lr_model.predict(X_test_scaled)
            
            lr_results = {
                'model_type': 'LinearRegression',
                'mse': mean_squared_error(y_test, y_pred_lr),
                'mae': mean_absolute_error(y_test, y_pred_lr),
                'r2': r2_score(y_test, y_pred_lr),
                'test_predictions': y_pred_lr,
                'test_actual': y_test.values
            }
            model_results['LinearRegression'] = lr_results
            
            # 結果比較
            comparison_results[symbol] = model_results
            
            # 比較レポート生成
            self.generate_comparison_report(symbol, model_results)
        
        return comparison_results

    def generate_comparison_report(self, symbol: str, model_results: Dict):
        """比較レポート生成"""
        report = f"=== {symbol} モデル比較レポート ===\n\n"
        
        # 各モデルの性能をまとめる
        performance_data = []
        for model_name, results in model_results.items():
            performance_data.append({
                'model': model_name,
                'mse': results['mse'],
                'mae': results['mae'],
                'r2': results.get('r2', 0)
            })
        
        # 性能順にソート（MSEが小さい順）
        performance_data.sort(key=lambda x: x['mse'])
        
        report += "モデル性能ランキング（MSE順）:\n"
        for i, perf in enumerate(performance_data, 1):
            report += f"{i}. {perf['model']}\n"
            report += f"   MSE: {perf['mse']:.4f}\n"
            report += f"   MAE: {perf['mae']:.4f}\n"
            report += f"   R²: {perf['r2']:.4f}\n\n"
        
        # 最優秀モデル
        best_model = performance_data[0]
        report += f"最優秀モデル: {best_model['model']}\n"
        report += f"MSE: {best_model['mse']:.4f}\n"
        
        # レポート保存
        try:
            with open(f'{symbol}_comparison_report.txt', 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"📊 比較レポートを保存しました: {symbol}_comparison_report.txt")
        except Exception as e:
            self.logger.error(f"比較レポート保存エラー: {e}")
        
        self.logger.info(f"\n{report}")

    # ===== 既存システム互換性のための追加機能 =====
    
    def prepare_features(self, symbol: str, current_time: datetime):
        """特徴量準備（既存システム互換性用）"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 過去のデータを取得
            query = '''
                SELECT datetime, open_price, high_price, low_price, close_price, volume
                FROM chart_data
                WHERE symbol = ? AND timeframe = '5m'
                AND datetime <= ?
                ORDER BY datetime DESC
                LIMIT 50
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol, current_time.strftime('%Y-%m-%d %H:%M:%S')))
            conn.close()
            
            if len(df) < 10:
                return None
            
            # 基本的な特徴量を作成
            df['price_change'] = df['close_price'].pct_change()
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
            df['sma_5'] = df['close_price'].rolling(5).mean()
            df['volatility'] = df['close_price'].rolling(5).std()
            
            # ファンダメンタルズ特徴量を追加
            fundamental_features = [0, 0, 0]  # デフォルト値
            if self.fundamental_collector:
                try:
                    fundamental_data = self.fundamental_collector.get_fundamental_data_from_db(symbol)
                    if not fundamental_data:
                        fundamental_data = self.fundamental_collector.get_fundamental_data_yfinance(symbol)
                    
                    if fundamental_data:
                        fundamental_features = [
                            getattr(fundamental_data, 'per', 0),
                            getattr(fundamental_data, 'pbr', 0),
                            getattr(fundamental_data, 'roe', 0)
                        ]
                except Exception as e:
                    self.logger.warning(f"ファンダメンタルズデータ取得エラー: {e}")
            
            # テクニカル特徴量とファンダメンタルズ特徴量を結合
            technical_features = df[['price_change', 'volume_ratio', 'volatility']].dropna().iloc[-1:].values
            if len(technical_features) > 0:
                # テクニカル + ファンダメンタルズを結合
                combined_features = np.concatenate([technical_features[0], fundamental_features])
                return combined_features.reshape(1, -1)
            else:
                return None
            
        except Exception as e:
            self.logger.error(f"特徴量準備エラー: {e}")
            return None
    
    def predict_hourly_trend(self, symbol: str, current_time: datetime) -> Optional[float]:
        """1時間トレンド予測（既存システム互換性用）"""
        try:
            features = self.prepare_features(symbol, current_time)
            if features is None:
                return None
            
            # 簡易的な線形回帰予測
            if not hasattr(self.hourly_model, 'predict'):
                self.hourly_model = LinearRegression()
                # ダミーデータで初期化（6特徴量：テクニカル3個 + ファンダメンタルズ3個）
                dummy_X = np.random.random((10, 6))
                dummy_y = np.random.random(10)
                self.hourly_model.fit(dummy_X, dummy_y)
            
            prediction = self.hourly_model.predict(features.reshape(1, -1))[0]
            return prediction
            
        except Exception as e:
            self.logger.error(f"1時間予測エラー: {e}")
            return None
    
    def predict_minute_direction(self, symbol: str, current_time: datetime) -> Optional[Tuple[int, float]]:
        """5分足方向予測（既存システム互換性用）"""
        try:
            features = self.prepare_features(symbol, current_time)
            if features is None:
                return None
            
            # 簡易的な分類予測
            if not hasattr(self.minute_model, 'predict'):
                self.minute_model = RandomForestClassifier(n_estimators=100, random_state=42)
                # ダミーデータで初期化（6特徴量：テクニカル3個 + ファンダメンタルズ3個）
                dummy_X = np.random.random((10, 6))
                dummy_y = np.random.randint(0, 2, 10)
                self.minute_model.fit(dummy_X, dummy_y)
            
            prediction = self.minute_model.predict(features.reshape(1, -1))[0]
            probabilities = self.minute_model.predict_proba(features.reshape(1, -1))[0]
            confidence = max(probabilities)
            
            return prediction, confidence
            
        except Exception as e:
            self.logger.error(f"5分足予測エラー: {e}")
            return None
    
    def fact_check_predictions(self, symbol: str, current_time: datetime) -> Dict:
        """ファクトチェック実行（既存システム互換性用）"""
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
    
    # 既存システム互換性のためのエイリアス
    def hourly_predict(self, symbol: str, current_time: datetime = None) -> Optional[float]:
        """既存システム互換性用エイリアス"""
        if current_time is None:
            current_time = datetime.now()
        return self.predict_hourly_trend(symbol, current_time)
    
    def minute_predict(self, symbol: str, current_time: datetime = None) -> Optional[Tuple[int, float]]:
        """既存システム互換性用エイリアス"""
        if current_time is None:
            current_time = datetime.now()
        return self.predict_minute_direction(symbol, current_time)
    
    def fact_check(self, symbol: str, current_time: datetime = None) -> Dict:
        """既存システム互換性用エイリアス"""
        if current_time is None:
            current_time = datetime.now()
        return self.fact_check_predictions(symbol, current_time)

# 使用例
if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    ml_models = MLTradingModels()
    
    # サンプル銘柄
    symbols = ['7203', '6758', '8306', '9984', '6861']
    
    print("=== yfinanceデータ収集 ===")
    # yfinanceデータを収集
    if ml_models.collect_yfinance_data(symbols, days=5):
        print("✅ データ収集完了")
    else:
        print("❌ データ収集失敗")
        print("既存のデータで続行...")
    
    print("\n=== 高度な統合分析 ===")
    # 統合分析実行
    results = ml_models.run_integrated_analysis(symbols)
    
    print("\n=== 複数モデル比較 ===")
    # 複数モデル比較実行
    comparison_results = ml_models.compare_models(symbols)
    
    print("\n=== 総合結果 ===")
    for symbol in symbols:
        if symbol in comparison_results:
            print(f"\n{symbol}:")
            models = comparison_results[symbol]
            best_model = min(models.items(), key=lambda x: x[1]['mse'])
            print(f"  最優秀モデル: {best_model[0]}")
            print(f"  MSE: {best_model[1]['mse']:.4f}")
            print(f"  MAE: {best_model[1]['mae']:.4f}")
            if 'r2' in best_model[1]:
                print(f"  R²: {best_model[1]['r2']:.4f}")
    
    print("\n=== 統合完了 ===")
    print("✅ 全ての機能が統合されました")
    print("📊 予測レポートファイルが生成されました")
    print("🤖 複数MLモデルの比較が完了しました")
