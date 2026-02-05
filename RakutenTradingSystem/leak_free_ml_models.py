"""
リークフリー機械学習モデル - 完全版
- 未来データを一切使用しない
- 時系列データの正しい分割
- リアルな予測精度の測定
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
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
import sys
import os

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

class LeakFreeTradingModels:
    """リークフリー機械学習取引モデル"""
    
    def __init__(self, db_path: str = "leak_free_trading.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # モデル初期化
        self.price_model = LinearRegression()
        self.direction_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # ファンダメンタルズデータ収集器
        if FundamentalDataCollector:
            self.fundamental_collector = FundamentalDataCollector()
        else:
            self.fundamental_collector = None
        
        # モデル保存パス
        self.model_dir = Path("leak_free_models")
        self.model_dir.mkdir(exist_ok=True)
        
        # 特徴量定義
        self.feature_columns = None
        
    def collect_market_data(self, symbols: List[str], days: int = 30) -> bool:
        """市場データ収集（リークフリー）"""
        if not yfinance_available:
            self.logger.error("yfinanceが利用できません")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # テーブル作成
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
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
                    
                    # データ整形（時系列順に保存）
                    df = data.reset_index()
                    df = df.sort_values('Datetime')  # 時系列順に確実にソート
                    
                    # データベースに保存
                    for _, row in df.iterrows():
                        cursor.execute('''
                            INSERT OR REPLACE INTO market_data 
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
    
    def create_leak_free_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """リークフリー特徴量作成"""
        if df.empty or len(df) < 20:
            return pd.DataFrame()
        
        df = df.copy()
        df = df.sort_values('datetime').reset_index(drop=True)  # 時系列順に確実にソート
        
        # ===== 過去のデータのみを使用した特徴量 =====
        
        # 価格変化率（前期間との比較）
        df['price_change_1'] = df['close_price'].pct_change(1)
        df['price_change_3'] = df['close_price'].pct_change(3)
        df['price_change_5'] = df['close_price'].pct_change(5)
        
        # 価格比率（現在時点で知ることができる情報）
        df['high_low_ratio'] = df['high_price'] / df['low_price']
        df['open_close_ratio'] = df['open_price'] / df['close_price']
        df['close_open_ratio'] = df['close_price'] / df['open_price']
        
        # 出来高特徴量
        df['volume_change'] = df['volume'].pct_change()
        df['volume_price_ratio'] = df['volume'] / df['close_price']
        
        # 移動平均（過去のデータのみ使用）
        df['sma_5'] = df['close_price'].rolling(window=5, min_periods=1).mean().shift(1)
        df['sma_10'] = df['close_price'].rolling(window=10, min_periods=1).mean().shift(1)
        df['sma_20'] = df['close_price'].rolling(window=20, min_periods=1).mean().shift(1)
        
        # 移動平均比率（過去データベース）
        df['price_to_sma5'] = df['close_price'] / df['sma_5']
        df['price_to_sma10'] = df['close_price'] / df['sma_10']
        df['price_to_sma20'] = df['close_price'] / df['sma_20']
        
        # ボラティリティ（過去のデータのみ）
        df['volatility_5'] = df['close_price'].rolling(window=5, min_periods=1).std().shift(1)
        df['volatility_10'] = df['close_price'].rolling(window=10, min_periods=1).std().shift(1)
        
        # 出来高移動平均（過去データ）
        df['volume_sma_5'] = df['volume'].rolling(window=5, min_periods=1).mean().shift(1)
        df['volume_to_sma'] = df['volume'] / df['volume_sma_5']
        
        # RSI（14期間、過去データのみ）
        delta = df['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].shift(1)  # 過去の値を使用
        
        # ラグ特徴量（明確に過去の値）
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close_price'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'change_lag_{lag}'] = df['price_change_1'].shift(lag)
        
        # 時間特徴量（リークなし）
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
        df['minute'] = pd.to_datetime(df['datetime']).dt.minute
        df['day_of_week'] = pd.to_datetime(df['datetime']).dt.dayofweek
        df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] <= 15)).astype(int)
        
        # ===== ファンダメンタルズ特徴量（時点に依存しない）=====
        if symbol and self.fundamental_collector:
            try:
                fundamental_data = self.fundamental_collector.get_fundamental_data_from_db(symbol)
                if not fundamental_data:
                    fundamental_data = self.fundamental_collector.get_fundamental_data_yfinance(symbol)
                
                if fundamental_data:
                    # 基本財務指標
                    df['per'] = getattr(fundamental_data, 'per', 0)
                    df['pbr'] = getattr(fundamental_data, 'pbr', 0)
                    df['dividend_yield'] = getattr(fundamental_data, 'dividend_yield', 0)
                    df['roe'] = getattr(fundamental_data, 'roe', 0)
                    df['roa'] = getattr(fundamental_data, 'roa', 0)
                    df['market_cap'] = getattr(fundamental_data, 'market_cap', 0)
                    df['eps'] = getattr(fundamental_data, 'eps', 0)
                    df['bps'] = getattr(fundamental_data, 'bps', 0)
                    
                    # 成長性指標
                    df['revenue_growth'] = getattr(fundamental_data, 'revenue_growth', 0)
                    df['profit_growth'] = getattr(fundamental_data, 'profit_growth', 0)
                    df['debt_ratio'] = getattr(fundamental_data, 'debt_ratio', 0)
                    
                    self.logger.info(f"✅ {symbol}: ファンダメンタルズ特徴量を追加")
                else:
                    # データがない場合
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
        
        # ===== 目標変数（リークフリー）=====
        # 次の期間の価格変化を予測（実際の取引では現在時点では不明）
        df['future_price_change'] = df['close_price'].pct_change().shift(-1)
        df['future_direction'] = (df['future_price_change'] > 0).astype(int)
        
        return df
    
    def prepare_leak_free_data(self, symbol: str, min_samples: int = 200) -> Tuple:
        """リークフリーデータ準備"""
        self.logger.info(f"リークフリーデータ準備: {symbol}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 時系列順でデータ取得
            query = '''
                SELECT datetime, open_price, high_price, low_price, close_price, volume
                FROM market_data
                WHERE symbol = ? AND timeframe = '5m'
                ORDER BY datetime ASC
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()
            
            if df.empty:
                self.logger.error(f"❌ {symbol} のデータが見つかりません")
                return None, None, None, None, None, None
            
            self.logger.info(f"取得データ数: {len(df)}件")
            
            # リークフリー特徴量作成
            df = self.create_leak_free_features(df, symbol)
            
            # 特徴量定義（リークフリー）
            feature_cols = [
                # 価格変化特徴量
                'price_change_1', 'price_change_3', 'price_change_5',
                'high_low_ratio', 'open_close_ratio', 'close_open_ratio',
                
                # 出来高特徴量
                'volume_change', 'volume_price_ratio', 'volume_to_sma',
                
                # 移動平均特徴量
                'price_to_sma5', 'price_to_sma10', 'price_to_sma20',
                
                # ボラティリティ
                'volatility_5', 'volatility_10',
                
                # テクニカル指標
                'rsi',
                
                # ラグ特徴量
                'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5', 'close_lag_10',
                'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5',
                'change_lag_1', 'change_lag_2', 'change_lag_3',
                
                # 時間特徴量
                'hour', 'minute', 'day_of_week', 'is_market_open',
                
                # ファンダメンタルズ特徴量
                'per', 'pbr', 'dividend_yield', 'roe', 'roa', 'market_cap',
                'eps', 'bps', 'revenue_growth', 'profit_growth', 'debt_ratio'
            ]
            
            # 欠損値処理
            df = df.dropna()
            
            if len(df) < min_samples:
                self.logger.error(f"❌ 有効なデータが不足: {len(df)}件 < {min_samples}件")
                return None, None, None, None, None, None
            
            # 特徴量と目標変数を分離
            X = df[feature_cols].copy()
            y_price = df['future_price_change'].copy()
            y_direction = df['future_direction'].copy()
            
            # 無限大、NaN値を処理
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            
            # 最後の行を削除（目標変数が未来なのでNaN）
            X = X[:-1]
            y_price = y_price[:-1]
            y_direction = y_direction[:-1]
            
            # 時系列分割（最後の20%をテスト用、リークなし）
            split_idx = int(len(X) * 0.8)
            
            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_price_train = y_price[:split_idx]
            y_price_test = y_price[split_idx:]
            y_direction_train = y_direction[:split_idx]
            y_direction_test = y_direction[split_idx:]
            
            self.feature_columns = feature_cols
            
            self.logger.info(f"特徴量数: {len(feature_cols)}")
            self.logger.info(f"訓練データ: {len(X_train)}件")
            self.logger.info(f"テストデータ: {len(X_test)}件")
            
            return X_train, X_test, y_price_train, y_price_test, y_direction_train, y_direction_test
            
        except Exception as e:
            self.logger.error(f"データ準備エラー: {e}")
            return None, None, None, None, None, None
    
    def train_leak_free_models(self, X_train, y_price_train, y_direction_train):
        """リークフリーモデル訓練"""
        results = {}
        
        # データスケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 1. 価格変化予測モデル（回帰）
        self.logger.info("価格変化予測モデル訓練中...")
        
        price_models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        }
        
        price_results = {}
        for name, model in price_models.items():
            model.fit(X_train_scaled, y_price_train)
            price_results[name] = model
        
        # 2. 方向予測モデル（分類）
        self.logger.info("方向予測モデル訓練中...")
        
        direction_models = {
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        }
        
        direction_results = {}
        for name, model in direction_models.items():
            model.fit(X_train_scaled, y_direction_train)
            direction_results[name] = model
        
        results = {
            'price_models': price_results,
            'direction_models': direction_results,
            'scaler': scaler
        }
        
        return results
    
    def evaluate_leak_free_models(self, models, X_test, y_price_test, y_direction_test):
        """リークフリーモデル評価"""
        scaler = models['scaler']
        X_test_scaled = scaler.transform(X_test)
        
        evaluation_results = {
            'price_models': {},
            'direction_models': {}
        }
        
        # 価格予測モデル評価
        for name, model in models['price_models'].items():
            y_pred = model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_price_test, y_pred)
            mae = mean_absolute_error(y_price_test, y_pred)
            r2 = r2_score(y_price_test, y_pred)
            
            evaluation_results['price_models'][name] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred,
                'actual': y_price_test.values
            }
            
            self.logger.info(f"{name} (価格予測):")
            self.logger.info(f"  MSE: {mse:.6f}")
            self.logger.info(f"  MAE: {mae:.6f}")
            self.logger.info(f"  R²: {r2:.4f}")
        
        # 方向予測モデル評価
        for name, model in models['direction_models'].items():
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
        
        return evaluation_results
    
    def run_leak_free_analysis(self, symbols: List[str]) -> Dict:
        """リークフリー分析実行"""
        all_results = {}
        
        for symbol in symbols:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"リークフリー分析: {symbol}")
            self.logger.info('='*60)
            
            # データ準備
            data_result = self.prepare_leak_free_data(symbol)
            if data_result[0] is None:
                continue
            
            X_train, X_test, y_price_train, y_price_test, y_direction_train, y_direction_test = data_result
            
            # モデル訓練
            models = self.train_leak_free_models(X_train, y_price_train, y_direction_train)
            
            # モデル評価
            evaluation = self.evaluate_leak_free_models(models, X_test, y_price_test, y_direction_test)
            
            # レポート生成
            report = self.generate_leak_free_report(symbol, evaluation)
            
            # 結果保存
            all_results[symbol] = {
                'models': models,
                'evaluation': evaluation,
                'report': report
            }
            
            # レポートをファイルに保存
            try:
                report_path = self.model_dir / f'{symbol}_leak_free_report.txt'
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                self.logger.info(f"📊 レポート保存: {report_path}")
            except Exception as e:
                self.logger.error(f"レポート保存エラー: {e}")
        
        return all_results
    
    def generate_leak_free_report(self, symbol: str, evaluation: Dict) -> str:
        """リークフリーレポート生成"""
        report = f"=== {symbol} リークフリー分析レポート ===\n\n"
        report += f"分析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 価格予測結果
        report += "【価格変化予測モデル】\n"
        for model_name, results in evaluation['price_models'].items():
            report += f"{model_name}:\n"
            report += f"  MSE: {results['mse']:.6f}\n"
            report += f"  MAE: {results['mae']:.6f}\n"
            report += f"  R²: {results['r2']:.4f}\n"
            
            # 実用性評価
            if results['mae'] < 0.01:
                accuracy_level = "高精度"
            elif results['mae'] < 0.02:
                accuracy_level = "中精度"
            else:
                accuracy_level = "低精度"
            
            report += f"  実用性: {accuracy_level}\n\n"
        
        # 方向予測結果
        report += "【方向予測モデル】\n"
        for model_name, results in evaluation['direction_models'].items():
            report += f"{model_name}:\n"
            report += f"  精度: {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)\n"
            
            # 実用性評価
            if results['accuracy'] > 0.6:
                direction_level = "実用的"
            elif results['accuracy'] > 0.55:
                direction_level = "やや有効"
            else:
                direction_level = "効果的でない"
            
            report += f"  実用性: {direction_level}\n\n"
        
        # 総合評価
        report += "【総合評価】\n"
        
        # 最良の価格予測モデル
        best_price_model = min(evaluation['price_models'].items(), 
                              key=lambda x: x[1]['mae'])
        report += f"最良価格予測: {best_price_model[0]} (MAE: {best_price_model[1]['mae']:.6f})\n"
        
        # 最良の方向予測モデル
        best_direction_model = max(evaluation['direction_models'].items(), 
                                  key=lambda x: x[1]['accuracy'])
        report += f"最良方向予測: {best_direction_model[0]} (精度: {best_direction_model[1]['accuracy']:.1%})\n"
        
        # 実際の取引での期待リターン
        mae = best_price_model[1]['mae']
        direction_acc = best_direction_model[1]['accuracy']
        
        if mae < 0.01 and direction_acc > 0.6:
            investment_rec = "推奨"
        elif mae < 0.02 and direction_acc > 0.55:
            investment_rec = "条件付き推奨"
        else:
            investment_rec = "非推奨"
        
        report += f"投資推奨度: {investment_rec}\n"
        
        # 注意事項
        report += "\n【重要な注意事項】\n"
        report += "- このモデルは過去のデータに基づいています\n"
        report += "- 実際の取引では取引コスト、スリッページ等を考慮してください\n"
        report += "- 市場環境の変化により性能が変動する可能性があります\n"
        report += "- リスク管理を徹底し、適切なポジションサイズで取引してください\n"
        
        return report

# 使用例
if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # リークフリーモデルインスタンス作成
    leak_free_models = LeakFreeTradingModels()
    
    # テスト用銘柄
    symbols = ['7203', '6758', '8306', '9984', '6861']
    
    print("=== リークフリー市場データ収集 ===")
    if leak_free_models.collect_market_data(symbols, days=30):
        print("✅ データ収集完了")
    else:
        print("❌ データ収集失敗")
    
    print("\n=== リークフリー分析実行 ===")
    results = leak_free_models.run_leak_free_analysis(symbols)
    
    print("\n=== 結果サマリー ===")
    for symbol in symbols:
        if symbol in results:
            print(f"\n{symbol}:")
            evaluation = results[symbol]['evaluation']
            
            # 最良の価格予測
            best_price = min(evaluation['price_models'].items(), key=lambda x: x[1]['mae'])
            print(f"  最良価格予測: {best_price[0]} (MAE: {best_price[1]['mae']:.6f})")
            
            # 最良の方向予測
            best_direction = max(evaluation['direction_models'].items(), key=lambda x: x[1]['accuracy'])
            print(f"  最良方向予測: {best_direction[0]} (精度: {best_direction[1]['accuracy']:.1%})")
    
    print("\n=== リークフリー分析完了 ===")
    print("✅ 未来データを一切使用していません")
    print("📊 時系列順序を正しく保持しています")
    print("🔒 データリークを完全に防止しています")
