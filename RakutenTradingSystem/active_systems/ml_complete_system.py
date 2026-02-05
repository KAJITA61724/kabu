"""
機械学習モデル - 完全統合版
- yfinance データ収集
- 高度な特徴量エンジニアリング
- 複数モデル比較（RandomForest, LinearRegression, SimpleMovingAverage, LightGBM）
- 予測精度評価
- 一気通貫でモデル作成から評価まで実行
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# yfinanceインポート
try:
    import yfinance as yf
    yfinance_available = True
except ImportError:
    yfinance_available = False

# LightGBMインポート
try:
    import lightgbm as lgb
    lightgbm_available = True
except ImportError:
    lightgbm_available = False

class MLTradingModels:
    """機械学習取引モデル - 完全統合版"""
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.feature_columns = None
        self.models = {}
        self.scalers = {}
        
        # モデル保存ディレクトリ
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # レポート保存ディレクトリ
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        
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
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """高度な特徴量を作成"""
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
    
    def prepare_data(self, symbol: str, period: int = 1000) -> tuple:
        """データを準備"""
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
            df = self.create_features(df)
            
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
    
    def compare_models(self, X, y):
        """複数のモデルを比較"""
        results = {}
        
        # データを分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # スケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. Random Forest
        self.logger.info("Random Forest モデルを訓練中...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        
        results['RandomForest'] = {
            'mse': mean_squared_error(y_test, rf_pred),
            'mae': mean_absolute_error(y_test, rf_pred),
            'r2': r2_score(y_test, rf_pred),
            'predictions': rf_pred,
            'model': rf_model
        }
        
        # 2. Linear Regression
        self.logger.info("Linear Regression モデルを訓練中...")
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        
        results['LinearRegression'] = {
            'mse': mean_squared_error(y_test, lr_pred),
            'mae': mean_absolute_error(y_test, lr_pred),
            'r2': r2_score(y_test, lr_pred),
            'predictions': lr_pred,
            'model': lr_model
        }
        
        # 3. Simple Moving Average
        self.logger.info("Simple Moving Average モデルを訓練中...")
        window = 5
        sma_pred = []
        for i in range(window, len(y_test) + window):
            if i - window < len(y_train):
                pred = np.mean(y_train.iloc[max(0, i-window):i])
                sma_pred.append(pred)
        
        # 長さを調整
        sma_pred = np.array(sma_pred[:len(y_test)])
        if len(sma_pred) < len(y_test):
            # 不足分を最後の値で埋める
            last_val = sma_pred[-1] if len(sma_pred) > 0 else np.mean(y_train)
            sma_pred = np.concatenate([sma_pred, np.full(len(y_test) - len(sma_pred), last_val)])
        
        results['SimpleMovingAverage'] = {
            'mse': mean_squared_error(y_test, sma_pred),
            'mae': mean_absolute_error(y_test, sma_pred),
            'r2': r2_score(y_test, sma_pred),
            'predictions': sma_pred,
            'model': None
        }
        
        # 4. LightGBM (利用可能な場合)
        if lightgbm_available:
            try:
                self.logger.info("LightGBM モデルを訓練中...")
                lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
                lgb_model.fit(X_train, y_train)
                lgb_pred = lgb_model.predict(X_test)
                
                results['LightGBM'] = {
                    'mse': mean_squared_error(y_test, lgb_pred),
                    'mae': mean_absolute_error(y_test, lgb_pred),
                    'r2': r2_score(y_test, lgb_pred),
                    'predictions': lgb_pred,
                    'model': lgb_model
                }
            except Exception as e:
                self.logger.warning(f"LightGBM訓練エラー: {e}")
        
        # スケーラーを保存
        self.scalers['scaler'] = scaler
        
        return results, y_test
    
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
            lgb_rank = next(i for i, score in enumerate(model_scores, 1) if score['model'] == 'LightGBM')
            report += f"🚀 LightGBM性能（{lgb_rank}位）:\n"
            report += f"   MSE: {lgb_metrics['mse']:.4f}\n"
            report += f"   MAE: {lgb_metrics['mae']:.4f}\n"
            report += f"   R²: {lgb_metrics['r2']:.4f}\n\n"
        
        # 現在価格と予測
        try:
            conn = sqlite3.connect(self.db_path)
            query = '''
                SELECT close_price FROM chart_data
                WHERE symbol = ? AND timeframe = '5M'
                ORDER BY datetime DESC LIMIT 1
            '''
            result = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()
            
            if not result.empty:
                current_price = result['close_price'].iloc[0]
                best_model_name = best_model['model']
                best_pred = results[best_model_name]['predictions'][-1]
                
                predicted_change = best_pred - current_price
                change_pct = (predicted_change / current_price) * 100
                
                report += f"💰 価格予測（{best_model_name}）:\n"
                report += f"   現在価格: {current_price:.2f}\n"
                report += f"   予測価格: {best_pred:.2f}\n"
                report += f"   予測変動: {predicted_change:.2f} ({change_pct:.2f}%)\n\n"
        except Exception as e:
            self.logger.error(f"価格予測エラー: {e}")
        
        return report
    
    def save_models(self, symbol: str, results: dict):
        """モデルを保存"""
        try:
            for model_name, metrics in results.items():
                if metrics['model'] is not None:
                    model_path = self.model_dir / f"{symbol}_{model_name}_model.pkl"
                    joblib.dump(metrics['model'], model_path)
                    self.logger.info(f"モデル保存: {model_path}")
            
            # スケーラーを保存
            if 'scaler' in self.scalers:
                scaler_path = self.model_dir / f"{symbol}_scaler.pkl"
                joblib.dump(self.scalers['scaler'], scaler_path)
                self.logger.info(f"スケーラー保存: {scaler_path}")
                
        except Exception as e:
            self.logger.error(f"モデル保存エラー: {e}")
    
    def get_previous_trading_day(self) -> str:
        """前の営業日を取得（月曜の場合は金曜日）"""
        from datetime import datetime, timedelta
        
        today = datetime.now()
        
        # 今日が月曜日（0）の場合は3日前（金曜日）
        if today.weekday() == 0:  # Monday
            previous_day = today - timedelta(days=3)
        # 今日が日曜日（6）の場合は2日前（金曜日）  
        elif today.weekday() == 6:  # Sunday
            previous_day = today - timedelta(days=2)
        # その他の場合は1日前
        else:
            previous_day = today - timedelta(days=1)
        
        return previous_day.strftime('%Y-%m-%d')
    
    def filter_high_volume_symbols(self, symbols: List[str], min_volume: int = 300000) -> List[str]:
        """前の営業日の取引数量が30万株以上の銘柄をフィルター"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 前の営業日を取得
            previous_trading_day = self.get_previous_trading_day()
            
            self.logger.info(f"前の営業日: {previous_trading_day}")
            
            filtered_symbols = []
            
            for symbol in symbols:
                query = '''
                    SELECT AVG(volume) as avg_volume, COUNT(*) as data_count
                    FROM chart_data 
                    WHERE symbol = ? AND datetime LIKE ? AND timeframe = '5M'
                '''
                
                result = pd.read_sql_query(query, conn, params=(symbol, f'{previous_trading_day}%'))
                
                if not result.empty and result['avg_volume'].iloc[0] is not None:
                    avg_volume = result['avg_volume'].iloc[0]
                    data_count = result['data_count'].iloc[0]
                    
                    if avg_volume >= min_volume:
                        filtered_symbols.append(symbol)
                        self.logger.info(f"✅ {symbol}: 平均出来高 {avg_volume:.0f}株 (≥{min_volume}) データ数:{data_count}")
                    else:
                        self.logger.info(f"❌ {symbol}: 平均出来高 {avg_volume:.0f}株 (<{min_volume}) データ数:{data_count}")
                else:
                    self.logger.warning(f"⚠️ {symbol}: {previous_trading_day}のデータなし")
            
            conn.close()
            
            self.logger.info(f"フィルター結果: {len(filtered_symbols)}/{len(symbols)}銘柄が30万株以上")
            return filtered_symbols
            
        except Exception as e:
            self.logger.error(f"出来高フィルターエラー: {e}")
            return symbols  # エラー時は元の銘柄リストを返す

    def run_full_analysis(self, symbols: List[str]) -> Dict:
        """完全な分析を実行（データ収集から評価まで）"""
        all_results = {}
        
        # 0. 出来高フィルター（30万株以上、営業日考慮）
        self.logger.info("=== 出来高フィルター（30万株以上、営業日考慮） ===")
        current_day = datetime.now().strftime('%A')  # 曜日名を取得
        self.logger.info(f"今日は{current_day}です")
        
        filtered_symbols = self.filter_high_volume_symbols(symbols)
        
        if not filtered_symbols:
            self.logger.warning("⚠️ フィルター後の銘柄がありません。元の銘柄リストを使用します。")
            filtered_symbols = symbols
        
        # 1. データ収集
        self.logger.info("=== yfinanceデータ収集 ===")
        if self.collect_yfinance_data(filtered_symbols, days=5):
            self.logger.info("✅ データ収集完了")
        else:
            self.logger.warning("⚠️ データ収集で問題がありました。既存データで続行...")
        
        # 2. 各銘柄でモデル比較
        for symbol in filtered_symbols:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🔍 銘柄分析: {symbol}")
            self.logger.info('='*60)
            
            # データ準備
            X, y, df, feature_cols = self.prepare_data(symbol)
            
            if X is None:
                self.logger.warning(f"⚠️ {symbol} のデータが不足しています。スキップします。")
                continue
            
            # モデル比較
            comparison_results, y_test = self.compare_models(X, y)
            
            # レポート生成
            report = self.generate_comparison_report(symbol, comparison_results, y_test)
            self.logger.info(f"\n{report}")
            
            # 結果をファイルに保存
            try:
                report_path = self.reports_dir / f'{symbol}_comparison_report.txt'
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                self.logger.info(f"📊 レポート保存: {report_path}")
            except Exception as e:
                self.logger.error(f"レポート保存エラー: {e}")
            
            # モデル保存
            self.save_models(symbol, comparison_results)
            
            all_results[symbol] = {
                'comparison_results': comparison_results,
                'y_test': y_test,
                'report': report
            }
        
        # 3. フィルター結果のサマリー
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 フィルター結果サマリー")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"元の銘柄数: {len(symbols)}")
        self.logger.info(f"フィルター後: {len(filtered_symbols)}")
        self.logger.info(f"分析完了: {len(all_results)}")
        
        return all_results
    
    def print_summary(self, all_results: Dict):
        """全体のサマリーを表示"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("📈 全体サマリー")
        self.logger.info('='*60)
        
        for symbol, results in all_results.items():
            comparison_results = results['comparison_results']
            
            # 最優秀モデルを取得
            best_model_name = min(comparison_results.keys(), 
                                key=lambda x: comparison_results[x]['mae'])
            best_mae = comparison_results[best_model_name]['mae']
            
            self.logger.info(f"{symbol}: 最優秀モデル = {best_model_name} (MAE: {best_mae:.4f})")
        
        # LightGBMの利用状況
        if lightgbm_available:
            self.logger.info("\n🚀 LightGBMが利用されました")
        else:
            self.logger.info("\n⚠️ LightGBMが利用できません (pip install lightgbm で追加可能)")


def main():
    """メイン実行関数"""
    # ログ設定
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ml_analysis.log', encoding='utf-8')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 機械学習モデル統合分析システム 開始")
    logger.info("="*60)
    
    # MLモデルインスタンス作成
    ml_models = MLTradingModels()
    
    # 分析対象銘柄
    symbols = ['7203', '6758', '8306', '9984', '6861']
    logger.info(f"分析対象銘柄: {', '.join(symbols)}")
    
    # 完全な分析実行
    all_results = ml_models.run_full_analysis(symbols)
    
    # サマリー表示
    ml_models.print_summary(all_results)
    
    logger.info("\n✅ 分析完了!")
    logger.info("📊 レポートファイル: reports/フォルダ内")
    logger.info("🤖 モデルファイル: models/フォルダ内")
    logger.info("📝 ログファイル: ml_analysis.log")


if __name__ == "__main__":
    main()
