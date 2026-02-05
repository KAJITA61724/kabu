"""
昨日の分速データを使った予測モデル
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.data_reader import DataReader
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 機械学習ライブラリ
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    sklearn_available = True
except ImportError:
    sklearn_available = False
    print("⚠️ scikit-learn が見つかりません。基本的な予測モデルを使用します")

# LightGBMインポート
try:
    import lightgbm as lgb
    lightgbm_available = True
except ImportError:
    lightgbm_available = False

class PredictionModel:
    """株価予測モデル"""
    
    def __init__(self, db_path='trading_data.db'):
        self.reader = DataReader(db_path)
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量を作成"""
        if df.empty or len(df) < 10:
            return pd.DataFrame()
        
        # 基本的な特徴量
        df = df.copy()
        
        # 価格系の特徴量
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
        
        # 目標変数（次の期間の価格）
        df['target'] = df['close_price'].shift(-1)
        
        return df
    
    def prepare_data(self, symbol: str, period: int = 1000) -> tuple:
        """データを準備"""
        print(f"データ準備中: {symbol}")
        
        # データを取得
        df = self.reader.get_latest_data(symbol, limit=period)
        
        if df.empty:
            print(f"❌ {symbol} のデータが見つかりません")
            return None, None, None, None
        
        print(f"取得データ数: {len(df)}件")
        
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
            print(f"❌ 有効なデータが不足しています ({len(df)}件)")
            return None, None, None, None
        
        # 特徴量と目標変数を分離
        X = df[feature_cols].copy()
        y = df['target'].copy()
        
        # NaNや無限大の値を処理
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        print(f"特徴量数: {len(feature_cols)}")
        print(f"有効サンプル数: {len(X)}")
        
        self.feature_columns = feature_cols
        
        return X, y, df, feature_cols
    
    def train_model(self, X, y, model_type='comparison'):
        """モデルを訓練"""
        if model_type == 'comparison':
            return self.compare_models(X, y)
        elif sklearn_available and model_type == 'advanced':
            return self._train_sklearn_model(X, y)
        else:
            return self._train_simple_model(X, y)
    
    def compare_models(self, X, y):
        """複数のモデルを比較"""
        if not sklearn_available:
            print("scikit-learn が利用できません。基本的なモデルのみ使用します。")
            return self._train_simple_model(X, y)
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        
        # 訓練・テストデータの分割
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'SimpleMovingAverage': None  # 後で実装
        }
        
        # LightGBMを追加（インストール済みの場合）
        if lightgbm_available:
            import lightgbm as lgb
            models['LightGBM'] = lgb.LGBMRegressor(random_state=42, verbose=-1)
        
        results = {}
        
        for name, model in models.items():
            if model is None:
                # 単純移動平均の実装
                pred = [np.mean(y_train[-5:]) for _ in range(len(y_test))]
                mse = mean_squared_error(y_test, pred)
                r2 = r2_score(y_test, pred)
                results[name] = {'mse': mse, 'r2': r2, 'model': None}
                print(f"{name}: MSE={mse:.4f}, R²={r2:.4f}")
            else:
                try:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, pred)
                    r2 = r2_score(y_test, pred)
                    results[name] = {'mse': mse, 'r2': r2, 'model': model}
                    print(f"{name}: MSE={mse:.4f}, R²={r2:.4f}")
                except Exception as e:
                    print(f"{name} エラー: {e}")
                    results[name] = {'mse': float('inf'), 'r2': -float('inf'), 'model': None}
        
        # 最適なモデルを選択
        best_model = min(results.items(), key=lambda x: x[1]['mse'])
        print(f"\n最適なモデル: {best_model[0]}")
        
        return best_model[1]['model']
    
    def _train_sklearn_model(self, X, y):
        """scikit-learnを使用したモデル訓練"""
        print("高度なモデル（Random Forest）を訓練中...")
        
        # データを分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # スケーリング
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # モデル訓練
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # 予測
        y_pred = self.model.predict(X_test_scaled)
        
        # 評価指標
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"MAE: {mae:.4f}")
        
        # 特徴量重要度
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n上位10の重要な特徴量:")
        print(feature_importance.head(10))
        
        return {
            'model_type': 'RandomForest',
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'feature_importance': feature_importance,
            'test_predictions': y_pred,
            'test_actual': y_test.values
        }
    
    def _train_simple_model(self, X, y):
        """シンプルなモデル訓練"""
        print("シンプルなモデル（移動平均ベース）を訓練中...")
        
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
        
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        return {
            'model_type': 'SimpleMovingAverage',
            'mse': mse,
            'mae': mae,
            'test_predictions': y_pred,
            'test_actual': y_test
        }
    
    def predict_next_price(self, symbol: str, periods: int = 5) -> list:
        """次の期間の価格を予測"""
        if self.model is None:
            print("❌ モデルが訓練されていません")
            return []
        
        # 最新データを取得
        df = self.reader.get_latest_data(symbol, limit=100)
        if df.empty:
            return []
        
        # 特徴量を作成
        df = self.create_features(df)
        
        # 最新の特徴量を取得
        if sklearn_available and hasattr(self.model, 'predict'):
            X_latest = df[self.feature_columns].tail(1)
            X_latest = X_latest.replace([np.inf, -np.inf], np.nan).fillna(X_latest.median())
            
            if self.scaler:
                X_latest_scaled = self.scaler.transform(X_latest)
                prediction = self.model.predict(X_latest_scaled)[0]
            else:
                prediction = self.model.predict(X_latest)[0]
        else:
            # シンプルモデルの場合
            prediction = df['close_price'].tail(5).mean()
        
        return [prediction]
    
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
        
        # 次の価格予測
        next_price = self.predict_next_price(symbol)
        if next_price:
            current_price = self.reader.get_latest_data(symbol, limit=1)['close_price'].iloc[0]
            predicted_change = next_price[0] - current_price
            change_pct = (predicted_change / current_price) * 100
            
            report += f"現在価格: {current_price:.2f}\n"
            report += f"予測価格: {next_price[0]:.2f}\n"
            report += f"予測変動: {predicted_change:.2f} ({change_pct:.2f}%)\n"
        
        return report


def main():
    """メイン実行関数"""
    print("=== 株価予測モデル作成 ===")
    print()
    
    # データの確認
    reader = DataReader()
    symbols = reader.get_available_symbols()
    
    if not symbols:
        print("❌ データベースにデータがありません")
        print("まず 'python run_data_collection.py' でデータを収集してください")
        return
    
    print(f"利用可能な銘柄: {', '.join(symbols)}")
    
    # 各銘柄で予測モデルを作成
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"銘柄: {symbol}")
        print('='*50)
        
        model = PredictionModel()
        
        # データ準備
        X, y, df, feature_cols = model.prepare_data(symbol)
        
        if X is None:
            continue
        
        # モデル訓練
        model_type = 'advanced' if sklearn_available else 'simple'
        results = model.train_model(X, y, model_type)
        
        # レポート生成
        report = model.generate_prediction_report(symbol, results)
        print(report)
        
        # 結果をファイルに保存
        with open(f'{symbol}_prediction_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📊 レポートを保存しました: {symbol}_prediction_report.txt")


if __name__ == "__main__":
    main()
