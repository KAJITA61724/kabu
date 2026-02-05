"""
yfinanceデータを使った予測モデル
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

# yfinanceデータ収集
from yfinance_data_test import YFinanceDataCollector

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    sklearn_available = True
except ImportError:
    sklearn_available = False

class YFinancePredictionModel:
    """yfinanceデータを使った予測モデル"""
    
    def __init__(self, db_path='trading_data.db'):
        self.reader = DataReader(db_path)
        self.collector = YFinanceDataCollector(db_path)
        self.models = {}
        self.scalers = {}
        
    def collect_training_data(self, symbols: list, days: int = 5) -> bool:
        """訓練用データを収集"""
        print(f"訓練データ収集開始: {len(symbols)}銘柄, {days}日間")
        
        results = self.collector.test_multiple_symbols(symbols, days_back=days)
        
        success_count = 0
        total_data = 0
        
        for symbol, result in results.items():
            if 'error' not in result:
                success_count += 1
                total_data += result['data_count']
                print(f"✅ {symbol}: {result['data_count']}件")
            else:
                print(f"❌ {symbol}: {result['error']}")
        
        print(f"\n収集結果: {success_count}/{len(symbols)}銘柄, 総データ{total_data}件")
        return success_count > 0
        
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """高度な特徴量を作成"""
        if df.empty or len(df) < 20:
            return pd.DataFrame()
        
        df = df.copy()
        df = df.sort_values('datetime')
        
        # 基本的な価格特徴量
        df['price_change'] = df['close_price'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        df['high_low_ratio'] = df['high_price'] / df['low_price']
        df['open_close_ratio'] = df['open_price'] / df['close_price']
        
        # 移動平均
        for window in [5, 10, 20]:
            df[f'sma_{window}'] = df['close_price'].rolling(window=window).mean()
            df[f'sma_ratio_{window}'] = df['close_price'] / df[f'sma_{window}']
        
        # ボラティリティ
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['close_price'].rolling(window=window).std()
            df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / df['close_price']
        
        # 出来高特徴量
        df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio_5'] = df['volume'] / df['volume_sma_5']
        df['volume_ratio_20'] = df['volume'] / df['volume_sma_20']
        df['price_volume_ratio'] = df['close_price'] / df['volume'] * 1000000
        
        # RSI
        delta = df['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close_price'].ewm(span=12).mean()
        exp2 = df['close_price'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # ボリンジャーバンド
        df['bb_middle'] = df['close_price'].rolling(window=20).mean()
        bb_std = df['close_price'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close_price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ラグ特徴量
        for lag in [1, 2, 3, 5, 10]:
            df[f'price_lag_{lag}'] = df['close_price'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'change_lag_{lag}'] = df['price_change'].shift(lag)
        
        # 時間特徴量
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
        df['minute'] = pd.to_datetime(df['datetime']).dt.minute
        df['day_of_week'] = pd.to_datetime(df['datetime']).dt.dayofweek
        df['is_morning'] = (df['hour'] < 12).astype(int)
        df['is_afternoon'] = (df['hour'] >= 12).astype(int)
        
        # 目標変数（5分後の価格変動率）
        df['target'] = df['close_price'].shift(-1) / df['close_price'] - 1
        
        return df
    
    def train_model(self, symbol: str, model_type='random_forest') -> dict:
        """個別銘柄のモデル訓練"""
        print(f"\n{'='*60}")
        print(f"モデル訓練開始: {symbol}")
        print('='*60)
        
        # データ取得
        df = self.reader.get_latest_data(symbol, limit=1000)
        
        if df.empty:
            return {'error': f'{symbol}のデータがありません'}
        
        print(f"データ取得: {len(df)}件")
        
        # 特徴量作成
        df = self.create_advanced_features(df)
        
        # 特徴量選択
        feature_cols = [
            'price_change', 'price_change_abs', 'high_low_ratio', 'open_close_ratio',
            'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20',
            'volatility_ratio_5', 'volatility_ratio_10', 'volatility_ratio_20',
            'volume_ratio_5', 'volume_ratio_20', 'price_volume_ratio',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position',
            'price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_5', 'price_lag_10',
            'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5', 'volume_lag_10',
            'change_lag_1', 'change_lag_2', 'change_lag_3', 'change_lag_5', 'change_lag_10',
            'hour', 'minute', 'day_of_week', 'is_morning', 'is_afternoon'
        ]
        
        # 欠損値処理
        df = df.dropna()
        
        if len(df) < 50:
            return {'error': f'{symbol}の有効データが不足 ({len(df)}件)'}
        
        # 特徴量とターゲット分離
        X = df[feature_cols].copy()
        y = df['target'].copy()
        
        # 異常値処理
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        print(f"訓練データ: {len(X_train)}件, テストデータ: {len(X_test)}件")
        
        # スケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # モデル訓練
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()
        
        model.fit(X_train_scaled, y_train)
        
        # 予測
        y_pred = model.predict(X_test_scaled)
        
        # 評価指標
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # 方向性の正確度
        direction_accuracy = np.mean((y_test > 0) == (y_pred > 0))
        
        results = {
            'symbol': symbol,
            'model_type': model_type,
            'data_count': len(df),
            'train_count': len(X_train),
            'test_count': len(X_test),
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'direction_accuracy': direction_accuracy,
            'feature_count': len(feature_cols)
        }
        
        # 特徴量重要度（Random Forestの場合）
        if model_type == 'random_forest':
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            results['feature_importance'] = feature_importance.head(10)
            
            print("\n重要な特徴量 (上位10):")
            print(feature_importance.head(10).to_string(index=False))
        
        # モデルとスケーラー保存
        self.models[symbol] = model
        self.scalers[symbol] = scaler
        
        print(f"\n評価結果:")
        print(f"  MSE: {mse:.6f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  方向性正確度: {direction_accuracy:.1%}")
        
        return results
    
    def predict_next_move(self, symbol: str, periods: int = 1) -> dict:
        """次の動きを予測"""
        if symbol not in self.models:
            return {'error': f'{symbol}のモデルがありません'}
        
        # 最新データ取得
        df = self.reader.get_latest_data(symbol, limit=100)
        df = self.create_advanced_features(df)
        
        if df.empty:
            return {'error': 'データがありません'}
        
        # 最新の特徴量
        feature_cols = [
            'price_change', 'price_change_abs', 'high_low_ratio', 'open_close_ratio',
            'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20',
            'volatility_ratio_5', 'volatility_ratio_10', 'volatility_ratio_20',
            'volume_ratio_5', 'volume_ratio_20', 'price_volume_ratio',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position',
            'price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_5', 'price_lag_10',
            'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5', 'volume_lag_10',
            'change_lag_1', 'change_lag_2', 'change_lag_3', 'change_lag_5', 'change_lag_10',
            'hour', 'minute', 'day_of_week', 'is_morning', 'is_afternoon'
        ]
        
        X_latest = df[feature_cols].tail(1)
        X_latest = X_latest.replace([np.inf, -np.inf], np.nan).fillna(X_latest.median())
        
        # 予測
        model = self.models[symbol]
        scaler = self.scalers[symbol]
        
        X_scaled = scaler.transform(X_latest)
        prediction = model.predict(X_scaled)[0]
        
        current_price = df['close_price'].iloc[-1]
        predicted_price = current_price * (1 + prediction)
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'predicted_change_pct': prediction * 100,
            'predicted_price': predicted_price,
            'prediction_direction': 'UP' if prediction > 0 else 'DOWN',
            'confidence': abs(prediction) * 100
        }
    
    def generate_trading_signals(self, symbols: list) -> pd.DataFrame:
        """トレーディングシグナル生成"""
        signals = []
        
        for symbol in symbols:
            if symbol in self.models:
                prediction = self.predict_next_move(symbol)
                
                if 'error' not in prediction:
                    # シグナル強度計算
                    confidence = prediction['confidence']
                    direction = prediction['prediction_direction']
                    
                    if confidence > 0.5:  # 0.5%以上の変動予測
                        if direction == 'UP':
                            signal = 'BUY'
                        else:
                            signal = 'SELL'
                    else:
                        signal = 'HOLD'
                    
                    signals.append({
                        'symbol': symbol,
                        'signal': signal,
                        'confidence': confidence,
                        'predicted_change': prediction['predicted_change_pct'],
                        'current_price': prediction['current_price'],
                        'predicted_price': prediction['predicted_price']
                    })
        
        return pd.DataFrame(signals)


def main():
    """メイン実行"""
    print("=== yfinanceデータ予測モデル ===")
    print()
    
    # 主要銘柄
    symbols = ['7203', '6758', '8306', '9984', '6861']
    
    model = YFinancePredictionModel()
    
    # 1. データ収集
    print("1. 訓練データ収集")
    if not model.collect_training_data(symbols, days=5):
        print("❌ データ収集に失敗しました")
        return
    
    # 2. モデル訓練
    print("\n2. モデル訓練")
    results = {}
    for symbol in symbols:
        result = model.train_model(symbol, 'random_forest')
        results[symbol] = result
    
    # 3. 訓練結果サマリー
    print("\n" + "="*80)
    print("訓練結果サマリー")
    print("="*80)
    
    for symbol, result in results.items():
        if 'error' in result:
            print(f"❌ {symbol}: {result['error']}")
        else:
            print(f"✅ {symbol}:")
            print(f"   データ件数: {result['data_count']}")
            print(f"   R²: {result['r2']:.4f}")
            print(f"   MAE: {result['mae']:.6f}")
            print(f"   方向性正確度: {result['direction_accuracy']:.1%}")
    
    # 4. 予測とシグナル生成
    print("\n" + "="*80)
    print("現在の予測とシグナル")
    print("="*80)
    
    signals = model.generate_trading_signals(symbols)
    
    if not signals.empty:
        print(signals.to_string(index=False))
        
        # 強い買い/売りシグナル
        strong_signals = signals[signals['confidence'] > 1.0]
        if not strong_signals.empty:
            print("\n強いシグナル (信頼度1%以上):")
            print(strong_signals.to_string(index=False))
    
    print("\n処理完了")


if __name__ == "__main__":
    main()
