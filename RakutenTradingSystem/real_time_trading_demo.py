"""
高精度リークフリーモデル - リアルタイム取引デモ
既存の訓練済みモデルを使用した実際のデータでの予測デモ
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

# 高精度モデルをインポート
from high_precision_ml_models import HighPrecisionLeakFreeModels

class RealTimeTradingDemo:
    """リアルタイム取引デモ"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.high_precision_model = HighPrecisionLeakFreeModels()
        self.trained_models = {}
        
    def load_trained_models(self, symbols: list):
        """既に訓練済みのモデルを読み込み"""
        self.logger.info("既存の訓練済みモデルを準備中...")
        
        for symbol in symbols:
            try:
                # データ準備
                data_result = self.high_precision_model.prepare_high_precision_data(symbol)
                if data_result[0] is None:
                    self.logger.warning(f"❌ {symbol}: データ準備失敗")
                    continue
                
                X_train, X_test, y_price_train, y_price_test, y_direction_train, y_direction_test = data_result
                
                # モデル訓練（既存のロジック使用）
                models = self.high_precision_model.train_ensemble_models(X_train, y_price_train, y_direction_train)
                
                # 評価
                evaluation = self.high_precision_model.evaluate_ensemble_models(models, X_test, y_price_test, y_direction_test)
                
                # 最良モデルを特定
                best_price_model = min(evaluation['price_models'].items(), key=lambda x: x[1]['mae'])
                best_direction_model = max(evaluation['direction_models'].items(), key=lambda x: x[1]['accuracy'])
                
                self.trained_models[symbol] = {
                    'models': models,
                    'evaluation': evaluation,
                    'best_price': best_price_model,
                    'best_direction': best_direction_model,
                    'feature_columns': self.high_precision_model.feature_columns
                }
                
                self.logger.info(f"✅ {symbol}: モデル準備完了")
                self.logger.info(f"  最良価格予測: {best_price_model[0]} (MAE: {best_price_model[1]['mae']:.6f})")
                self.logger.info(f"  最良方向予測: {best_direction_model[0]} (精度: {best_direction_model[1]['accuracy']:.1%})")
                
            except Exception as e:
                self.logger.error(f"❌ {symbol} モデル準備エラー: {e}")
    
    def get_latest_prediction(self, symbol: str) -> dict:
        """最新データに基づく予測"""
        if symbol not in self.trained_models:
            return {'error': f'{symbol}のモデルが見つかりません'}
        
        try:
            # 最新データを取得
            conn = self.high_precision_model.db_path
            import sqlite3
            conn = sqlite3.connect(self.high_precision_model.db_path)
            
            # 最新50件のデータを取得
            query = '''
                SELECT datetime, open_price, high_price, low_price, close_price, volume
                FROM extended_market_data
                WHERE symbol = ? AND timeframe = '5m'
                ORDER BY datetime DESC
                LIMIT 50
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()
            
            if df.empty:
                return {'error': f'{symbol}の最新データが見つかりません'}
            
            # データを時系列順に並び替え
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # 最新時刻
            latest_time = df['datetime'].iloc[-1]
            latest_price = df['close_price'].iloc[-1]
            
            # 上位時間軸データも取得
            query_1h = '''
                SELECT datetime, open_price, high_price, low_price, close_price, volume
                FROM extended_market_data
                WHERE symbol = ? AND timeframe = '1h'
                ORDER BY datetime DESC
                LIMIT 20
            '''
            
            conn = sqlite3.connect(self.high_precision_model.db_path)
            df_1h = pd.read_sql_query(query_1h, conn, params=(symbol,))
            conn.close()
            df_1d = pd.DataFrame()  # 日足データは省略
            
            # 特徴量作成
            df_features = self.high_precision_model.create_advanced_features(df, df_1h, df_1d, symbol)
            
            if df_features.empty:
                return {'error': '特徴量作成に失敗しました'}
            
            # 最新の特徴量を取得
            feature_cols = self.trained_models[symbol]['feature_columns']
            available_features = [col for col in feature_cols if col in df_features.columns]
            
            latest_features = df_features[available_features].iloc[-1:].values
            
            # 無限大、NaN値を処理
            latest_features = pd.DataFrame(latest_features, columns=available_features)
            latest_features = latest_features.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # 予測実行
            models_data = self.trained_models[symbol]['models']
            best_price_name = self.trained_models[symbol]['best_price'][0]
            best_direction_name = self.trained_models[symbol]['best_direction'][0]
            
            # 価格予測
            price_model_data = models_data['price_models'][best_price_name]
            price_scaler = price_model_data['scaler']
            price_selector = price_model_data['selector']
            price_model = price_model_data['model']
            
            X_selected = price_selector.transform(latest_features)
            X_scaled = price_scaler.transform(X_selected)
            price_prediction = price_model.predict(X_scaled)[0]
            
            # 方向予測
            direction_model_data = models_data['direction_models'][best_direction_name]
            direction_scaler = direction_model_data['scaler']
            direction_selector = direction_model_data['selector']
            direction_model = direction_model_data['model']
            
            X_direction_selected = direction_selector.transform(latest_features)
            X_direction_scaled = direction_scaler.transform(X_direction_selected)
            direction_prediction = direction_model.predict(X_direction_scaled)[0]
            direction_probability = direction_model.predict_proba(X_direction_scaled)[0]
            direction_confidence = max(direction_probability)
            
            # 予測価格計算
            predicted_price = latest_price * (1 + price_prediction)
            price_change_percent = price_prediction * 100
            
            # 取引推奨度計算
            model_performance = self.trained_models[symbol]['evaluation']
            price_mae = self.trained_models[symbol]['best_price'][1]['mae']
            direction_accuracy = self.trained_models[symbol]['best_direction'][1]['accuracy']
            
            # 取引信号生成
            trade_signal = self._generate_trade_signal(
                direction_prediction, direction_confidence, 
                price_change_percent, price_mae, direction_accuracy
            )
            
            result = {
                'symbol': symbol,
                'datetime': latest_time,
                'current_price': latest_price,
                'predicted_price': predicted_price,
                'price_change_percent': price_change_percent,
                'direction': '上昇' if direction_prediction == 1 else '下降',
                'direction_confidence': direction_confidence,
                'trade_signal': trade_signal,
                'model_performance': {
                    'price_mae': price_mae,
                    'direction_accuracy': direction_accuracy
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"予測エラー ({symbol}): {e}")
            return {'error': f'予測エラー: {e}'}
    
    def _generate_trade_signal(self, direction, confidence, price_change_percent, mae, accuracy):
        """取引信号生成"""
        # 基本条件
        high_confidence = confidence > 0.6
        significant_change = abs(price_change_percent) > 0.1
        good_accuracy = accuracy > 0.55
        low_error = mae < 0.001
        
        if high_confidence and significant_change and good_accuracy:
            if direction == 1 and price_change_percent > 0.2:
                return {
                    'action': 'STRONG_BUY',
                    'confidence': confidence,
                    'reason': f'強い上昇信号 ({price_change_percent:.2f}%予測)'
                }
            elif direction == 1 and price_change_percent > 0:
                return {
                    'action': 'BUY',
                    'confidence': confidence,
                    'reason': f'上昇信号 ({price_change_percent:.2f}%予測)'
                }
            elif direction == 0 and price_change_percent < -0.2:
                return {
                    'action': 'STRONG_SELL',
                    'confidence': confidence,
                    'reason': f'強い下降信号 ({price_change_percent:.2f}%予測)'
                }
            elif direction == 0 and price_change_percent < 0:
                return {
                    'action': 'SELL',
                    'confidence': confidence,
                    'reason': f'下降信号 ({price_change_percent:.2f}%予測)'
                }
        
        return {
            'action': 'HOLD',
            'confidence': confidence,
            'reason': ' 明確な信号なし、様子見推奨'
        }
    
    def run_live_demo(self, symbols: list, interval_minutes: int = 5):
        """ライブデモ実行"""
        self.logger.info("🚀 高精度リークフリーモデル - ライブ取引デモ開始")
        self.logger.info("="*70)
        
        # モデル準備
        self.load_trained_models(symbols)
        
        if not self.trained_models:
            self.logger.error("❌ 使用可能なモデルがありません")
            return
        
        # 現在時刻での予測実行
        current_time = datetime.now()
        self.logger.info(f"📊 予測実行時刻: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("")
        
        for symbol in symbols:
            if symbol not in self.trained_models:
                continue
                
            self.logger.info(f"🔍 {symbol} 分析中...")
            
            prediction = self.get_latest_prediction(symbol)
            
            if 'error' in prediction:
                self.logger.error(f"❌ {symbol}: {prediction['error']}")
                continue
            
            # 結果表示
            self.logger.info(f"📈 {symbol} 予測結果:")
            self.logger.info(f"  現在価格: ¥{prediction['current_price']:.2f}")
            self.logger.info(f"  予測価格: ¥{prediction['predicted_price']:.2f}")
            self.logger.info(f"  変化率: {prediction['price_change_percent']:+.3f}%")
            self.logger.info(f"  方向: {prediction['direction']}")
            self.logger.info(f"  信頼度: {prediction['direction_confidence']:.1%}")
            
            # 取引信号
            signal = prediction['trade_signal']
            action_emoji = {
                'STRONG_BUY': '🚀',
                'BUY': '📈',
                'HOLD': '⏸️',
                'SELL': '📉',
                'STRONG_SELL': '🔻'
            }
            
            self.logger.info(f"  {action_emoji.get(signal['action'], '❓')} 取引信号: {signal['action']}")
            self.logger.info(f"  理由: {signal['reason']}")
            
            # モデル性能
            perf = prediction['model_performance']
            self.logger.info(f"  モデル性能: MAE={perf['price_mae']:.6f}, 方向精度={perf['direction_accuracy']:.1%}")
            self.logger.info("")
        
        self.logger.info("✅ ライブデモ完了")
        self.logger.info("="*70)

def main():
    """メイン実行"""
    # ログ設定
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('real_time_demo.log', encoding='utf-8')
        ]
    )
    
    # デモ実行
    demo = RealTimeTradingDemo()
    
    # テスト用銘柄（高精度モデルで良い結果が出た銘柄）
    symbols = ['7203', '6758', '8306']
    
    demo.run_live_demo(symbols)

if __name__ == "__main__":
    main()
