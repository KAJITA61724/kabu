"""
実用的な株価予測モデル - 少量データ対応版
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

class PracticalPredictionModel:
    """実用的な株価予測モデル"""
    
    def __init__(self, db_path='trading_data.db'):
        self.reader = DataReader(db_path)
        self.model_results = {}
        
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本的な特徴量を作成（少量データ対応）"""
        if df.empty or len(df) < 2:
            return pd.DataFrame()
        
        df = df.copy()
        
        # 価格変動率
        df['price_change'] = df['close_price'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # 価格比率
        df['high_low_ratio'] = df['high_price'] / df['low_price']
        df['open_close_ratio'] = df['open_price'] / df['close_price']
        
        # 出来高分析
        df['volume_change'] = df['volume'].pct_change()
        df['price_volume_ratio'] = df['close_price'] / df['volume'] * 1000000  # 正規化
        
        # トレンド分析（2期間移動平均）
        if len(df) >= 2:
            df['price_trend'] = df['close_price'].rolling(window=2).mean()
            df['volume_trend'] = df['volume'].rolling(window=2).mean()
        
        # 時間系特徴量
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
        df['minute'] = pd.to_datetime(df['datetime']).dt.minute
        
        return df
    
    def simple_trend_prediction(self, symbol: str) -> dict:
        """シンプルなトレンド予測"""
        # データを取得
        df = self.reader.get_latest_data(symbol, limit=10)
        
        if df.empty or len(df) < 2:
            return {'error': 'データが不足しています'}
        
        # 特徴量作成
        df = self.create_basic_features(df)
        
        # 現在の状況分析
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 基本統計
        analysis = {
            'symbol': symbol,
            'current_price': latest['close_price'],
            'previous_price': prev['close_price'],
            'price_change': latest['close_price'] - prev['close_price'],
            'price_change_pct': ((latest['close_price'] - prev['close_price']) / prev['close_price']) * 100,
            'volume_change_pct': ((latest['volume'] - prev['volume']) / prev['volume']) * 100 if prev['volume'] > 0 else 0,
            'trend_direction': 'up' if latest['close_price'] > prev['close_price'] else 'down',
            'volatility': df['price_change_abs'].mean() * 100,
            'data_points': len(df)
        }
        
        # トレンド強度計算
        if len(df) >= 3:
            recent_changes = df['price_change'].tail(3)
            if recent_changes.mean() > 0:
                analysis['trend_strength'] = 'strong_up' if recent_changes.mean() > 0.01 else 'weak_up'
            else:
                analysis['trend_strength'] = 'strong_down' if recent_changes.mean() < -0.01 else 'weak_down'
        else:
            analysis['trend_strength'] = 'neutral'
        
        # 出来高分析
        if latest['volume'] > df['volume'].mean():
            analysis['volume_signal'] = 'high'
        else:
            analysis['volume_signal'] = 'low'
        
        # 予測信頼度
        confidence_factors = []
        
        # データ量
        if len(df) >= 5:
            confidence_factors.append(0.3)
        elif len(df) >= 3:
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.1)
        
        # トレンドの一貫性
        if len(df) >= 3:
            trend_consistency = abs(df['price_change'].tail(3).mean()) / df['price_change'].tail(3).std()
            if trend_consistency > 1:
                confidence_factors.append(0.3)
            else:
                confidence_factors.append(0.1)
        
        # 出来高の支援
        if analysis['volume_signal'] == 'high':
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.1)
        
        analysis['confidence'] = min(sum(confidence_factors), 1.0)
        
        return analysis
    
    def moving_average_prediction(self, symbol: str, periods: int = 3) -> dict:
        """移動平均ベースの予測"""
        df = self.reader.get_latest_data(symbol, limit=max(periods, 5))
        
        if df.empty or len(df) < 2:
            return {'error': 'データが不足しています'}
        
        # 移動平均計算
        if len(df) >= periods:
            ma = df['close_price'].tail(periods).mean()
            current_price = df['close_price'].iloc[-1]
            
            # 移動平均からの乖離
            deviation = (current_price - ma) / ma * 100
            
            # 次期予測（単純移動平均）
            if len(df) >= periods + 1:
                next_prediction = df['close_price'].tail(periods).mean()
            else:
                next_prediction = ma
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'moving_average': ma,
                'deviation_pct': deviation,
                'predicted_price': next_prediction,
                'prediction_change': next_prediction - current_price,
                'prediction_change_pct': ((next_prediction - current_price) / current_price) * 100,
                'signal': 'buy' if deviation < -2 else 'sell' if deviation > 2 else 'hold'
            }
        
        return {'error': 'データが不足しています'}
    
    def momentum_analysis(self, symbol: str) -> dict:
        """モメンタム分析"""
        df = self.reader.get_latest_data(symbol, limit=10)
        
        if df.empty or len(df) < 3:
            return {'error': 'データが不足しています'}
        
        # 価格変動率計算
        df['price_change'] = df['close_price'].pct_change()
        df['cumulative_return'] = (1 + df['price_change']).cumprod() - 1
        
        # モメンタム指標
        momentum_3 = df['cumulative_return'].iloc[-1] - df['cumulative_return'].iloc[-4] if len(df) >= 4 else 0
        momentum_5 = df['cumulative_return'].iloc[-1] - df['cumulative_return'].iloc[-6] if len(df) >= 6 else 0
        
        # 出来高加重平均価格（VWAP）の簡易版
        if len(df) >= 3:
            vwap = (df['close_price'] * df['volume']).sum() / df['volume'].sum()
            vwap_signal = 'above' if df['close_price'].iloc[-1] > vwap else 'below'
        else:
            vwap = df['close_price'].mean()
            vwap_signal = 'neutral'
        
        return {
            'symbol': symbol,
            'momentum_3': momentum_3 * 100,
            'momentum_5': momentum_5 * 100,
            'vwap': vwap,
            'vwap_signal': vwap_signal,
            'momentum_signal': 'positive' if momentum_3 > 0 else 'negative'
        }
    
    def comprehensive_analysis(self, symbol: str) -> dict:
        """包括的な分析"""
        print(f"\n{'='*60}")
        print(f"包括的分析: {symbol}")
        print('='*60)
        
        # 各分析を実行
        trend_analysis = self.simple_trend_prediction(symbol)
        ma_analysis = self.moving_average_prediction(symbol)
        momentum_analysis = self.momentum_analysis(symbol)
        
        # 統合分析
        comprehensive = {
            'symbol': symbol,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'trend_analysis': trend_analysis,
            'moving_average_analysis': ma_analysis,
            'momentum_analysis': momentum_analysis
        }
        
        # 統合シグナル
        signals = []
        
        # トレンド分析からのシグナル
        if 'trend_direction' in trend_analysis:
            if trend_analysis['trend_direction'] == 'up' and trend_analysis['confidence'] > 0.5:
                signals.append('buy')
            elif trend_analysis['trend_direction'] == 'down' and trend_analysis['confidence'] > 0.5:
                signals.append('sell')
            else:
                signals.append('hold')
        
        # 移動平均からのシグナル
        if 'signal' in ma_analysis:
            signals.append(ma_analysis['signal'])
        
        # モメンタムからのシグナル
        if 'momentum_signal' in momentum_analysis:
            if momentum_analysis['momentum_signal'] == 'positive':
                signals.append('buy')
            else:
                signals.append('sell')
        
        # 最終シグナル（多数決）
        buy_count = signals.count('buy')
        sell_count = signals.count('sell')
        hold_count = signals.count('hold')
        
        if buy_count > sell_count and buy_count > hold_count:
            final_signal = 'BUY'
        elif sell_count > buy_count and sell_count > hold_count:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'
        
        comprehensive['final_signal'] = final_signal
        comprehensive['signal_confidence'] = max(buy_count, sell_count, hold_count) / len(signals)
        
        return comprehensive
    
    def generate_report(self, symbol: str) -> str:
        """分析レポート生成"""
        analysis = self.comprehensive_analysis(symbol)
        
        report = f"""
=== {symbol} 株価予測レポート ===
生成時刻: {analysis['analysis_time']}

【トレンド分析】
現在価格: {analysis['trend_analysis'].get('current_price', 'N/A')}
価格変動: {analysis['trend_analysis'].get('price_change_pct', 'N/A'):.2f}%
トレンド方向: {analysis['trend_analysis'].get('trend_direction', 'N/A')}
トレンド強度: {analysis['trend_analysis'].get('trend_strength', 'N/A')}
信頼度: {analysis['trend_analysis'].get('confidence', 0):.1%}

【移動平均分析】
移動平均: {analysis['moving_average_analysis'].get('moving_average', 'N/A')}
乖離率: {analysis['moving_average_analysis'].get('deviation_pct', 'N/A'):.2f}%
予測価格: {analysis['moving_average_analysis'].get('predicted_price', 'N/A')}
MA信号: {analysis['moving_average_analysis'].get('signal', 'N/A')}

【モメンタム分析】
3期間モメンタム: {analysis['momentum_analysis'].get('momentum_3', 'N/A'):.2f}%
5期間モメンタム: {analysis['momentum_analysis'].get('momentum_5', 'N/A'):.2f}%
VWAP: {analysis['momentum_analysis'].get('vwap', 'N/A')}
VWAP信号: {analysis['momentum_analysis'].get('vwap_signal', 'N/A')}

【統合判定】
最終信号: {analysis['final_signal']}
信号信頼度: {analysis['signal_confidence']:.1%}

【推奨アクション】
"""
        
        if analysis['final_signal'] == 'BUY':
            report += "✅ 買い推奨 - 上昇トレンドが期待されます"
        elif analysis['final_signal'] == 'SELL':
            report += "❌ 売り推奨 - 下降トレンドが期待されます"
        else:
            report += "⏸️ 様子見推奨 - 明確なトレンドが見られません"
        
        return report


def main():
    """メイン実行関数"""
    print("=== 実用的株価予測システム ===")
    print()
    
    model = PracticalPredictionModel()
    
    # 利用可能な銘柄を確認
    symbols = model.reader.get_available_symbols()
    
    if not symbols:
        print("❌ データベースにデータがありません")
        return
    
    print(f"利用可能な銘柄: {', '.join(symbols)}")
    
    # 各銘柄の分析
    for symbol in symbols:
        report = model.generate_report(symbol)
        print(report)
        
        # レポートをファイルに保存
        with open(f'{symbol}_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📊 レポートを保存しました: {symbol}_analysis_report.txt")
        print("-" * 80)


if __name__ == "__main__":
    main()
