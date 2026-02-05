"""
データベースから取得したデータの分析ツール
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.data_reader import DataReader
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataAnalyzer:
    """データ分析クラス"""
    
    def __init__(self, db_path='trading_data.db'):
        self.reader = DataReader(db_path)
    
    def calculate_technical_indicators(self, symbol: str, period: int = 100) -> pd.DataFrame:
        """テクニカル指標を計算"""
        df = self.reader.get_latest_data(symbol, limit=period)
        
        if df.empty or len(df) < 5:
            return pd.DataFrame()
        
        # 移動平均線
        df['sma_5'] = df['close_price'].rolling(window=5).mean()
        df['sma_20'] = df['close_price'].rolling(window=20).mean()
        
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
        
        # ボリンジャーバンド
        df['bb_middle'] = df['close_price'].rolling(window=20).mean()
        bb_std = df['close_price'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # 出来高移動平均
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        return df
    
    def detect_patterns(self, symbol: str, period: int = 50) -> dict:
        """チャートパターンを検出"""
        df = self.reader.get_latest_data(symbol, limit=period)
        
        if df.empty or len(df) < 10:
            return {}
        
        patterns = {}
        
        # 最新価格情報
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 価格変動
        price_change = latest['close_price'] - prev['close_price']
        price_change_pct = (price_change / prev['close_price']) * 100
        
        patterns['price_change'] = price_change
        patterns['price_change_pct'] = round(price_change_pct, 2)
        
        # 出来高分析
        avg_volume = df['volume'].tail(10).mean()
        patterns['volume_ratio'] = round(latest['volume'] / avg_volume, 2)
        
        # 高値・安値分析
        high_10 = df['high_price'].tail(10).max()
        low_10 = df['low_price'].tail(10).min()
        current_position = (latest['close_price'] - low_10) / (high_10 - low_10)
        patterns['position_in_range'] = round(current_position * 100, 1)
        
        # トレンド分析
        if len(df) >= 5:
            recent_closes = df['close_price'].tail(5).values
            if all(recent_closes[i] > recent_closes[i-1] for i in range(1, len(recent_closes))):
                patterns['trend'] = '上昇トレンド'
            elif all(recent_closes[i] < recent_closes[i-1] for i in range(1, len(recent_closes))):
                patterns['trend'] = '下降トレンド'
            else:
                patterns['trend'] = '横ばい'
        
        return patterns
    
    def compare_symbols(self, symbols: list, period: int = 30) -> pd.DataFrame:
        """複数銘柄の比較分析"""
        comparison_data = []
        
        for symbol in symbols:
            df = self.reader.get_latest_data(symbol, limit=period)
            if df.empty:
                continue
            
            # 統計情報
            stats = {
                'symbol': symbol,
                'data_count': len(df),
                'avg_price': df['close_price'].mean(),
                'volatility': df['close_price'].std(),
                'max_price': df['high_price'].max(),
                'min_price': df['low_price'].min(),
                'avg_volume': df['volume'].mean(),
                'total_volume': df['volume'].sum()
            }
            
            # 価格変動率
            if len(df) >= 2:
                price_change = (df['close_price'].iloc[-1] - df['close_price'].iloc[0]) / df['close_price'].iloc[0] * 100
                stats['price_change_pct'] = price_change
            
            comparison_data.append(stats)
        
        return pd.DataFrame(comparison_data)
    
    def generate_report(self, symbol: str) -> str:
        """銘柄レポート生成"""
        report = f"=== {symbol} 分析レポート ===\n\n"
        
        # 基本統計
        stats = self.reader.get_data_statistics(symbol)
        if stats['data_count'] == 0:
            return f"{symbol}: データがありません"
        
        report += f"データ件数: {stats['data_count']}\n"
        report += f"平均価格: {stats['avg_price']}\n"
        report += f"価格レンジ: {stats['min_price']} ～ {stats['max_price']}\n"
        report += f"平均出来高: {stats['avg_volume']:,}\n\n"
        
        # テクニカル分析
        df = self.calculate_technical_indicators(symbol)
        if not df.empty:
            latest = df.iloc[-1]
            report += "テクニカル指標 (最新):\n"
            report += f"  SMA5: {latest['sma_5']:.2f}\n"
            report += f"  SMA20: {latest['sma_20']:.2f}\n"
            if not pd.isna(latest['rsi']):
                report += f"  RSI: {latest['rsi']:.2f}\n"
            if not pd.isna(latest['macd']):
                report += f"  MACD: {latest['macd']:.2f}\n"
            report += "\n"
        
        # パターン分析
        patterns = self.detect_patterns(symbol)
        if patterns:
            report += "パターン分析:\n"
            report += f"  価格変動: {patterns['price_change']:.2f} ({patterns['price_change_pct']}%)\n"
            report += f"  出来高倍率: {patterns['volume_ratio']}倍\n"
            report += f"  レンジ内位置: {patterns['position_in_range']}%\n"
            report += f"  トレンド: {patterns['trend']}\n"
        
        return report


def main():
    """メイン実行関数"""
    analyzer = DataAnalyzer()
    
    print("=== データ分析ツール ===")
    print()
    
    # 利用可能な銘柄を取得
    symbols = analyzer.reader.get_available_symbols()
    if not symbols:
        print("❌ データベースにデータがありません")
        return
    
    print(f"利用可能な銘柄: {', '.join(symbols)}")
    print()
    
    # 各銘柄の分析レポートを生成
    for symbol in symbols:
        print(analyzer.generate_report(symbol))
        print("-" * 50)
    
    # 複数銘柄比較
    if len(symbols) > 1:
        print("\n=== 銘柄比較 ===")
        comparison = analyzer.compare_symbols(symbols)
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
