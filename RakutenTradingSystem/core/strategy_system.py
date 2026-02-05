"""
TradingViewスタイル戦略システム - 方法2の実装
- 複数の戦略ベースシグナル
- 戦略統合システム
- 予測との整合性チェック
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

class StrategySignal(Enum):
    """戦略シグナル"""
    BUY = 1
    SELL = -1
    HOLD = 0

@dataclass
class Strategy:
    """戦略定義"""
    name: str
    weight: float
    min_confidence: float
    
@dataclass
class StrategyResult:
    """戦略実行結果"""
    strategy_name: str
    signal: StrategySignal
    confidence: float
    entry_price: float
    timestamp: datetime

class TradingViewStrategies:
    """TradingViewスタイル戦略システム"""
    
    def __init__(self, db_path: str = "enhanced_trading.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # 戦略定義
        self.strategies = {
            'rsi_oversold': Strategy("RSI過売り", 1.0, 0.7),
            'rsi_overbought': Strategy("RSI過買い", 1.0, 0.7),
            'macd_crossover': Strategy("MACDクロスオーバー", 1.2, 0.6),
            'bollinger_squeeze': Strategy("ボリンジャースクイーズ", 0.8, 0.8),
            'volume_breakout': Strategy("出来高ブレイクアウト", 1.1, 0.75),
            'moving_average_trend': Strategy("移動平均トレンド", 0.9, 0.65),
            'support_resistance': Strategy("サポート・レジスタンス", 1.3, 0.8)
        }
        
        # 戦略結果履歴
        self.strategy_history = []
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI計算"""
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[float, float, float]:
        """MACD計算"""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0
        
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        return (
            macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0.0,
            signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else 0.0,
            histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0.0
        )
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[float, float, float]:
        """ボリンジャーバンド計算"""
        if len(prices) < period:
            current_price = prices.iloc[-1]
            return current_price, current_price * 1.02, current_price * 0.98
        
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        
        return (
            sma.iloc[-1] if not pd.isna(sma.iloc[-1]) else prices.iloc[-1],
            upper.iloc[-1] if not pd.isna(upper.iloc[-1]) else prices.iloc[-1] * 1.02,
            lower.iloc[-1] if not pd.isna(lower.iloc[-1]) else prices.iloc[-1] * 0.98
        )
    
    def get_market_data(self, symbol: str, current_time: datetime, lookback_hours: int = 24) -> Optional[pd.DataFrame]:
        """市場データ取得"""
        try:
            conn = sqlite3.connect(self.db_path)
            start_time = current_time - timedelta(hours=lookback_hours)
            
            query = '''
                SELECT 
                    timestamp,
                    close_price,
                    volume,
                    ma_5min,
                    ma_20min,
                    ma_60min
                FROM five_minute_data
                WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol, start_time, current_time))
            conn.close()
            
            if len(df) < 12:  # 最低1時間分のデータ
                return None
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
            
        except Exception as e:
            self.logger.error(f"市場データ取得エラー: {e}")
            return None
    
    def rsi_strategy(self, symbol: str, current_time: datetime) -> Optional[StrategyResult]:
        """RSI戦略"""
        try:
            df = self.get_market_data(symbol, current_time, 6)  # 6時間
            if df is None or len(df) < 15:
                return None
            
            rsi = self.calculate_rsi(df['close_price'])
            current_price = df['close_price'].iloc[-1]
            
            signal = StrategySignal.HOLD
            confidence = 0.5
            
            if rsi < 30:  # 過売り
                signal = StrategySignal.BUY
                confidence = min(0.9, (30 - rsi) / 30 + 0.6)
            elif rsi > 70:  # 過買い
                signal = StrategySignal.SELL
                confidence = min(0.9, (rsi - 70) / 30 + 0.6)
            
            strategy_name = 'rsi_oversold' if signal == StrategySignal.BUY else 'rsi_overbought'
            
            return StrategyResult(
                strategy_name=strategy_name,
                signal=signal,
                confidence=confidence,
                entry_price=current_price,
                timestamp=current_time
            )
            
        except Exception as e:
            self.logger.error(f"RSI戦略エラー: {e}")
            return None
    
    def macd_strategy(self, symbol: str, current_time: datetime) -> Optional[StrategyResult]:
        """MACD戦略"""
        try:
            df = self.get_market_data(symbol, current_time, 12)  # 12時間
            if df is None or len(df) < 26:
                return None
            
            macd, signal_line, histogram = self.calculate_macd(df['close_price'])
            current_price = df['close_price'].iloc[-1]
            
            # 前のMACDとシグナルライン
            if len(df) > 1:
                prev_macd, prev_signal, _ = self.calculate_macd(df['close_price'].iloc[:-1])
            else:
                prev_macd, prev_signal = macd, signal_line
            
            signal = StrategySignal.HOLD
            confidence = 0.5
            
            # ゴールデンクロス/デッドクロス
            if prev_macd <= prev_signal and macd > signal_line:  # ゴールデンクロス
                signal = StrategySignal.BUY
                confidence = min(0.9, abs(macd - signal_line) * 100 + 0.6)
            elif prev_macd >= prev_signal and macd < signal_line:  # デッドクロス
                signal = StrategySignal.SELL
                confidence = min(0.9, abs(macd - signal_line) * 100 + 0.6)
            
            return StrategyResult(
                strategy_name='macd_crossover',
                signal=signal,
                confidence=confidence,
                entry_price=current_price,
                timestamp=current_time
            )
            
        except Exception as e:
            self.logger.error(f"MACD戦略エラー: {e}")
            return None
    
    def bollinger_strategy(self, symbol: str, current_time: datetime) -> Optional[StrategyResult]:
        """ボリンジャーバンド戦略"""
        try:
            df = self.get_market_data(symbol, current_time, 8)  # 8時間
            if df is None or len(df) < 20:
                return None
            
            sma, upper, lower = self.calculate_bollinger_bands(df['close_price'])
            current_price = df['close_price'].iloc[-1]
            
            signal = StrategySignal.HOLD
            confidence = 0.5
            
            # バンド幅（スクイーズ検出）
            band_width = (upper - lower) / sma
            avg_band_width = ((df['close_price'].rolling(20).std() * 2) / df['close_price'].rolling(20).mean()).mean()
            
            if band_width < avg_band_width * 0.8:  # スクイーズ状態
                # 価格がバンドを突破
                if current_price > upper:
                    signal = StrategySignal.BUY
                    confidence = min(0.9, (current_price - upper) / upper + 0.7)
                elif current_price < lower:
                    signal = StrategySignal.SELL
                    confidence = min(0.9, (lower - current_price) / lower + 0.7)
            
            return StrategyResult(
                strategy_name='bollinger_squeeze',
                signal=signal,
                confidence=confidence,
                entry_price=current_price,
                timestamp=current_time
            )
            
        except Exception as e:
            self.logger.error(f"ボリンジャー戦略エラー: {e}")
            return None
    
    def volume_breakout_strategy(self, symbol: str, current_time: datetime) -> Optional[StrategyResult]:
        """出来高ブレイクアウト戦略"""
        try:
            df = self.get_market_data(symbol, current_time, 6)  # 6時間
            if df is None or len(df) < 12:
                return None
            
            current_price = df['close_price'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(12).mean().iloc[-1]  # 1時間平均
            
            # 価格変動
            price_change = (current_price - df['close_price'].iloc[-2]) / df['close_price'].iloc[-2]
            
            signal = StrategySignal.HOLD
            confidence = 0.5
            
            # 出来高が平均の1.5倍以上かつ価格変動
            if current_volume > avg_volume * 1.5:
                if price_change > 0.01:  # 1%以上上昇
                    signal = StrategySignal.BUY
                    confidence = min(0.9, (current_volume / avg_volume - 1) * 0.5 + 0.7)
                elif price_change < -0.01:  # 1%以上下落
                    signal = StrategySignal.SELL
                    confidence = min(0.9, (current_volume / avg_volume - 1) * 0.5 + 0.7)
            
            return StrategyResult(
                strategy_name='volume_breakout',
                signal=signal,
                confidence=confidence,
                entry_price=current_price,
                timestamp=current_time
            )
            
        except Exception as e:
            self.logger.error(f"出来高ブレイクアウト戦略エラー: {e}")
            return None
    
    def moving_average_strategy(self, symbol: str, current_time: datetime) -> Optional[StrategyResult]:
        """移動平均戦略"""
        try:
            df = self.get_market_data(symbol, current_time, 4)  # 4時間
            if df is None or len(df) < 12:
                return None
            
            current_price = df['close_price'].iloc[-1]
            ma_5 = df['ma_5min'].iloc[-1] if not pd.isna(df['ma_5min'].iloc[-1]) else current_price
            ma_20 = df['ma_20min'].iloc[-1] if not pd.isna(df['ma_20min'].iloc[-1]) else current_price
            ma_60 = df['ma_60min'].iloc[-1] if not pd.isna(df['ma_60min'].iloc[-1]) else current_price
            
            signal = StrategySignal.HOLD
            confidence = 0.5
            
            # トレンド判定
            if ma_5 > ma_20 > ma_60 and current_price > ma_5:  # 上昇トレンド
                signal = StrategySignal.BUY
                confidence = min(0.9, ((ma_5 - ma_60) / ma_60) * 10 + 0.6)
            elif ma_5 < ma_20 < ma_60 and current_price < ma_5:  # 下降トレンド
                signal = StrategySignal.SELL
                confidence = min(0.9, ((ma_60 - ma_5) / ma_60) * 10 + 0.6)
            
            return StrategyResult(
                strategy_name='moving_average_trend',
                signal=signal,
                confidence=confidence,
                entry_price=current_price,
                timestamp=current_time
            )
            
        except Exception as e:
            self.logger.error(f"移動平均戦略エラー: {e}")
            return None
    
    def support_resistance_strategy(self, symbol: str, current_time: datetime) -> Optional[StrategyResult]:
        """サポート・レジスタンス戦略"""
        try:
            df = self.get_market_data(symbol, current_time, 12)  # 12時間
            if df is None or len(df) < 24:
                return None
            
            current_price = df['close_price'].iloc[-1]
            prices = df['close_price']
            
            # 直近の高値・安値
            recent_high = prices.iloc[-12:].max()  # 1時間の高値
            recent_low = prices.iloc[-12:].min()   # 1時間の安値
            
            # より長期の高値・安値
            long_high = prices.max()
            long_low = prices.min()
            
            signal = StrategySignal.HOLD
            confidence = 0.5
            
            # サポートライン近くでの反発
            if abs(current_price - recent_low) / recent_low < 0.005:  # 0.5%以内
                signal = StrategySignal.BUY
                confidence = 0.8
            # レジスタンスライン近くでの反落
            elif abs(current_price - recent_high) / recent_high < 0.005:  # 0.5%以内
                signal = StrategySignal.SELL
                confidence = 0.8
            
            return StrategyResult(
                strategy_name='support_resistance',
                signal=signal,
                confidence=confidence,
                entry_price=current_price,
                timestamp=current_time
            )
            
        except Exception as e:
            self.logger.error(f"サポート・レジスタンス戦略エラー: {e}")
            return None
    
    def execute_all_strategies(self, symbol: str, current_time: datetime) -> List[StrategyResult]:
        """全戦略実行"""
        results = []
        
        # 各戦略実行
        strategies_methods = [
            self.rsi_strategy,
            self.macd_strategy,
            self.bollinger_strategy,
            self.volume_breakout_strategy,
            self.moving_average_strategy,
            self.support_resistance_strategy
        ]
        
        for strategy_method in strategies_methods:
            try:
                result = strategy_method(symbol, current_time)
                if result and result.signal != StrategySignal.HOLD:
                    results.append(result)
            except Exception as e:
                self.logger.error(f"戦略実行エラー: {e}")
        
        return results
    
    def aggregate_signals(self, results: List[StrategyResult]) -> Dict:
        """シグナル統合"""
        if not results:
            return {
                'final_signal': StrategySignal.HOLD,
                'confidence': 0.0,
                'buy_weight': 0.0,
                'sell_weight': 0.0,
                'strategy_count': 0
            }
        
        buy_weight = 0.0
        sell_weight = 0.0
        
        for result in results:
            strategy = self.strategies.get(result.strategy_name)
            if not strategy:
                continue
            
            # 信頼度フィルタ
            if result.confidence < strategy.min_confidence:
                continue
            
            weight = strategy.weight * result.confidence
            
            if result.signal == StrategySignal.BUY:
                buy_weight += weight
            elif result.signal == StrategySignal.SELL:
                sell_weight += weight
        
        # 最終判定
        total_weight = buy_weight + sell_weight
        final_signal = StrategySignal.HOLD
        final_confidence = 0.0
        
        if total_weight > 0:
            if buy_weight > sell_weight * 1.2:  # 買いが20%以上優勢
                final_signal = StrategySignal.BUY
                final_confidence = buy_weight / total_weight
            elif sell_weight > buy_weight * 1.2:  # 売りが20%以上優勢
                final_signal = StrategySignal.SELL
                final_confidence = sell_weight / total_weight
        
        return {
            'final_signal': final_signal,
            'confidence': final_confidence,
            'buy_weight': buy_weight,
            'sell_weight': sell_weight,
            'strategy_count': len(results),
            'strategy_details': results
        }
    
    def get_trading_signal(self, symbol: str, current_time: datetime) -> Dict:
        """取引シグナル取得（方法2のメイン機能）"""
        try:
            # 全戦略実行
            strategy_results = self.execute_all_strategies(symbol, current_time)
            
            # シグナル統合
            aggregated = self.aggregate_signals(strategy_results)
            
            self.logger.info(f"戦略シグナル - シンボル: {symbol}, シグナル: {aggregated['final_signal']}, "
                           f"信頼度: {aggregated['confidence']:.3f}, 戦略数: {aggregated['strategy_count']}")
            
            # 履歴に追加
            self.strategy_history.append({
                'symbol': symbol,
                'timestamp': current_time,
                'result': aggregated
            })
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"取引シグナル取得エラー: {e}")
            return {
                'final_signal': StrategySignal.HOLD,
                'confidence': 0.0,
                'buy_weight': 0.0,
                'sell_weight': 0.0,
                'strategy_count': 0
            }
    
    def get_strategy_performance(self) -> Dict:
        """戦略パフォーマンス取得"""
        if not self.strategy_history:
            return {}
        
        performance = {}
        
        for strategy_name in self.strategies.keys():
            strategy_results = [
                h for h in self.strategy_history 
                if any(r.strategy_name == strategy_name for r in h['result'].get('strategy_details', []))
            ]
            
            if strategy_results:
                performance[strategy_name] = {
                    'signal_count': len(strategy_results),
                    'avg_confidence': np.mean([
                        r.confidence for h in strategy_results 
                        for r in h['result']['strategy_details'] 
                        if r.strategy_name == strategy_name
                    ])
                }
        
        return performance

# 使用例
if __name__ == "__main__":
    tv_strategies = TradingViewStrategies()
    
    # テスト実行
    current_time = datetime.now()
    signal_result = tv_strategies.get_trading_signal('7203', current_time)
    
    print("戦略シグナル結果:")
    print(f"最終シグナル: {signal_result['final_signal']}")
    print(f"信頼度: {signal_result['confidence']:.3f}")
    print(f"買い重み: {signal_result['buy_weight']:.2f}")
    print(f"売り重み: {signal_result['sell_weight']:.2f}")
    print(f"戦略数: {signal_result['strategy_count']}")
