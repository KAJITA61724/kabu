"""
テクニカル指標計算システム
- RSI、MACD、ボリンジャーバンド等の計算
- 価格モメンタム、ボラティリティ計算
- 統計指標のデータベース保存
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import talib
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """テクニカル指標計算クラス"""
    
    def __init__(self, db_path: str = "enhanced_trading.db"):
        self.db_path = db_path
        
        # 指標計算パラメータ
        self.parameters = {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bollinger_period': 20,
            'bollinger_std': 2,
            'volume_ma_period': 20,
            'momentum_period': 10,
            'volatility_period': 20
        }
    
    def get_historical_data(self, symbol: str, hours_back: int = 72) -> pd.DataFrame:
        """指定銘柄の履歴データ取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                timestamp,
                open_price,
                high_price,
                low_price,
                close_price,
                volume
            FROM five_minute_data
            WHERE symbol = ? AND timestamp >= ?
            ORDER BY timestamp ASC
        '''
        
        df = pd.read_sql_query(query, conn, params=(symbol, cutoff_time))
        conn.close()
        
        if df.empty:
            return pd.DataFrame()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI計算"""
        try:
            return talib.RSI(prices.values, timeperiod=period)
        except:
            # TALibが使えない場合の代替実装
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """MACD計算"""
        try:
            macd, signal_line, histogram = talib.MACD(prices.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return {
                'macd': macd,
                'signal': signal_line,
                'histogram': histogram
            }
        except:
            # TALibが使えない場合の代替実装
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line
            
            return {
                'macd': macd.values,
                'signal': signal_line.values,
                'histogram': histogram.values
            }
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict:
        """ボリンジャーバンド計算"""
        try:
            upper, middle, lower = talib.BBANDS(prices.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
            return {
                'upper': upper,
                'middle': middle,
                'lower': lower
            }
        except:
            # TALibが使えない場合の代替実装
            rolling_mean = prices.rolling(window=period).mean()
            rolling_std = prices.rolling(window=period).std()
            
            return {
                'upper': (rolling_mean + rolling_std * std_dev).values,
                'middle': rolling_mean.values,
                'lower': (rolling_mean - rolling_std * std_dev).values
            }
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> Dict:
        """出来高指標計算"""
        volume = df['volume']
        prices = df['close_price']
        
        # 出来高移動平均
        volume_ma = volume.rolling(window=self.parameters['volume_ma_period']).mean()
        
        # 出来高比率
        volume_ratio = volume / volume_ma
        
        # 出来高加重平均価格（VWAP）
        vwap = (prices * volume).cumsum() / volume.cumsum()
        
        # 出来高勢い指標
        volume_momentum = volume.pct_change(periods=5)
        
        # Money Flow Index (MFI)
        try:
            high = df['high_price']
            low = df['low_price']
            close = df['close_price']
            
            mfi = talib.MFI(high.values, low.values, close.values, volume.values, timeperiod=14)
        except:
            # 代替実装
            typical_price = (df['high_price'] + df['low_price'] + df['close_price']) / 3
            money_flow = typical_price * volume
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            
            mfi = 100 - (100 / (1 + positive_flow / negative_flow))
            mfi = mfi.values
        
        return {
            'volume_ma': volume_ma.values,
            'volume_ratio': volume_ratio.values,
            'vwap': vwap.values,
            'volume_momentum': volume_momentum.values,
            'mfi': mfi
        }
    
    def calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict:
        """モメンタム指標計算"""
        prices = df['close_price']
        
        # 価格モメンタム
        momentum = prices.pct_change(periods=self.parameters['momentum_period'])
        
        # 加速度（モメンタムの変化率）
        acceleration = momentum.diff()
        
        # 相対的な強さ
        price_vs_ma = prices / prices.rolling(window=20).mean() - 1
        
        # トレンド強度
        trend_strength = prices.rolling(window=20).apply(
            lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) == 20 else 0
        )
        
        return {
            'momentum': momentum.values,
            'acceleration': acceleration.values,
            'price_vs_ma': price_vs_ma.values,
            'trend_strength': trend_strength.values
        }
    
    def calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict:
        """ボラティリティ指標計算"""
        prices = df['close_price']
        high = df['high_price']
        low = df['low_price']
        
        # 実現ボラティリティ
        returns = prices.pct_change()
        realized_volatility = returns.rolling(window=self.parameters['volatility_period']).std() * np.sqrt(288)  # 年率換算
        
        # True Range
        try:
            tr = talib.TRANGE(high.values, low.values, prices.values)
            atr = talib.ATR(high.values, low.values, prices.values, timeperiod=14)
        except:
            # 代替実装
            tr1 = high - low
            tr2 = abs(high - prices.shift(1))
            tr3 = abs(low - prices.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            tr = tr.values
            atr = atr.values
        
        # ボラティリティ比率
        volatility_ratio = realized_volatility / realized_volatility.rolling(window=60).mean()
        
        return {
            'realized_volatility': realized_volatility.values,
            'true_range': tr,
            'atr': atr,
            'volatility_ratio': volatility_ratio.values
        }
    
    def calculate_all_indicators(self, symbol: str) -> Dict:
        """全指標計算"""
        df = self.get_historical_data(symbol, hours_back=72)
        
        if df.empty or len(df) < 50:
            logging.warning(f"データ不足: {symbol} - データ数: {len(df)}")
            return {}
        
        indicators = {}
        
        try:
            # 基本指標
            indicators['rsi'] = self.calculate_rsi(df['close_price'], self.parameters['rsi_period'])
            
            # MACD
            macd_data = self.calculate_macd(df['close_price'])
            indicators.update(macd_data)
            
            # ボリンジャーバンド
            bb_data = self.calculate_bollinger_bands(df['close_price'])
            indicators.update(bb_data)
            
            # 出来高指標
            volume_data = self.calculate_volume_indicators(df)
            indicators.update(volume_data)
            
            # モメンタム指標
            momentum_data = self.calculate_momentum_indicators(df)
            indicators.update(momentum_data)
            
            # ボラティリティ指標
            volatility_data = self.calculate_volatility_indicators(df)
            indicators.update(volatility_data)
            
            # タイムスタンプ追加
            indicators['timestamps'] = df.index.tolist()
            
        except Exception as e:
            logging.error(f"指標計算エラー {symbol}: {e}")
            return {}
        
        return indicators
    
    def save_indicators_to_db(self, symbol: str, indicators: Dict):
        """指標をデータベースに保存"""
        if not indicators or 'timestamps' not in indicators:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamps = indicators['timestamps']
        rsi_values = indicators.get('rsi', [])
        macd_values = indicators.get('macd', [])
        volatility_values = indicators.get('realized_volatility', [])
        momentum_values = indicators.get('momentum', [])
        volume_ratio_values = indicators.get('volume_ratio', [])
        
        saved_count = 0
        
        for i, timestamp in enumerate(timestamps):
            try:
                # 各指標の値を取得（配列長が違う場合に対応）
                rsi_val = rsi_values[i] if i < len(rsi_values) and not np.isnan(rsi_values[i]) else None
                macd_val = macd_values[i] if i < len(macd_values) and not np.isnan(macd_values[i]) else None
                volatility_val = volatility_values[i] if i < len(volatility_values) and not np.isnan(volatility_values[i]) else None
                momentum_val = momentum_values[i] if i < len(momentum_values) and not np.isnan(momentum_values[i]) else None
                volume_ratio_val = volume_ratio_values[i] if i < len(volume_ratio_values) and not np.isnan(volume_ratio_values[i]) else None
                
                cursor.execute('''
                    INSERT OR REPLACE INTO statistics 
                    (symbol, timestamp, rsi_14, macd_signal, price_volatility, 
                     price_momentum, volume_ratio)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, timestamp, rsi_val, macd_val, volatility_val,
                    momentum_val, volume_ratio_val
                ))
                
                saved_count += 1
                
            except Exception as e:
                logging.error(f"指標保存エラー {symbol} {timestamp}: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        logging.info(f"指標保存完了 {symbol}: {saved_count}件")
    
    def calculate_and_save_all_symbols(self):
        """全銘柄の指標計算・保存"""
        conn = sqlite3.connect(self.db_path)
        
        # 利用可能な銘柄取得
        query = '''
            SELECT DISTINCT symbol 
            FROM five_minute_data 
            WHERE timestamp >= datetime('now', '-3 days')
        '''
        
        symbols = pd.read_sql_query(query, conn)['symbol'].tolist()
        conn.close()
        
        logging.info(f"指標計算開始: {len(symbols)}銘柄")
        
        processed_count = 0
        error_count = 0
        
        for symbol in symbols:
            try:
                # 指標計算
                indicators = self.calculate_all_indicators(symbol)
                
                if indicators:
                    # データベース保存
                    self.save_indicators_to_db(symbol, indicators)
                    processed_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                logging.error(f"指標計算エラー {symbol}: {e}")
                error_count += 1
        
        logging.info(f"指標計算完了: 成功 {processed_count}銘柄, エラー {error_count}銘柄")
    
    def get_latest_indicators(self, symbol: str) -> Dict:
        """最新の指標値取得"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                timestamp,
                rsi_14,
                macd_signal,
                price_volatility,
                price_momentum,
                volume_ratio
            FROM statistics
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 1
        '''
        
        result = conn.execute(query, (symbol,)).fetchone()
        conn.close()
        
        if result:
            return {
                'timestamp': result[0],
                'rsi': result[1],
                'macd': result[2],
                'volatility': result[3],
                'momentum': result[4],
                'volume_ratio': result[5]
            }
        else:
            return {}
    
    def get_indicator_summary(self) -> Dict:
        """指標サマリー取得"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                symbol,
                COUNT(*) as indicator_count,
                AVG(rsi_14) as avg_rsi,
                AVG(price_volatility) as avg_volatility,
                AVG(volume_ratio) as avg_volume_ratio,
                MAX(timestamp) as last_update
            FROM statistics
            WHERE timestamp >= datetime('now', '-1 day')
            GROUP BY symbol
            ORDER BY symbol
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        summary = {
            'total_symbols': len(df),
            'avg_rsi': df['avg_rsi'].mean() if not df.empty else 0,
            'avg_volatility': df['avg_volatility'].mean() if not df.empty else 0,
            'avg_volume_ratio': df['avg_volume_ratio'].mean() if not df.empty else 0,
            'symbols_with_indicators': len(df[df['indicator_count'] > 0]),
            'symbol_details': df.to_dict('records')
        }
        
        return summary

# 使用例
if __name__ == "__main__":
    indicators = TechnicalIndicators()
    
    # 全銘柄の指標計算
    indicators.calculate_and_save_all_symbols()
    
    # サマリー表示
    summary = indicators.get_indicator_summary()
    print(f"指標計算完了: {summary['total_symbols']}銘柄")
    print(f"平均RSI: {summary['avg_rsi']:.2f}")
    print(f"平均ボラティリティ: {summary['avg_volatility']:.4f}")
    print(f"平均出来高比率: {summary['avg_volume_ratio']:.2f}")