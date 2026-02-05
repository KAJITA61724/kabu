"""
バックテストシステム - 方法1と方法2の成績比較
- 履歴データを使った戦略テスト
- 成績比較レポート
- 戦略最適化
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ml_models import MLTradingModels
from strategy_system import TradingViewStrategies, StrategySignal
from integrated_trading_system import TradingMethod, Position, PositionStatus

class BacktestEngine:
    """バックテストエンジン"""
    
    def __init__(self, db_path: str = "enhanced_trading.db", initial_capital: float = 2000000):
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)
        
        # モデルとストラテジー
        self.ml_models = MLTradingModels(db_path)
        self.tv_strategies = TradingViewStrategies(db_path)
        
        # バックテスト設定
        self.max_positions = 5
        self.position_size_ratio = 0.2
        self.max_position_duration = 40  # 分
        self.stop_loss_rate = 0.02
        
        # 結果保存
        self.results = {}
        
    def get_historical_data(self, start_date: datetime, end_date: datetime, symbols: List[str] = None) -> pd.DataFrame:
        """履歴データ取得"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if symbols:
                symbol_filter = f"AND symbol IN ({','.join(['?' for _ in symbols])})"
                params = [start_date, end_date] + symbols
            else:
                symbol_filter = ""
                params = [start_date, end_date]
            
            query = f'''
                SELECT 
                    timestamp,
                    symbol,
                    close_price,
                    volume,
                    ma_5min,
                    ma_20min,
                    ma_60min
                FROM five_minute_data
                WHERE timestamp >= ? AND timestamp <= ?
                {symbol_filter}
                ORDER BY timestamp, symbol
            '''
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
            
        except Exception as e:
            self.logger.error(f"履歴データ取得エラー: {e}")
            return pd.DataFrame()
    
    def simulate_method1(self, data: pd.DataFrame, symbols: List[str]) -> Dict:
        """方法1シミュレーション"""
        capital = self.initial_capital
        positions = {}
        trades = []
        
        for timestamp in data['timestamp'].unique():
            current_data = data[data['timestamp'] == timestamp]
            current_time = pd.to_datetime(timestamp)
            
            # 既存ポジションチェック
            positions_to_close = []
            for pos_id, position in positions.items():
                # 40分経過チェック
                time_elapsed = (current_time - position['entry_time']).total_seconds() / 60
                if time_elapsed >= self.max_position_duration:
                    positions_to_close.append((pos_id, "時間制限"))
                    continue
                
                # 価格取得
                price_data = current_data[current_data['symbol'] == position['symbol']]
                if len(price_data) == 0:
                    continue
                
                current_price = price_data.iloc[0]['close_price']
                
                # ストップロスチェック
                if position['direction'] == 1 and current_price <= position['stop_loss']:
                    positions_to_close.append((pos_id, "ストップロス"))
                elif position['direction'] == -1 and current_price >= position['stop_loss']:
                    positions_to_close.append((pos_id, "ストップロス"))
                else:
                    # ML予測チェック（簡易版）
                    price_change = (current_price - position['entry_price']) / position['entry_price']
                    if position['direction'] == 1 and price_change < -0.01:  # 1%下落
                        positions_to_close.append((pos_id, "予測変更"))
                    elif position['direction'] == -1 and price_change > 0.01:  # 1%上昇
                        positions_to_close.append((pos_id, "予測変更"))
            
            # ポジション手仕舞い
            for pos_id, reason in positions_to_close:
                position = positions[pos_id]
                price_data = current_data[current_data['symbol'] == position['symbol']]
                if len(price_data) > 0:
                    exit_price = price_data.iloc[0]['close_price']
                    
                    if position['direction'] == 1:
                        profit_loss = (exit_price - position['entry_price']) * position['quantity']
                    else:
                        profit_loss = (position['entry_price'] - exit_price) * position['quantity']
                    
                    capital += position['quantity'] * exit_price
                    
                    trades.append({
                        'method': 'METHOD_1',
                        'symbol': position['symbol'],
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'direction': position['direction'],
                        'quantity': position['quantity'],
                        'profit_loss': profit_loss,
                        'reason': reason
                    })
                
                del positions[pos_id]
            
            # 新規エントリーチェック
            if len(positions) < self.max_positions:
                for symbol in symbols:
                    if any(pos['symbol'] == symbol for pos in positions.values()):
                        continue
                    
                    symbol_data = current_data[current_data['symbol'] == symbol]
                    if len(symbol_data) == 0:
                        continue
                    
                    # 簡易ML判定（実際のMLモデルの代わり）
                    current_price = symbol_data.iloc[0]['close_price']
                    ma_5 = symbol_data.iloc[0]['ma_5min']
                    ma_20 = symbol_data.iloc[0]['ma_20min']
                    
                    if pd.notna(ma_5) and pd.notna(ma_20):
                        if current_price > ma_5 > ma_20:  # 上昇トレンド
                            direction = 1
                            confidence = 0.8
                        elif current_price < ma_5 < ma_20:  # 下降トレンド
                            direction = -1
                            confidence = 0.8
                        else:
                            continue
                        
                        if confidence >= 0.8:  # 高信頼度のみ
                            # ポジションサイズ計算
                            available_capital = capital * self.position_size_ratio
                            quantity = int((available_capital / current_price) // 100) * 100
                            
                            if quantity >= 100 and quantity * current_price <= capital:
                                pos_id = f"{symbol}_{current_time.strftime('%Y%m%d_%H%M%S')}"
                                
                                # ストップロス設定
                                if direction == 1:
                                    stop_loss = current_price * (1 - self.stop_loss_rate)
                                else:
                                    stop_loss = current_price * (1 + self.stop_loss_rate)
                                
                                positions[pos_id] = {
                                    'symbol': symbol,
                                    'direction': direction,
                                    'quantity': quantity,
                                    'entry_price': current_price,
                                    'entry_time': current_time,
                                    'stop_loss': stop_loss
                                }
                                
                                capital -= quantity * current_price
                                break
        
        # 残りポジション手仕舞い
        final_time = data['timestamp'].max()
        for position in positions.values():
            final_data = data[(data['timestamp'] == final_time) & (data['symbol'] == position['symbol'])]
            if len(final_data) > 0:
                exit_price = final_data.iloc[0]['close_price']
                
                if position['direction'] == 1:
                    profit_loss = (exit_price - position['entry_price']) * position['quantity']
                else:
                    profit_loss = (position['entry_price'] - exit_price) * position['quantity']
                
                capital += position['quantity'] * exit_price
                
                trades.append({
                    'method': 'METHOD_1',
                    'symbol': position['symbol'],
                    'entry_time': position['entry_time'],
                    'exit_time': final_time,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'direction': position['direction'],
                    'quantity': position['quantity'],
                    'profit_loss': profit_loss,
                    'reason': 'バックテスト終了'
                })
        
        return {
            'final_capital': capital,
            'trades': trades,
            'total_return': (capital - self.initial_capital) / self.initial_capital
        }
    
    def simulate_method2(self, data: pd.DataFrame, symbols: List[str]) -> Dict:
        """方法2シミュレーション"""
        capital = self.initial_capital
        positions = {}
        trades = []
        
        for timestamp in data['timestamp'].unique():
            current_data = data[data['timestamp'] == timestamp]
            current_time = pd.to_datetime(timestamp)
            
            # 既存ポジションチェック（方法1と同様のロジック）
            positions_to_close = []
            for pos_id, position in positions.items():
                time_elapsed = (current_time - position['entry_time']).total_seconds() / 60
                if time_elapsed >= self.max_position_duration:
                    positions_to_close.append((pos_id, "時間制限"))
                    continue
                
                price_data = current_data[current_data['symbol'] == position['symbol']]
                if len(price_data) == 0:
                    continue
                
                current_price = price_data.iloc[0]['close_price']
                
                if position['direction'] == 1 and current_price <= position['stop_loss']:
                    positions_to_close.append((pos_id, "ストップロス"))
                elif position['direction'] == -1 and current_price >= position['stop_loss']:
                    positions_to_close.append((pos_id, "ストップロス"))
                else:
                    # 戦略変更チェック（簡易版）
                    ma_5 = price_data.iloc[0]['ma_5min']
                    ma_20 = price_data.iloc[0]['ma_20min']
                    
                    if pd.notna(ma_5) and pd.notna(ma_20):
                        if position['direction'] == 1 and current_price < ma_5:  # 買いポジションで移動平均割れ
                            positions_to_close.append((pos_id, "戦略変更"))
                        elif position['direction'] == -1 and current_price > ma_5:  # 売りポジションで移動平均上抜け
                            positions_to_close.append((pos_id, "戦略変更"))
            
            # ポジション手仕舞い
            for pos_id, reason in positions_to_close:
                position = positions[pos_id]
                price_data = current_data[current_data['symbol'] == position['symbol']]
                if len(price_data) > 0:
                    exit_price = price_data.iloc[0]['close_price']
                    
                    if position['direction'] == 1:
                        profit_loss = (exit_price - position['entry_price']) * position['quantity']
                    else:
                        profit_loss = (position['entry_price'] - exit_price) * position['quantity']
                    
                    capital += position['quantity'] * exit_price
                    
                    trades.append({
                        'method': 'METHOD_2',
                        'symbol': position['symbol'],
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'direction': position['direction'],
                        'quantity': position['quantity'],
                        'profit_loss': profit_loss,
                        'reason': reason
                    })
                
                del positions[pos_id]
            
            # 新規エントリーチェック（戦略ベース）
            if len(positions) < self.max_positions:
                for symbol in symbols:
                    if any(pos['symbol'] == symbol for pos in positions.values()):
                        continue
                    
                    symbol_data = current_data[current_data['symbol'] == symbol]
                    if len(symbol_data) == 0:
                        continue
                    
                    # 簡易戦略判定
                    current_price = symbol_data.iloc[0]['close_price']
                    ma_5 = symbol_data.iloc[0]['ma_5min']
                    ma_20 = symbol_data.iloc[0]['ma_20min']
                    ma_60 = symbol_data.iloc[0]['ma_60min']
                    volume = symbol_data.iloc[0]['volume']
                    
                    # 複数戦略の組み合わせ
                    signal_strength = 0
                    
                    # 移動平均戦略
                    if pd.notna(ma_5) and pd.notna(ma_20) and pd.notna(ma_60):
                        if ma_5 > ma_20 > ma_60 and current_price > ma_5:
                            signal_strength += 1  # 買いシグナル
                        elif ma_5 < ma_20 < ma_60 and current_price < ma_5:
                            signal_strength -= 1  # 売りシグナル
                    
                    # 出来高戦略（前のデータと比較）
                    prev_data = data[(data['timestamp'] < timestamp) & (data['symbol'] == symbol)].tail(1)
                    if len(prev_data) > 0:
                        prev_volume = prev_data.iloc[0]['volume']
                        if volume > prev_volume * 1.5:  # 出来高増加
                            if current_price > prev_data.iloc[0]['close_price']:
                                signal_strength += 0.5
                            else:
                                signal_strength -= 0.5
                    
                    # シグナル強度による取引判定
                    if signal_strength >= 1.5:  # 強い買いシグナル
                        direction = 1
                        confidence = min(0.9, signal_strength / 2)
                    elif signal_strength <= -1.5:  # 強い売りシグナル
                        direction = -1
                        confidence = min(0.9, abs(signal_strength) / 2)
                    else:
                        continue
                    
                    if confidence >= 0.7:  # 高信頼度のみ
                        available_capital = capital * self.position_size_ratio
                        quantity = int((available_capital / current_price) // 100) * 100
                        
                        if quantity >= 100 and quantity * current_price <= capital:
                            pos_id = f"{symbol}_{current_time.strftime('%Y%m%d_%H%M%S')}"
                            
                            if direction == 1:
                                stop_loss = current_price * (1 - self.stop_loss_rate)
                            else:
                                stop_loss = current_price * (1 + self.stop_loss_rate)
                            
                            positions[pos_id] = {
                                'symbol': symbol,
                                'direction': direction,
                                'quantity': quantity,
                                'entry_price': current_price,
                                'entry_time': current_time,
                                'stop_loss': stop_loss
                            }
                            
                            capital -= quantity * current_price
                            break
        
        # 残りポジション手仕舞い
        final_time = data['timestamp'].max()
        for position in positions.values():
            final_data = data[(data['timestamp'] == final_time) & (data['symbol'] == position['symbol'])]
            if len(final_data) > 0:
                exit_price = final_data.iloc[0]['close_price']
                
                if position['direction'] == 1:
                    profit_loss = (exit_price - position['entry_price']) * position['quantity']
                else:
                    profit_loss = (position['entry_price'] - exit_price) * position['quantity']
                
                capital += position['quantity'] * exit_price
                
                trades.append({
                    'method': 'METHOD_2',
                    'symbol': position['symbol'],
                    'entry_time': position['entry_time'],
                    'exit_time': final_time,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'direction': position['direction'],
                    'quantity': position['quantity'],
                    'profit_loss': profit_loss,
                    'reason': 'バックテスト終了'
                })
        
        return {
            'final_capital': capital,
            'trades': trades,
            'total_return': (capital - self.initial_capital) / self.initial_capital
        }
    
    def run_backtest(self, start_date: datetime, end_date: datetime, symbols: List[str] = None) -> Dict:
        """バックテスト実行"""
        self.logger.info(f"バックテスト開始: {start_date} - {end_date}")
        
        # データ取得
        if symbols is None:
            prime_df = pd.read_csv("prime_symbols.csv")
            symbols = prime_df[prime_df['suitable_for_daytrading'] == True]['symbol'].astype(str).tolist()[:20]
        
        data = self.get_historical_data(start_date, end_date, symbols)
        
        if data.empty:
            self.logger.error("履歴データが取得できませんでした")
            return {}
        
        self.logger.info(f"データ取得完了: {len(data)} レコード, {len(symbols)} 銘柄")
        
        # 方法1実行
        self.logger.info("方法1バックテスト実行中...")
        method1_result = self.simulate_method1(data, symbols)
        
        # 方法2実行
        self.logger.info("方法2バックテスト実行中...")
        method2_result = self.simulate_method2(data, symbols)
        
        # 結果統合
        results = {
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'symbols': symbols
            },
            'method1': method1_result,
            'method2': method2_result,
            'comparison': self.compare_methods(method1_result, method2_result)
        }
        
        self.results = results
        self.logger.info("バックテスト完了")
        
        return results
    
    def compare_methods(self, method1: Dict, method2: Dict) -> Dict:
        """方法比較"""
        try:
            method1_trades = method1['trades']
            method2_trades = method2['trades']
            
            # 基本統計
            method1_stats = self.calculate_trade_stats(method1_trades)
            method2_stats = self.calculate_trade_stats(method2_trades)
            
            comparison = {
                'method1_stats': method1_stats,
                'method2_stats': method2_stats,
                'better_method': 'METHOD_1' if method1['total_return'] > method2['total_return'] else 'METHOD_2',
                'return_difference': abs(method1['total_return'] - method2['total_return']),
                'summary': {
                    'method1_return': method1['total_return'],
                    'method2_return': method2['total_return'],
                    'method1_trades': len(method1_trades),
                    'method2_trades': len(method2_trades),
                    'method1_win_rate': method1_stats['win_rate'],
                    'method2_win_rate': method2_stats['win_rate']
                }
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"方法比較エラー: {e}")
            return {}
    
    def calculate_trade_stats(self, trades: List[Dict]) -> Dict:
        """取引統計計算"""
        if not trades:
            return {'total_trades': 0}
        
        profits = [trade['profit_loss'] for trade in trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        stats = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) if trades else 0,
            'total_profit': sum(profits),
            'avg_profit': np.mean(profits),
            'avg_winning_trade': np.mean(winning_trades) if winning_trades else 0,
            'avg_losing_trade': np.mean(losing_trades) if losing_trades else 0,
            'max_profit': max(profits) if profits else 0,
            'max_loss': min(profits) if profits else 0,
            'profit_factor': sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf')
        }
        
        return stats
    
    def generate_report(self, save_file: str = None) -> str:
        """レポート生成"""
        if not self.results:
            return "バックテスト結果がありません"
        
        report = f"""
=== バックテスト結果レポート ===

期間: {self.results['period']['start_date']} - {self.results['period']['end_date']}
銘柄数: {len(self.results['period']['symbols'])}
初期資金: {self.initial_capital:,.0f}円

【方法1（ML ファクトチェック）】
最終資金: {self.results['method1']['final_capital']:,.0f}円
リターン率: {self.results['method1']['total_return']*100:.2f}%
取引回数: {self.results['comparison']['method1_stats']['total_trades']}
勝率: {self.results['comparison']['method1_stats']['win_rate']*100:.1f}%
平均損益: {self.results['comparison']['method1_stats']['avg_profit']:,.0f}円
最大利益: {self.results['comparison']['method1_stats']['max_profit']:,.0f}円
最大損失: {self.results['comparison']['method1_stats']['max_loss']:,.0f}円

【方法2（ストラテジーベース）】
最終資金: {self.results['method2']['final_capital']:,.0f}円
リターン率: {self.results['method2']['total_return']*100:.2f}%
取引回数: {self.results['comparison']['method2_stats']['total_trades']}
勝率: {self.results['comparison']['method2_stats']['win_rate']*100:.1f}%
平均損益: {self.results['comparison']['method2_stats']['avg_profit']:,.0f}円
最大利益: {self.results['comparison']['method2_stats']['max_profit']:,.0f}円
最大損失: {self.results['comparison']['method2_stats']['max_loss']:,.0f}円

【比較結果】
優位な方法: {self.results['comparison']['better_method']}
リターン差: {self.results['comparison']['return_difference']*100:.2f}%

        """
        
        if save_file:
            with open(save_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # JSON結果も保存
            json_file = save_file.replace('.txt', '.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        
        return report
    
    def plot_results(self, save_path: str = None):
        """結果プロット"""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # リターン比較
        methods = ['方法1 (ML)', '方法2 (戦略)']
        returns = [
            self.results['method1']['total_return'] * 100,
            self.results['method2']['total_return'] * 100
        ]
        
        axes[0, 0].bar(methods, returns, color=['blue', 'orange'])
        axes[0, 0].set_title('リターン率比較 (%)')
        axes[0, 0].set_ylabel('リターン率 (%)')
        
        # 取引回数比較
        trade_counts = [
            self.results['comparison']['method1_stats']['total_trades'],
            self.results['comparison']['method2_stats']['total_trades']
        ]
        
        axes[0, 1].bar(methods, trade_counts, color=['blue', 'orange'])
        axes[0, 1].set_title('取引回数比較')
        axes[0, 1].set_ylabel('取引回数')
        
        # 勝率比較
        win_rates = [
            self.results['comparison']['method1_stats']['win_rate'] * 100,
            self.results['comparison']['method2_stats']['win_rate'] * 100
        ]
        
        axes[1, 0].bar(methods, win_rates, color=['blue', 'orange'])
        axes[1, 0].set_title('勝率比較 (%)')
        axes[1, 0].set_ylabel('勝率 (%)')
        
        # 平均損益比較
        avg_profits = [
            self.results['comparison']['method1_stats']['avg_profit'],
            self.results['comparison']['method2_stats']['avg_profit']
        ]
        
        axes[1, 1].bar(methods, avg_profits, color=['blue', 'orange'])
        axes[1, 1].set_title('平均損益比較 (円)')
        axes[1, 1].set_ylabel('平均損益 (円)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"結果グラフ保存: {save_path}")
        
        plt.show()

# 使用例
if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    
    # バックテスト実行
    backtest = BacktestEngine()
    
    # 過去1週間のデータでテスト
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # 主要銘柄でテスト
    test_symbols = ['7203', '6758', '8306', '9984', '4755']  # トヨタ、ソニー、MUFG、SBG、楽天
    
    results = backtest.run_backtest(start_date, end_date, test_symbols)
    
    # レポート生成
    report = backtest.generate_report('backtest_report.txt')
    print(report)
    
    # グラフ表示
    backtest.plot_results('backtest_results.png')
