"""
リアルタイム監視付きデモトレードシステム
仕様通りの5分ごとファクトチェック・40分手仕舞い機能を実装
"""

import sys
from pathlib import Path
import logging
import sqlite3
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import threading
import json

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from core.ml_models import MLTradingModels
from core.strategy_system import TradingViewStrategies, StrategySignal

class TradingSignal(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class DemoMethod(Enum):
    METHOD_1_ML = "ml_factcheck"
    METHOD_2_STRATEGY = "strategy_based"

@dataclass
class Position:
    symbol: str
    side: TradingSignal
    entry_price: float
    quantity: int
    entry_time: datetime
    method: DemoMethod
    confidence: float
    # リアルタイム監視用
    last_check_time: datetime
    consecutive_fails: int = 0
    max_hold_time: int = 40  # 40分

@dataclass
class Trade:
    symbol: str
    side: TradingSignal
    entry_price: float
    exit_price: float
    quantity: int
    entry_time: datetime
    exit_time: datetime
    pnl: float
    exit_reason: str
    method: DemoMethod
    confidence: float

class RealTimeTradeMonitor:
    """リアルタイム取引監視システム"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.capital = config.get('capital', 2000000)
        self.running = False
        self.monitor_thread = None
        
        # モデル初期化
        self.ml_models = MLTradingModels()
        self.tv_strategies = TradingViewStrategies()
        
        # ログ設定
        self.logger = logging.getLogger(__name__)
        
        # データベース初期化
        self.db_path = "realtime_demo_trading.db"
        self.init_database()
    
    def init_database(self):
        """データベース初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # ポジションテーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                side TEXT,
                entry_price REAL,
                quantity INTEGER,
                entry_time DATETIME,
                method TEXT,
                confidence REAL,
                last_check_time DATETIME,
                consecutive_fails INTEGER
            )
        ''')
        
        # 取引履歴テーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity INTEGER,
                entry_time DATETIME,
                exit_time DATETIME,
                pnl REAL,
                exit_reason TEXT,
                method TEXT,
                confidence REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
        self.logger.info("リアルタイム取引データベース初期化完了")
    
    def get_current_price(self, symbol: str) -> float:
        """現在価格取得（デモ用）"""
        try:
            conn = sqlite3.connect("enhanced_trading.db")
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT close_price FROM five_minute_data 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (symbol,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return result[0]
            return 0.0
            
        except Exception as e:
            self.logger.error(f"価格取得エラー {symbol}: {e}")
            return 0.0
    
    def method1_factcheck(self, symbol: str, current_time: datetime) -> Dict:
        """方法1: MLファクトチェック"""
        try:
            return self.ml_models.fact_check_predictions(symbol, current_time)
        except Exception as e:
            self.logger.error(f"ML ファクトチェックエラー {symbol}: {e}")
            return {'should_trade': False, 'direction': None, 'confidence': 0.0}
    
    def method2_strategy_check(self, symbol: str, current_time: datetime) -> Dict:
        """方法2: ストラテジーチェック"""
        try:
            result = self.tv_strategies.get_trading_signal(symbol, current_time)
            
            # 予測方向の一致をチェック
            if result['final_signal'] == StrategySignal.HOLD:
                return {'should_continue': False, 'direction': None, 'confidence': 0.0}
            
            direction = 1 if result['final_signal'] == StrategySignal.BUY else 0
            return {
                'should_continue': result['confidence'] > 0.5,
                'direction': direction,
                'confidence': result['confidence']
            }
            
        except Exception as e:
            self.logger.error(f"ストラテジーチェックエラー {symbol}: {e}")
            return {'should_continue': False, 'direction': None, 'confidence': 0.0}
    
    def enter_position(self, symbol: str, side: TradingSignal, price: float, 
                      method: DemoMethod, confidence: float) -> bool:
        """ポジション新規建て"""
        try:
            # 資金計算
            position_size = min(self.capital * 0.3, 500000)  # 30%または50万円
            quantity = int(position_size / price)
            
            if quantity <= 0:
                self.logger.warning(f"数量不足: {symbol}")
                return False
            
            # ポジション作成
            position = Position(
                symbol=symbol,
                side=side,
                entry_price=price,
                quantity=quantity,
                entry_time=datetime.now(),
                method=method,
                confidence=confidence,
                last_check_time=datetime.now()
            )
            
            self.positions[symbol] = position
            self.capital -= position_size
            
            self.logger.info(f"✅ エントリー: {symbol} {side.value} {quantity}株 @{price:.2f} (信頼度:{confidence:.3f})")
            
            # データベース保存
            self.save_position_to_db(position)
            
            return True
            
        except Exception as e:
            self.logger.error(f"エントリーエラー {symbol}: {e}")
            return False
    
    def exit_position(self, symbol: str, exit_price: float, reason: str):
        """ポジション決済"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # 損益計算
        if position.side == TradingSignal.BUY:
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity
        
        # 取引記録作成
        trade = Trade(
            symbol=symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            pnl=pnl,
            exit_reason=reason,
            method=position.method,
            confidence=position.confidence
        )
        
        self.trades.append(trade)
        self.capital += position.entry_price * position.quantity + pnl
        
        self.logger.info(f"❌ 決済: {symbol} {position.side.value} 損益:{pnl:,.0f}円 理由:{reason}")
        
        # データベース保存
        self.save_trade_to_db(trade)
        
        # ポジション削除
        del self.positions[symbol]
        self.remove_position_from_db(symbol)
    
    def monitor_positions(self):
        """ポジション監視メインループ"""
        self.logger.info("🔍 リアルタイム監視開始")
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # 各ポジションをチェック
                for symbol in list(self.positions.keys()):
                    position = self.positions[symbol]
                    
                    # 5分経過チェック
                    if (current_time - position.last_check_time).total_seconds() >= 300:  # 5分
                        self.check_position_continuation(symbol, current_time)
                    
                    # 40分経過チェック
                    if (current_time - position.entry_time).total_seconds() >= 2400:  # 40分
                        current_price = self.get_current_price(symbol)
                        self.exit_position(symbol, current_price, "40分経過")
                
                # 1分待機
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"監視ループエラー: {e}")
                time.sleep(60)
    
    def check_position_continuation(self, symbol: str, current_time: datetime):
        """ポジション継続判定"""
        position = self.positions[symbol]
        
        # 方法別チェック
        if position.method == DemoMethod.METHOD_1_ML:
            result = self.method1_factcheck(symbol, current_time)
            should_continue = result['should_trade']
            
            # 方向一致チェック
            if should_continue:
                position_direction = 1 if position.side == TradingSignal.BUY else 0
                prediction_direction = result['direction']
                should_continue = (position_direction == prediction_direction)
        
        elif position.method == DemoMethod.METHOD_2_STRATEGY:
            result = self.method2_strategy_check(symbol, current_time)
            should_continue = result['should_continue']
            
            # 方向一致チェック
            if should_continue:
                position_direction = 1 if position.side == TradingSignal.BUY else 0
                prediction_direction = result['direction']
                should_continue = (position_direction == prediction_direction)
        
        else:
            should_continue = False
        
        # 継続判定
        if should_continue:
            position.consecutive_fails = 0
            self.logger.info(f"🟢 継続: {symbol} - 予測方向一致")
        else:
            position.consecutive_fails += 1
            self.logger.warning(f"🔴 予測外れ: {symbol} - 連続失敗:{position.consecutive_fails}")
            
            # 即時手仕舞い
            current_price = self.get_current_price(symbol)
            self.exit_position(symbol, current_price, "予測外れ")
            return
        
        # 最終チェック時間更新
        position.last_check_time = current_time
        self.update_position_in_db(position)
    
    def start_demo_trading(self, duration_minutes: int = 60):
        """デモトレード開始"""
        self.logger.info(f"🚀 リアルタイムデモトレード開始 (実行時間:{duration_minutes}分)")
        
        self.running = True
        
        # 監視スレッド開始
        self.monitor_thread = threading.Thread(target=self.monitor_positions)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # メインループ（新規エントリー検索）
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        while datetime.now() < end_time and self.running:
            try:
                # 新規エントリー候補検索
                self.search_new_entries()
                
                # 5分待機
                time.sleep(300)
                
            except KeyboardInterrupt:
                self.logger.info("手動停止")
                break
            except Exception as e:
                self.logger.error(f"メインループエラー: {e}")
                time.sleep(60)
        
        self.stop_demo_trading()
    
    def search_new_entries(self):
        """新規エントリー検索"""
        if len(self.positions) >= 3:  # 最大3ポジション
            return
        
        # 銘柄リスト取得
        symbols = self.get_available_symbols()
        
        for symbol in symbols:
            if symbol in self.positions:
                continue
            
            current_time = datetime.now()
            current_price = self.get_current_price(symbol)
            
            if current_price <= 0:
                continue
            
            # 方法1: MLファクトチェック
            ml_result = self.method1_factcheck(symbol, current_time)
            if ml_result['should_trade'] and ml_result['confidence'] >= 0.8:
                side = TradingSignal.BUY if ml_result['direction'] == 1 else TradingSignal.SELL
                if self.enter_position(symbol, side, current_price, DemoMethod.METHOD_1_ML, ml_result['confidence']):
                    break
            
            # 方法2: ストラテジーベース
            strategy_result = self.method2_strategy_check(symbol, current_time)
            if strategy_result['should_continue'] and strategy_result['confidence'] >= 0.7:
                side = TradingSignal.BUY if strategy_result['direction'] == 1 else TradingSignal.SELL
                if self.enter_position(symbol, side, current_price, DemoMethod.METHOD_2_STRATEGY, strategy_result['confidence']):
                    break
    
    def get_available_symbols(self) -> List[str]:
        """利用可能銘柄リスト取得"""
        try:
            conn = sqlite3.connect("enhanced_trading.db")
            cursor = conn.cursor()
            
            cursor.execute("SELECT DISTINCT symbol FROM five_minute_data ORDER BY symbol")
            symbols = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return symbols
            
        except Exception as e:
            self.logger.error(f"銘柄リスト取得エラー: {e}")
            return []
    
    def stop_demo_trading(self):
        """デモトレード停止"""
        self.logger.info("🛑 デモトレード停止")
        self.running = False
        
        # 全ポジション決済
        for symbol in list(self.positions.keys()):
            current_price = self.get_current_price(symbol)
            self.exit_position(symbol, current_price, "システム停止")
        
        # 結果表示
        self.show_results()
    
    def show_results(self):
        """結果表示"""
        if not self.trades:
            self.logger.info("取引履歴なし")
            return
        
        total_pnl = sum(trade.pnl for trade in self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        total_trades = len(self.trades)
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"📊 デモトレード結果")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"総取引数: {total_trades}")
        self.logger.info(f"勝率: {win_rate:.1f}% ({winning_trades}/{total_trades})")
        self.logger.info(f"総損益: {total_pnl:,.0f}円")
        self.logger.info(f"最終資本: {self.capital:,.0f}円")
        self.logger.info(f"利回り: {(total_pnl/self.config['capital'])*100:.2f}%")
        
        # 方法別結果
        ml_trades = [t for t in self.trades if t.method == DemoMethod.METHOD_1_ML]
        strategy_trades = [t for t in self.trades if t.method == DemoMethod.METHOD_2_STRATEGY]
        
        if ml_trades:
            ml_pnl = sum(t.pnl for t in ml_trades)
            self.logger.info(f"ML方法: {len(ml_trades)}取引 損益:{ml_pnl:,.0f}円")
        
        if strategy_trades:
            strategy_pnl = sum(t.pnl for t in strategy_trades)
            self.logger.info(f"戦略方法: {len(strategy_trades)}取引 損益:{strategy_pnl:,.0f}円")
    
    def save_position_to_db(self, position: Position):
        """ポジションをDBに保存"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO positions 
                (symbol, side, entry_price, quantity, entry_time, method, confidence, last_check_time, consecutive_fails)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position.symbol, position.side.value, position.entry_price, position.quantity,
                position.entry_time, position.method.value, position.confidence,
                position.last_check_time, position.consecutive_fails
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"ポジション保存エラー: {e}")
    
    def update_position_in_db(self, position: Position):
        """ポジション更新"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE positions 
                SET last_check_time = ?, consecutive_fails = ?
                WHERE symbol = ?
            """, (position.last_check_time, position.consecutive_fails, position.symbol))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"ポジション更新エラー: {e}")
    
    def remove_position_from_db(self, symbol: str):
        """ポジション削除"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"ポジション削除エラー: {e}")
    
    def save_trade_to_db(self, trade: Trade):
        """取引をDBに保存"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trades 
                (symbol, side, entry_price, exit_price, quantity, entry_time, exit_time, pnl, exit_reason, method, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.symbol, trade.side.value, trade.entry_price, trade.exit_price,
                trade.quantity, trade.entry_time, trade.exit_time, trade.pnl,
                trade.exit_reason, trade.method.value, trade.confidence
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"取引保存エラー: {e}")

def main():
    """メイン実行"""
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🎯 リアルタイム監視付きデモトレード")
    print("=" * 60)
    print("仕様:")
    print("- 5分ごとのファクトチェック")
    print("- 予測外れ時の即時手仕舞い")
    print("- 40分後の自動手仕舞い")
    print("- 最大3ポジション同時保有")
    print("=" * 60)
    
    # 設定
    config = {
        'capital': 2000000,
        'max_positions': 3,
        'demo_mode': True
    }
    
    # 実行時間設定
    duration = input("実行時間（分）を入力してください（デフォルト:60): ").strip()
    if not duration:
        duration = 60
    else:
        duration = int(duration)
    
    # システム開始
    monitor = RealTimeTradeMonitor(config)
    
    try:
        monitor.start_demo_trading(duration)
    except KeyboardInterrupt:
        print("\n手動停止されました")
        monitor.stop_demo_trading()

if __name__ == "__main__":
    main()
