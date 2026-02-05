"""
統合取引システム - 方法1と方法2の実装
- 40分間のポジション管理
- 5分ごとの予測チェック
- 自動手仕舞い機能
- 統合監視システム
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
import os

# 必要なモジュールのインポート
from fundamental_data_collector import FundamentalDataCollector
from multi_model_comparison import MultiModelComparison
from core.enhanced_data_collector import EnhancedDataCollector
from core.ml_models import MLTradingModels
from systems.demo_trading_system import DemoTradingSystem

class TradingMethod(Enum):
    """取引方法"""
    METHOD_1 = "ML_FACT_CHECK"
    METHOD_2 = "STRATEGY_BASED"

class PositionStatus(Enum):
    """ポジション状態"""
    OPEN = "open"
    CLOSED = "closed"
    STOPPED = "stopped"

@dataclass
class Position:
    """ポジション情報"""
    id: str
    symbol: str
    method: TradingMethod
    direction: int  # 1=買い, -1=売り
    quantity: int
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    status: PositionStatus = PositionStatus.OPEN
    stop_loss: Optional[float] = None
    profit_loss: Optional[float] = None
    reason: str = ""

class IntegratedTradingSystem:
    """統合取引システム"""
    
    def __init__(self, db_path: str = "enhanced_trading.db", initial_capital: float = 2000000):
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.logger = logging.getLogger(__name__)
        
        # モデルとストラテジー初期化
        self.ml_models = MLTradingModels(db_path)
        self.tv_strategies = TradingViewStrategies(db_path)
        
        # ポジション管理
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Position] = []
        
        # 取引設定
        self.max_positions = 5
        self.position_size_ratio = 0.2  # 資金の20%まで
        self.max_position_duration = 40  # 40分
        self.stop_loss_rate = 0.02  # 2%損失で停止
        
        # 監視用
        self.running = False
        self.monitor_thread = None
        
        self.setup_database()
    
    def setup_database(self):
        """データベース設定"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # ポジションテーブル
            conn.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    symbol TEXT,
                    method TEXT,
                    direction INTEGER,
                    quantity INTEGER,
                    entry_price REAL,
                    entry_time TEXT,
                    exit_price REAL,
                    exit_time TEXT,
                    status TEXT,
                    profit_loss REAL,
                    reason TEXT
                )
            ''')
            
            # 取引履歴テーブル
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trading_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id TEXT,
                    timestamp TEXT,
                    action TEXT,
                    price REAL,
                    quantity INTEGER,
                    method TEXT,
                    confidence REAL,
                    FOREIGN KEY (position_id) REFERENCES positions (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"データベース設定エラー: {e}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """現在価格取得"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT close_price 
                FROM five_minute_data 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            '''
            
            result = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()
            
            if len(result) > 0:
                return result.iloc[0]['close_price']
            
        except Exception as e:
            self.logger.error(f"現在価格取得エラー: {e}")
        
        return None
    
    def calculate_position_size(self, symbol: str, entry_price: float) -> int:
        """ポジションサイズ計算"""
        try:
            # 利用可能資金の20%まで
            available_capital = self.current_capital * self.position_size_ratio
            
            # 最大株数
            max_shares = int(available_capital / entry_price)
            
            # 100株単位に調整
            position_size = (max_shares // 100) * 100
            
            return max(100, position_size)  # 最低100株
            
        except Exception as e:
            self.logger.error(f"ポジションサイズ計算エラー: {e}")
            return 100
    
    def method1_trading_check(self, symbol: str, current_time: datetime) -> Dict:
        """方法1: MLファクトチェック取引判定"""
        try:
            # ファクトチェック実行
            fact_check = self.ml_models.fact_check_predictions(symbol, current_time)
            
            if fact_check['should_trade']:
                current_price = self.get_current_price(symbol)
                if current_price is None:
                    return {'should_trade': False}
                
                return {
                    'should_trade': True,
                    'method': TradingMethod.METHOD_1,
                    'direction': fact_check['direction'],
                    'confidence': fact_check['confidence'],
                    'entry_price': current_price,
                    'reason': f"ML予測一致: 信頼度{fact_check['confidence']:.3f}"
                }
            
            return {'should_trade': False}
            
        except Exception as e:
            self.logger.error(f"方法1取引判定エラー: {e}")
            return {'should_trade': False}
    
    def method2_trading_check(self, symbol: str, current_time: datetime) -> Dict:
        """方法2: ストラテジーベース取引判定"""
        try:
            # 戦略シグナル取得
            signal_result = self.tv_strategies.get_trading_signal(symbol, current_time)
            
            if (signal_result['final_signal'] != StrategySignal.HOLD and 
                signal_result['confidence'] > 0.7):
                
                current_price = self.get_current_price(symbol)
                if current_price is None:
                    return {'should_trade': False}
                
                direction = 1 if signal_result['final_signal'] == StrategySignal.BUY else -1
                
                return {
                    'should_trade': True,
                    'method': TradingMethod.METHOD_2,
                    'direction': direction,
                    'confidence': signal_result['confidence'],
                    'entry_price': current_price,
                    'reason': f"戦略シグナル: {signal_result['strategy_count']}戦略"
                }
            
            return {'should_trade': False}
            
        except Exception as e:
            self.logger.error(f"方法2取引判定エラー: {e}")
            return {'should_trade': False}
    
    def open_position(self, symbol: str, trade_info: Dict) -> Optional[str]:
        """ポジション開始"""
        try:
            if len(self.positions) >= self.max_positions:
                self.logger.warning("最大ポジション数に到達")
                return None
            
            # ポジションサイズ計算
            quantity = self.calculate_position_size(symbol, trade_info['entry_price'])
            required_capital = quantity * trade_info['entry_price']
            
            if required_capital > self.current_capital:
                self.logger.warning("資金不足")
                return None
            
            # ポジション作成
            position_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            position = Position(
                id=position_id,
                symbol=symbol,
                method=trade_info['method'],
                direction=trade_info['direction'],
                quantity=quantity,
                entry_price=trade_info['entry_price'],
                entry_time=datetime.now(),
                reason=trade_info['reason']
            )
            
            # ストップロス設定
            if trade_info['direction'] == 1:  # 買い
                position.stop_loss = trade_info['entry_price'] * (1 - self.stop_loss_rate)
            else:  # 売り
                position.stop_loss = trade_info['entry_price'] * (1 + self.stop_loss_rate)
            
            self.positions[position_id] = position
            self.current_capital -= required_capital
            
            # データベース保存
            self.save_position_to_db(position)
            
            self.logger.info(f"ポジション開始: {position_id} - {symbol} {trade_info['direction']} {quantity}株 @{trade_info['entry_price']}")
            
            return position_id
            
        except Exception as e:
            self.logger.error(f"ポジション開始エラー: {e}")
            return None
    
    def close_position(self, position_id: str, reason: str = "手仕舞い") -> bool:
        """ポジション終了"""
        try:
            if position_id not in self.positions:
                return False
            
            position = self.positions[position_id]
            current_price = self.get_current_price(position.symbol)
            
            if current_price is None:
                self.logger.error(f"価格取得失敗: {position.symbol}")
                return False
            
            # 損益計算
            if position.direction == 1:  # 買いポジション
                profit_loss = (current_price - position.entry_price) * position.quantity
            else:  # 売りポジション
                profit_loss = (position.entry_price - current_price) * position.quantity
            
            # ポジション更新
            position.exit_price = current_price
            position.exit_time = datetime.now()
            position.profit_loss = profit_loss
            position.status = PositionStatus.CLOSED
            position.reason = reason
            
            # 資金更新
            self.current_capital += position.quantity * current_price
            
            # 履歴に移動
            self.position_history.append(position)
            del self.positions[position_id]
            
            # データベース更新
            self.update_position_in_db(position)
            
            self.logger.info(f"ポジション終了: {position_id} - 損益: {profit_loss:,.0f}円 ({reason})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"ポジション終了エラー: {e}")
            return False
    
    def check_position_exit_conditions(self, position: Position) -> Optional[str]:
        """ポジション手仕舞い条件チェック"""
        try:
            current_time = datetime.now()
            current_price = self.get_current_price(position.symbol)
            
            if current_price is None:
                return None
            
            # 1. 時間制限チェック（40分）
            time_elapsed = (current_time - position.entry_time).total_seconds() / 60
            if time_elapsed >= self.max_position_duration:
                return "時間制限"
            
            # 2. ストップロスチェック
            if position.direction == 1 and current_price <= position.stop_loss:
                return "ストップロス"
            elif position.direction == -1 and current_price >= position.stop_loss:
                return "ストップロス"
            
            # 3. 予測チェック（5分ごと）
            if position.method == TradingMethod.METHOD_1:
                # ML予測チェック
                minute_result = self.ml_models.predict_minute_direction(position.symbol, current_time)
                if minute_result:
                    predicted_direction, confidence = minute_result
                    # 予測方向が反対になった場合
                    if predicted_direction != position.direction and confidence > 0.7:
                        return "予測変更"
            
            elif position.method == TradingMethod.METHOD_2:
                # ストラテジー予測チェック
                signal_result = self.tv_strategies.get_trading_signal(position.symbol, current_time)
                if signal_result['final_signal'] != StrategySignal.HOLD:
                    strategy_direction = 1 if signal_result['final_signal'] == StrategySignal.BUY else -1
                    # 戦略方向が反対になった場合
                    if strategy_direction != position.direction and signal_result['confidence'] > 0.7:
                        return "戦略変更"
            
            return None
            
        except Exception as e:
            self.logger.error(f"手仕舞い条件チェックエラー: {e}")
            return None
    
    def monitor_positions(self):
        """ポジション監視（5分ごと）"""
        while self.running:
            try:
                current_time = datetime.now()
                positions_to_close = []
                
                for position_id, position in self.positions.items():
                    exit_reason = self.check_position_exit_conditions(position)
                    if exit_reason:
                        positions_to_close.append((position_id, exit_reason))
                
                # ポジション終了
                for position_id, reason in positions_to_close:
                    self.close_position(position_id, reason)
                
                # 新規取引機会チェック
                self.check_new_trading_opportunities(current_time)
                
                # 5分待機
                time.sleep(300)  # 5分 = 300秒
                
            except Exception as e:
                self.logger.error(f"ポジション監視エラー: {e}")
                time.sleep(60)  # エラー時は1分待機
    
    def check_new_trading_opportunities(self, current_time: datetime):
        """新規取引機会チェック"""
        try:
            # プライム銘柄リスト取得
            prime_df = pd.read_csv("prime_symbols.csv")
            suitable_symbols = prime_df[prime_df['suitable_for_daytrading'] == True]['symbol'].astype(str).tolist()
            
            for symbol in suitable_symbols[:20]:  # 上位20銘柄をチェック
                # 既存ポジションチェック
                if any(pos.symbol == symbol for pos in self.positions.values()):
                    continue
                
                # 方法1チェック
                method1_result = self.method1_trading_check(symbol, current_time)
                if method1_result['should_trade']:
                    self.open_position(symbol, method1_result)
                    continue
                
                # 方法2チェック
                method2_result = self.method2_trading_check(symbol, current_time)
                if method2_result['should_trade']:
                    self.open_position(symbol, method2_result)
                
        except Exception as e:
            self.logger.error(f"新規取引機会チェックエラー: {e}")
    
    def start_trading(self):
        """取引開始"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_positions)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("統合取引システム開始")
    
    def stop_trading(self):
        """取引停止"""
        self.running = False
        
        # 全ポジション手仕舞い
        for position_id in list(self.positions.keys()):
            self.close_position(position_id, "システム停止")
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        self.logger.info("統合取引システム停止")
    
    def save_position_to_db(self, position: Position):
        """ポジションをデータベースに保存"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            conn.execute('''
                INSERT OR REPLACE INTO positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.id, position.symbol, position.method.value,
                position.direction, position.quantity, position.entry_price,
                position.entry_time.isoformat(), position.exit_price,
                position.exit_time.isoformat() if position.exit_time else None,
                position.status.value, position.profit_loss, position.reason
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"ポジション保存エラー: {e}")
    
    def update_position_in_db(self, position: Position):
        """データベースのポジション更新"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            conn.execute('''
                UPDATE positions 
                SET exit_price=?, exit_time=?, status=?, profit_loss=?, reason=?
                WHERE id=?
            ''', (
                position.exit_price,
                position.exit_time.isoformat() if position.exit_time else None,
                position.status.value,
                position.profit_loss,
                position.reason,
                position.id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"ポジション更新エラー: {e}")
    
    def get_performance_report(self) -> Dict:
        """パフォーマンスレポート取得"""
        try:
            total_trades = len(self.position_history)
            if total_trades == 0:
                return {'total_trades': 0}
            
            # 損益計算
            total_pnl = sum(pos.profit_loss for pos in self.position_history if pos.profit_loss)
            winning_trades = len([pos for pos in self.position_history if pos.profit_loss and pos.profit_loss > 0])
            
            # 方法別成績
            method1_trades = [pos for pos in self.position_history if pos.method == TradingMethod.METHOD_1]
            method2_trades = [pos for pos in self.position_history if pos.method == TradingMethod.METHOD_2]
            
            report = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'total_pnl': total_pnl,
                'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0,
                'current_capital': self.current_capital,
                'total_return_rate': (self.current_capital - self.initial_capital) / self.initial_capital,
                'active_positions': len(self.positions),
                'method1_performance': {
                    'trades': len(method1_trades),
                    'pnl': sum(pos.profit_loss for pos in method1_trades if pos.profit_loss),
                    'win_rate': len([pos for pos in method1_trades if pos.profit_loss and pos.profit_loss > 0]) / len(method1_trades) if method1_trades else 0
                },
                'method2_performance': {
                    'trades': len(method2_trades),
                    'pnl': sum(pos.profit_loss for pos in method2_trades if pos.profit_loss),
                    'win_rate': len([pos for pos in method2_trades if pos.profit_loss and pos.profit_loss > 0]) / len(method2_trades) if method2_trades else 0
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"パフォーマンスレポート作成エラー: {e}")
            return {}

# 使用例
if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # システム初期化
    trading_system = IntegratedTradingSystem()
    
    try:
        # 取引開始
        trading_system.start_trading()
        
        # 30分間実行（テスト用）
        time.sleep(1800)
        
    except KeyboardInterrupt:
        print("手動停止")
    finally:
        # 取引停止
        trading_system.stop_trading()
        
        # 成績表示
        report = trading_system.get_performance_report()
        print("=== 取引成績 ===")
        print(f"総取引数: {report.get('total_trades', 0)}")
        print(f"勝率: {report.get('win_rate', 0)*100:.1f}%")
        print(f"総損益: {report.get('total_pnl', 0):,.0f}円")
        print(f"リターン率: {report.get('total_return_rate', 0)*100:.2f}%")
