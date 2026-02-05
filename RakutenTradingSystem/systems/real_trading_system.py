"""
リアルトレードシステム - 統合版
既存のintegrated_trading_systemを整理統合
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# 既存モジュールをインポート
import sys
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.ml_models import MLTradingModels
from core.strategy_system import TradingViewStrategies, StrategySignal
from core.enhanced_data_collector import EnhancedDataCollector

class RealTradingSystem:
    """リアルトレードシステム統合クラス"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # ログディレクトリ設定
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        monthly_log_dir = log_dir / datetime.now().strftime("%Y%m")
        monthly_log_dir.mkdir(exist_ok=True)
        
        # ログハンドラー設定
        log_handler = logging.FileHandler(monthly_log_dir / 'real_trading.log')
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(log_handler)
        
        # コンポーネント初期化
        self.data_collector = EnhancedDataCollector()
        self.ml_models = MLTradingModels()
        self.tv_strategies = TradingViewStrategies()
        
        # 取引状態
        self.running = False
        self.trading_thread = None
        self.positions = {}
        self.capital = 2000000  # 初期資金
        
        # 取引設定
        self.max_positions = 3
        self.position_size_ratio = 0.25
        self.max_holding_minutes = 40
        self.check_interval_seconds = 300  # 5分間隔
        
        self.logger.info("リアルトレードシステム初期化完了")
    
    def start_ml_trading(self):
        """方法1（ML）取引開始"""
        print("\n🤖 方法1（ML）リアル取引開始")
        
        if self.running:
            print("⚠️ 取引は既に稼働中です")
            return
        
        self.trading_method = "ml"
        self._start_trading_engine()
    
    def start_strategy_trading(self):
        """方法2（戦略）取引開始"""
        print("\n📊 方法2（戦略）リアル取引開始")
        
        if self.running:
            print("⚠️ 取引は既に稼働中です")
            return
        
        self.trading_method = "strategy"
        self._start_trading_engine()
    
    def start_integrated_trading(self):
        """統合取引開始"""
        print("\n⚡ 統合リアル取引開始")
        
        if self.running:
            print("⚠️ 取引は既に稼働中です")
            return
        
        self.trading_method = "integrated"
        self._start_trading_engine()
    
    def _start_trading_engine(self):
        """取引エンジン開始"""
        try:
            # データ収集確認
            if not self._check_data_availability():
                print("❌ データが利用できません")
                return
            
            self.running = True
            self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
            self.trading_thread.start()
            
            print(f"✅ {self.trading_method}取引を開始しました")
            
        except Exception as e:
            print(f"❌ 取引開始エラー: {e}")
    
    def stop_trading(self):
        """取引停止"""
        if not self.running:
            print("⚠️ 取引は既に停止中です")
            return
        
        print("🛑 取引を停止します...")
        
        self.running = False
        
        if self.trading_thread:
            self.trading_thread.join(timeout=30)
        
        # 全ポジション決済
        self._close_all_positions("システム停止")
        
        print("✅ 取引を停止しました")
    
    def _check_data_availability(self) -> bool:
        """データ利用可能性確認"""
        try:
            # Excel接続確認
            if not self.data_collector.connected:
                if not self.data_collector.initialize_excel_connection():
                    return False
            
            # 銘柄リスト確認
            if not self.data_collector.symbols:
                self.data_collector.load_nikkei225_symbols()
            
            return True
            
        except Exception as e:
            self.logger.error(f"データ確認エラー: {e}")
            return False
    
    def _trading_loop(self):
        """取引メインループ"""
        self.logger.info(f"{self.trading_method}取引ループ開始")
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # 市場時間チェック
                if not self._is_market_hours(current_time):
                    time.sleep(60)
                    continue
                
                # 新規エントリーチェック
                self._check_new_entries(current_time)
                
                # 既存ポジション管理
                self._manage_positions(current_time)
                
                # 5分待機
                time.sleep(self.check_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"取引ループエラー: {e}")
                time.sleep(60)
        
        self.logger.info("取引ループ終了")
    
    def _is_market_hours(self, current_time: datetime) -> bool:
        """市場時間判定"""
        weekday = current_time.weekday()
        hour = current_time.hour
        minute = current_time.minute
        
        # 平日9:00-15:00
        if weekday >= 5:  # 土日
            return False
        
        if hour < 9 or hour >= 15:
            return False
        
        return True
    
    def _check_new_entries(self, current_time: datetime):
        """新規エントリーチェック"""
        if len(self.positions) >= self.max_positions:
            return
        
        try:
            # 最新データ収集
            collected_data = self.data_collector.collect_5min_data()
            
            if not collected_data:
                return
            
            # 方法別エントリー判定
            if self.trading_method == "ml":
                self._check_ml_entries(current_time, collected_data)
            elif self.trading_method == "strategy":
                self._check_strategy_entries(current_time, collected_data)
            elif self.trading_method == "integrated":
                self._check_integrated_entries(current_time, collected_data)
                
        except Exception as e:
            self.logger.error(f"新規エントリーチェックエラー: {e}")
    
    def _check_ml_entries(self, current_time: datetime, data: List[Dict]):
        """ML方法エントリーチェック"""
        for symbol_data in data[:10]:  # 上位10銘柄
            symbol = symbol_data['symbol']
            
            if symbol in self.positions:
                continue
            
            try:
                # MLファクトチェック
                result = self.ml_models.fact_check_predictions(symbol, current_time)
                
                if result['should_trade'] and result['confidence'] >= 0.8:
                    direction = "buy" if result['direction'] == 1 else "sell"
                    self._enter_position(symbol, direction, current_time, "ML", result)
                    
            except Exception as e:
                self.logger.error(f"ML判定エラー {symbol}: {e}")
    
    def _check_strategy_entries(self, current_time: datetime, data: List[Dict]):
        """戦略方法エントリーチェック"""
        for symbol_data in data[:10]:
            symbol = symbol_data['symbol']
            
            if symbol in self.positions:
                continue
            
            try:
                # 戦略シグナル取得
                signal_result = self.tv_strategies.get_trading_signal(symbol, current_time)
                
                if (signal_result['final_signal'] != StrategySignal.HOLD and 
                    signal_result['confidence'] > 0.7):
                    
                    direction = "buy" if signal_result['final_signal'] == StrategySignal.BUY else "sell"
                    self._enter_position(symbol, direction, current_time, "Strategy", signal_result)
                    
            except Exception as e:
                self.logger.error(f"戦略判定エラー {symbol}: {e}")
    
    def _check_integrated_entries(self, current_time: datetime, data: List[Dict]):
        """統合方法エントリーチェック"""
        # MLと戦略の両方をチェック
        self._check_ml_entries(current_time, data)
        self._check_strategy_entries(current_time, data)
    
    def _enter_position(self, symbol: str, direction: str, entry_time: datetime, 
                       method: str, signal_data: Dict):
        """ポジションエントリー"""
        try:
            # 現在価格取得
            current_price = self._get_current_price(symbol)
            if not current_price:
                return
            
            # ポジションサイズ計算
            quantity = self._calculate_position_size(current_price)
            
            position = {
                'symbol': symbol,
                'direction': direction,
                'quantity': quantity,
                'entry_price': current_price,
                'entry_time': entry_time,
                'method': method,
                'signal_data': signal_data,
                'status': 'open'
            }
            
            self.positions[symbol] = position
            
            self.logger.info(f"エントリー: {symbol} {direction} {quantity}株 @{current_price:.0f} ({method})")
            
        except Exception as e:
            self.logger.error(f"エントリーエラー {symbol}: {e}")
    
    def _manage_positions(self, current_time: datetime):
        """ポジション管理"""
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            
            try:
                # 現在価格取得
                current_price = self._get_current_price(symbol)
                if not current_price:
                    continue
                
                # 保有時間チェック
                holding_minutes = (current_time - position['entry_time']).total_seconds() / 60
                
                # 決済判定
                should_close = False
                close_reason = ""
                
                # 40分強制手仕舞い
                if holding_minutes >= self.max_holding_minutes:
                    should_close = True
                    close_reason = "時間切れ"
                else:
                    # 方法別チェック
                    if position['method'] == "ML":
                        should_close, close_reason = self._check_ml_exit(symbol, position, current_time)
                    elif position['method'] == "Strategy":
                        should_close, close_reason = self._check_strategy_exit(symbol, position, current_time)
                
                if should_close:
                    self._close_position(symbol, current_price, current_time, close_reason)
                    
            except Exception as e:
                self.logger.error(f"ポジション管理エラー {symbol}: {e}")
    
    def _check_ml_exit(self, symbol: str, position: Dict, current_time: datetime) -> tuple:
        """ML方法の手仕舞いチェック"""
        try:
            # 最新MLチェック
            result = self.ml_models.fact_check_predictions(symbol, current_time)
            
            # 予測が変わった場合
            original_direction = position['signal_data']['direction']
            if result['direction'] != original_direction or not result['should_trade']:
                return True, "ML予測変更"
            
        except Exception as e:
            self.logger.error(f"ML手仕舞いチェックエラー: {e}")
        
        return False, ""
    
    def _check_strategy_exit(self, symbol: str, position: Dict, current_time: datetime) -> tuple:
        """戦略方法の手仕舞いチェック"""
        try:
            # 最新戦略チェック
            signal_result = self.tv_strategies.get_trading_signal(symbol, current_time)
            
            # シグナルが変わった場合
            original_signal = position['signal_data']['final_signal']
            if signal_result['final_signal'] != original_signal:
                return True, "戦略変更"
            
        except Exception as e:
            self.logger.error(f"戦略手仕舞いチェックエラー: {e}")
        
        return False, ""
    
    def _close_position(self, symbol: str, exit_price: float, exit_time: datetime, reason: str):
        """ポジション決済"""
        try:
            position = self.positions[symbol]
            
            # 損益計算
            if position['direction'] == 'buy':
                pnl = (exit_price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - exit_price) * position['quantity']
            
            position['exit_price'] = exit_price
            position['exit_time'] = exit_time
            position['pnl'] = pnl
            position['close_reason'] = reason
            position['status'] = 'closed'
            
            # 取引履歴保存
            self._save_trade_history(position)
            
            self.logger.info(f"決済: {symbol} {reason} 損益: {pnl:,.0f}円")
            
            # ポジション削除
            del self.positions[symbol]
            
        except Exception as e:
            self.logger.error(f"決済エラー {symbol}: {e}")
    
    def _close_all_positions(self, reason: str):
        """全ポジション決済"""
        for symbol in list(self.positions.keys()):
            try:
                current_price = self._get_current_price(symbol)
                if current_price:
                    self._close_position(symbol, current_price, datetime.now(), reason)
            except Exception as e:
                self.logger.error(f"全決済エラー {symbol}: {e}")
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """現在価格取得"""
        try:
            # データベースから最新価格取得
            import sqlite3
            conn = sqlite3.connect("enhanced_trading.db")
            
            query = """
                SELECT close_price FROM five_minute_data
                WHERE symbol = ? 
                ORDER BY timestamp DESC
                LIMIT 1
            """
            
            result = conn.execute(query, (symbol,)).fetchone()
            conn.close()
            
            return result[0] if result else None
            
        except Exception as e:
            self.logger.error(f"価格取得エラー {symbol}: {e}")
            return None
    
    def _calculate_position_size(self, price: float) -> int:
        """ポジションサイズ計算"""
        position_value = self.capital * self.position_size_ratio
        quantity = int(position_value / price / 100) * 100  # 100株単位
        return max(quantity, 100)
    
    def _save_trade_history(self, position: Dict):
        """取引履歴保存"""
        try:
            import sqlite3
            conn = sqlite3.connect("enhanced_trading.db")
            
            # 取引履歴テーブル作成（存在しない場合）
            conn.execute('''
                CREATE TABLE IF NOT EXISTS real_trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    direction TEXT,
                    quantity INTEGER,
                    entry_price REAL,
                    exit_price REAL,
                    entry_time DATETIME,
                    exit_time DATETIME,
                    pnl REAL,
                    method TEXT,
                    close_reason TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 取引履歴挿入
            conn.execute('''
                INSERT INTO real_trade_history
                (symbol, direction, quantity, entry_price, exit_price, 
                 entry_time, exit_time, pnl, method, close_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position['symbol'], position['direction'], position['quantity'],
                position['entry_price'], position['exit_price'],
                position['entry_time'], position['exit_time'],
                position['pnl'], position['method'], position['close_reason']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"履歴保存エラー: {e}")
    
    def show_positions(self):
        """現在ポジション表示"""
        print("\n💼 現在のポジション")
        print("="*60)
        
        if not self.positions:
            print("ポジションはありません")
            return
        
        for symbol, position in self.positions.items():
            holding_minutes = (datetime.now() - position['entry_time']).total_seconds() / 60
            
            print(f"📊 {symbol}")
            print(f"   方向: {position['direction']}")
            print(f"   数量: {position['quantity']}株")
            print(f"   エントリー: {position['entry_price']:.0f}円")
            print(f"   方法: {position['method']}")
            print(f"   保有時間: {holding_minutes:.1f}分")
            print()
        
        print("="*60)
    
    def show_trading_history(self):
        """取引履歴表示"""
        print("\n📋 取引履歴")
        print("="*80)
        
        try:
            import sqlite3
            import pandas as pd
            
            conn = sqlite3.connect("enhanced_trading.db")
            
            query = """
                SELECT symbol, direction, quantity, entry_price, exit_price,
                       entry_time, exit_time, pnl, method, close_reason
                FROM real_trade_history
                ORDER BY exit_time DESC
                LIMIT 20
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                print("取引履歴がありません")
                return
            
            total_pnl = df['pnl'].sum()
            win_rate = (df['pnl'] > 0).mean() * 100
            
            print(f"📊 取引サマリー:")
            print(f"   総取引数: {len(df)}")
            print(f"   勝率: {win_rate:.1f}%")
            print(f"   総損益: {total_pnl:,.0f}円")
            print()
            
            print("📋 最新20取引:")
            for _, row in df.iterrows():
                profit_loss = "利益" if row['pnl'] > 0 else "損失"
                print(f"   {row['symbol']} {row['direction']} {row['quantity']}株")
                print(f"   {row['entry_price']:.0f}→{row['exit_price']:.0f} {profit_loss}:{row['pnl']:,.0f}円")
                print(f"   {row['method']} ({row['close_reason']}) {row['exit_time']}")
                print()
                
        except Exception as e:
            print(f"❌ 履歴表示エラー: {e}")
        
        print("="*80)
