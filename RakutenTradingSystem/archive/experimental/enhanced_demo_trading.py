"""
統合デモトレードシステム - 方法1、方法2対応版
- 履歴データを使用した1日遅れデモトレード
- 方法1（ML）と方法2（戦略）の両方をサポート
- 前々日データで銘柄選定、前日データでエントリー、当日で決済
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
import json
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 新しいモデルをインポート
from core.ml_models import MLTradingModels
from core.strategy_system import TradingViewStrategies, StrategySignal

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

# ログ設定
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# 月別ログディレクトリ
monthly_log_dir = log_dir / datetime.now().strftime("%Y%m")
monthly_log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(monthly_log_dir / 'enhanced_demo_trading.log'),
        logging.StreamHandler()
    ]
)

class DemoMethod(Enum):
    """デモ取引方法"""
    METHOD_1_ML = "ml_factcheck"
    METHOD_2_STRATEGY = "strategy_based"
    TRADITIONAL_VWAP = "traditional_vwap"

class TradingSignal(Enum):
    """取引シグナル"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

@dataclass
class EnhancedDemoPosition:
    """拡張デモポジション"""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    entry_price: float
    entry_date: date
    entry_vwap: float
    method: DemoMethod
    entry_reason: str
    confidence: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    status: str = "open"
    # 終了時
    exit_price: float = 0.0
    exit_date: Optional[date] = None
    realized_pnl: float = 0.0
    close_reason: str = ""
    # ML/戦略特有情報
    ml_hourly_prediction: float = 0.0
    ml_minute_prediction: int = 0
    strategy_signals: str = ""

class EnhancedHistoricalDataCollector:
    """拡張履歴データ収集・管理クラス"""
    
    def __init__(self, db_path: str = "enhanced_demo_trading.db"):
        self.db_path = db_path
        self.init_enhanced_database()
        
    def init_enhanced_database(self):
        """拡張データベース初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 5分足データテーブル（統合システムと同じ構造）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS five_minute_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                close_price REAL,
                volume INTEGER,
                ma_5min REAL,
                ma_20min REAL,
                ma_60min REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        ''')
        
        # 板情報テーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS demo_order_book (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                bid_price_1 REAL,
                ask_price_1 REAL,
                bid_volume_1 INTEGER,
                ask_volume_1 INTEGER,
                bid_ask_spread REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        ''')
        
        # 拡張デモ取引結果テーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS enhanced_demo_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER,
                entry_price REAL,
                entry_date DATE,
                entry_vwap REAL,
                exit_price REAL,
                exit_date DATE,
                realized_pnl REAL,
                method TEXT,
                entry_reason TEXT,
                confidence REAL,
                close_reason TEXT,
                ml_hourly_prediction REAL,
                ml_minute_prediction INTEGER,
                strategy_signals TEXT,
                demo_session TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # デモセッション管理テーブル（拡張版）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS enhanced_demo_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT UNIQUE NOT NULL,
                start_date DATE,
                end_date DATE,
                initial_capital REAL,
                final_capital REAL,
                method TEXT,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,
                avg_confidence REAL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logging.info("拡張デモ取引データベース初期化完了")
    
    def import_from_main_database(self, main_db_path: str = "enhanced_trading.db"):
        """メインデータベースからデータインポート"""
        try:
            main_conn = sqlite3.connect(main_db_path)
            demo_conn = sqlite3.connect(self.db_path)
            
            # 5分足データコピー
            query = '''
                SELECT f.symbol, f.timestamp, f.close_price, f.volume, 
                       COALESCE(m.ma_5min, f.close_price) as ma_5min,
                       COALESCE(m.ma_20min, f.close_price) as ma_20min,
                       COALESCE(m.ma_60min, f.close_price) as ma_60min
                FROM five_minute_data f
                LEFT JOIN moving_averages m ON f.symbol = m.symbol AND f.timestamp = m.timestamp
                WHERE f.timestamp >= ?
                ORDER BY f.timestamp DESC
            '''
            
            # 過去1週間のデータ取得
            week_ago = datetime.now() - timedelta(days=7)
            
            df = pd.read_sql_query(query, main_conn, params=(week_ago,))
            
            if not df.empty:
                df.to_sql('five_minute_data', demo_conn, if_exists='replace', index=False)
                logging.info(f"5分足データインポート完了: {len(df)}レコード")
            
            # 板情報コピー（存在する場合）
            try:
                order_query = '''
                    SELECT symbol, timestamp, bid_price_1, ask_price_1, 
                           bid_volume_1, ask_volume_1, bid_ask_spread
                    FROM order_book
                    WHERE timestamp >= ?
                '''
                
                order_df = pd.read_sql_query(order_query, main_conn, params=(week_ago,))
                
                if not order_df.empty:
                    order_df.to_sql('demo_order_book', demo_conn, if_exists='replace', index=False)
                    logging.info(f"板情報インポート完了: {len(order_df)}レコード")
                    
            except Exception as e:
                logging.warning(f"板情報インポートスキップ: {e}")
            
            main_conn.close()
            demo_conn.close()
            
        except Exception as e:
            logging.error(f"データインポートエラー: {e}")
    
    def get_available_data_range(self) -> Tuple[date, date]:
        """利用可能なデータ範囲取得"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT 
                    MIN(DATE(timestamp)) as min_date,
                    MAX(DATE(timestamp)) as max_date
                FROM five_minute_data
            '''
            
            result = conn.execute(query).fetchone()
            conn.close()
            
            if result and result[0] and result[1]:
                min_date = datetime.strptime(result[0], '%Y-%m-%d').date()
                max_date = datetime.strptime(result[1], '%Y-%m-%d').date()
                return min_date, max_date
            
        except Exception as e:
            logging.error(f"データ範囲取得エラー: {e}")
        
        return None, None
    
    def get_symbols_with_sufficient_data(self, target_date: date, days_back: int = 3) -> List[str]:
        """十分なデータがある銘柄リスト取得"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            start_date = target_date - timedelta(days=days_back)
            
            query = '''
                SELECT symbol, COUNT(*) as data_count
                FROM five_minute_data
                WHERE DATE(timestamp) BETWEEN ? AND ?
                GROUP BY symbol
                HAVING data_count >= ?
                ORDER BY data_count DESC
            '''
            
            min_data_points = days_back * 12 * 6  # 最低でも1日6時間分
            
            df = pd.read_sql_query(query, conn, params=(start_date, target_date, min_data_points))
            conn.close()
            
            return df['symbol'].tolist()
            
        except Exception as e:
            logging.error(f"銘柄リスト取得エラー: {e}")
            return []

class EnhancedDemoTradingStrategy:
    """拡張デモ取引戦略クラス"""
    
    def __init__(self, config, data_collector: EnhancedHistoricalDataCollector):
        self.config = config
        self.data_collector = data_collector
        self.ml_models = MLTradingModels(data_collector.db_path)
        self.tv_strategies = TradingViewStrategies(data_collector.db_path)
        
        self.capital = config.get('capital', 2000000)
        self.max_positions = config.get('max_positions', 3)
        
    def analyze_ml_candidates(self, target_date: date) -> List[Dict]:
        """方法1: ML候補分析"""
        analysis_date = target_date - timedelta(days=2)
        analysis_time = datetime.combine(analysis_date, datetime.min.time().replace(hour=14))  # 14:00想定
        
        symbols = self.data_collector.get_symbols_with_sufficient_data(target_date)[:30]
        
        ml_candidates = []
        
        for symbol in symbols:
            try:
                # MLファクトチェック実行
                result = self.ml_models.fact_check_predictions(symbol, analysis_time)
                
                if result['should_trade']:
                    ml_candidates.append({
                        'symbol': symbol,
                        'method': DemoMethod.METHOD_1_ML,
                        'direction': result['direction'],
                        'confidence': result['confidence'],
                        'hourly_prediction': result['hourly_prediction'],
                        'minute_prediction': result['minute_prediction'],
                        'reason': f"ML予測一致 (H:{result['hourly_prediction']:.3f}, M:{result['minute_prediction']}, C:{result['confidence']:.3f})"
                    })
                    
            except Exception as e:
                logging.error(f"ML分析エラー {symbol}: {e}")
                continue
        
        # 信頼度でソート
        ml_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        logging.info(f"ML候補: {len(ml_candidates)}銘柄 (分析日: {analysis_date})")
        return ml_candidates[:10]
    
    def analyze_strategy_candidates(self, target_date: date) -> List[Dict]:
        """方法2: 戦略候補分析"""
        analysis_date = target_date - timedelta(days=2)
        analysis_time = datetime.combine(analysis_date, datetime.min.time().replace(hour=14))
        
        symbols = self.data_collector.get_symbols_with_sufficient_data(target_date)[:30]
        
        strategy_candidates = []
        
        for symbol in symbols:
            try:
                # 戦略シグナル取得
                signal_result = self.tv_strategies.get_trading_signal(symbol, analysis_time)
                
                if (signal_result['final_signal'] != StrategySignal.HOLD and 
                    signal_result['confidence'] > 0.7):
                    
                    direction = 1 if signal_result['final_signal'] == StrategySignal.BUY else -1
                    
                    strategy_candidates.append({
                        'symbol': symbol,
                        'method': DemoMethod.METHOD_2_STRATEGY,
                        'direction': direction,
                        'confidence': signal_result['confidence'],
                        'strategy_count': signal_result['strategy_count'],
                        'buy_weight': signal_result['buy_weight'],
                        'sell_weight': signal_result['sell_weight'],
                        'reason': f"戦略シグナル ({signal_result['strategy_count']}戦略, C:{signal_result['confidence']:.3f})"
                    })
                    
            except Exception as e:
                logging.error(f"戦略分析エラー {symbol}: {e}")
                continue
        
        # 信頼度でソート
        strategy_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        logging.info(f"戦略候補: {len(strategy_candidates)}銘柄 (分析日: {analysis_date})")
        return strategy_candidates[:10]
    
    def get_market_data_for_date(self, symbol: str, target_date: date) -> Optional[Dict]:
        """指定日の市場データ取得"""
        try:
            conn = sqlite3.connect(self.data_collector.db_path)
            
            # その日の代表的なデータ（寄付き近くの価格）
            query = '''
                SELECT close_price, volume, ma_5min, ma_20min, ma_60min
                FROM five_minute_data
                WHERE symbol = ? AND DATE(timestamp) = ?
                ORDER BY timestamp
                LIMIT 1
            '''
            
            result = conn.execute(query, (symbol, target_date)).fetchone()
            conn.close()
            
            if result:
                return {
                    'close_price': result[0],
                    'volume': result[1],
                    'ma_5min': result[2],
                    'ma_20min': result[3],
                    'ma_60min': result[4]
                }
            
        except Exception as e:
            logging.error(f"市場データ取得エラー {symbol} {target_date}: {e}")
        
        return None
    
    def execute_entry_decision(self, candidate: Dict, entry_date: date) -> Optional[TradingSignal]:
        """エントリー判定実行"""
        symbol = candidate['symbol']
        method = candidate['method']
        
        # 前日データ取得
        market_data = self.get_market_data_for_date(symbol, entry_date)
        if not market_data:
            return None
        
        if method == DemoMethod.METHOD_1_ML:
            return self.execute_ml_entry(candidate, market_data, entry_date)
        elif method == DemoMethod.METHOD_2_STRATEGY:
            return self.execute_strategy_entry(candidate, market_data, entry_date)
        else:
            return None
    
    def execute_ml_entry(self, candidate: Dict, market_data: Dict, entry_date: date) -> Optional[TradingSignal]:
        """ML方法のエントリー実行"""
        # 前日14:00想定でMLチェック
        entry_time = datetime.combine(entry_date, datetime.min.time().replace(hour=14))
        
        try:
            result = self.ml_models.fact_check_predictions(candidate['symbol'], entry_time)
            
            if result['should_trade'] and result['confidence'] >= 0.8:
                if result['direction'] == 1:
                    return TradingSignal.BUY
                else:
                    return TradingSignal.SELL
                    
        except Exception as e:
            logging.error(f"MLエントリー判定エラー: {e}")
        
        return TradingSignal.HOLD
    
    def execute_strategy_entry(self, candidate: Dict, market_data: Dict, entry_date: date) -> Optional[TradingSignal]:
        """戦略方法のエントリー実行"""
        entry_time = datetime.combine(entry_date, datetime.min.time().replace(hour=14))
        
        try:
            signal_result = self.tv_strategies.get_trading_signal(candidate['symbol'], entry_time)
            
            if (signal_result['final_signal'] != StrategySignal.HOLD and 
                signal_result['confidence'] > 0.7):
                
                if signal_result['final_signal'] == StrategySignal.BUY:
                    return TradingSignal.BUY
                else:
                    return TradingSignal.SELL
                    
        except Exception as e:
            logging.error(f"戦略エントリー判定エラー: {e}")
        
        return TradingSignal.HOLD

class EnhancedDemoTradingSimulator:
    """拡張デモ取引シミュレーター"""
    
    def __init__(self, config, data_collector: EnhancedHistoricalDataCollector):
        self.config = config
        self.data_collector = data_collector
        self.strategy = EnhancedDemoTradingStrategy(config, data_collector)
        
        # 各方法ごとのセッション
        self.sessions = {
            DemoMethod.METHOD_1_ML: {
                'name': f"ml_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'capital': config.get('capital', 2000000),
                'positions': {},
                'trade_history': [],
                'daily_pnl': []
            },
            DemoMethod.METHOD_2_STRATEGY: {
                'name': f"strategy_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'capital': config.get('capital', 2000000),
                'positions': {},
                'trade_history': [],
                'daily_pnl': []
            }
        }
        
    def run_enhanced_demo_simulation(self, start_date: date, end_date: date, methods: List[DemoMethod]):
        """拡張デモシミュレーション実行"""
        logging.info(f"拡張デモシミュレーション開始: {start_date} - {end_date}")
        logging.info(f"実行方法: {[m.value for m in methods]}")
        
        for method in methods:
            logging.info(f"\n{'='*60}")
            logging.info(f"{method.value} シミュレーション開始")
            logging.info(f"{'='*60}")
            
            self.run_method_simulation(method, start_date, end_date)
            
        # 結果比較
        self.compare_methods(methods)
    
    def run_method_simulation(self, method: DemoMethod, start_date: date, end_date: date):
        """特定方法のシミュレーション実行"""
        session = self.sessions[method]
        current_date = start_date
        
        while current_date <= end_date:
            try:
                self.process_trading_day_enhanced(method, current_date)
                current_date += timedelta(days=1)
                
                # 土日スキップ
                while current_date.weekday() >= 5 and current_date <= end_date:
                    current_date += timedelta(days=1)
                    
            except Exception as e:
                logging.error(f"取引日処理エラー {current_date}: {e}")
                current_date += timedelta(days=1)
        
        # 最終決済
        self.close_all_positions_enhanced(method, current_date)
        
        # 結果分析
        self.analyze_method_results(method)
        
        # データベース保存
        self.save_session_results_enhanced(method)
    
    def process_trading_day_enhanced(self, method: DemoMethod, target_date: date):
        """拡張取引日処理"""
        session = self.sessions[method]
        
        logging.info(f"\n=== {target_date} {method.value} デモ取引 ===")
        
        # 候補分析
        if method == DemoMethod.METHOD_1_ML:
            candidates = self.strategy.analyze_ml_candidates(target_date)
        elif method == DemoMethod.METHOD_2_STRATEGY:
            candidates = self.strategy.analyze_strategy_candidates(target_date)
        else:
            candidates = []
        
        if not candidates:
            logging.info("候補銘柄なし")
            return
        
        logging.info(f"候補銘柄: {len(candidates)}銘柄")
        
        # 新規エントリー判定
        for candidate in candidates[:3]:  # 上位3銘柄
            symbol = candidate['symbol']
            
            if symbol in session['positions']:
                continue
            
            # エントリー判定（前日データ使用）
            entry_date = target_date - timedelta(days=1)
            signal = self.strategy.execute_entry_decision(candidate, entry_date)
            
            if signal != TradingSignal.HOLD:
                self.enter_enhanced_position(method, symbol, signal, target_date, candidate)
        
        # 既存ポジション管理
        self.manage_enhanced_positions(method, target_date)
    
    def enter_enhanced_position(self, method: DemoMethod, symbol: str, signal: TradingSignal, 
                              entry_date: date, candidate: Dict):
        """拡張ポジションエントリー"""
        try:
            session = self.sessions[method]
            
            # エントリー価格取得（当日寄付き想定）
            market_data = self.strategy.get_market_data_for_date(symbol, entry_date)
            if not market_data:
                return
            
            entry_price = market_data['close_price']
            if entry_price <= 0:
                return
            
            # ポジションサイズ計算
            quantity = self.calculate_position_size_enhanced(session['capital'], entry_price)
            required_capital = quantity * entry_price
            
            if required_capital > session['capital'] * 0.8:
                logging.warning(f"資金不足: {symbol} 必要資金: {required_capital:,.0f}円")
                return
            
            # 拡張ポジション作成
            position = EnhancedDemoPosition(
                symbol=symbol,
                side=signal.value,
                quantity=quantity,
                entry_price=entry_price,
                entry_date=entry_date,
                entry_vwap=market_data.get('close_price', entry_price),  # VWAP代用
                method=method,
                entry_reason=candidate['reason'],
                confidence=candidate['confidence'],
                ml_hourly_prediction=candidate.get('hourly_prediction', 0.0),
                ml_minute_prediction=candidate.get('minute_prediction', 0),
                strategy_signals=str(candidate) if method == DemoMethod.METHOD_2_STRATEGY else ""
            )
            
            session['positions'][symbol] = position
            session['capital'] -= required_capital
            
            logging.info(f"エントリー: {symbol} {signal.value} {quantity}株 @{entry_price:.0f} "
                        f"信頼度:{candidate['confidence']:.3f} 理由:{candidate['reason']}")
            
        except Exception as e:
            logging.error(f"エントリーエラー {symbol}: {e}")
    
    def calculate_position_size_enhanced(self, capital: float, price: float) -> int:
        """拡張ポジションサイズ計算"""
        position_value = capital * 0.25  # 25%ずつ投入
        quantity = int(position_value / price / 100) * 100  # 100株単位
        return max(quantity, 100)
    
    def manage_enhanced_positions(self, method: DemoMethod, current_date: date):
        """拡張ポジション管理"""
        session = self.sessions[method]
        
        for symbol in list(session['positions'].keys()):
            position = session['positions'][symbol]
            
            try:
                # 当日の価格データ取得
                market_data = self.strategy.get_market_data_for_date(symbol, current_date)
                if not market_data:
                    continue
                
                current_price = market_data['close_price']
                position.current_price = current_price
                
                # 損益計算
                if position.side == 'buy':
                    position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
                
                # 決済判定
                days_held = (current_date - position.entry_date).days
                profit_rate = position.unrealized_pnl / (position.entry_price * position.quantity)
                
                should_close = False
                close_reason = ""
                
                # 1日経過で自動決済
                if days_held >= 1:
                    should_close = True
                    close_reason = "1日保有"
                # 2%利確
                elif profit_rate >= 0.02:
                    should_close = True
                    close_reason = "利確"
                # 0.8%ロスカット
                elif profit_rate <= -0.008:
                    should_close = True
                    close_reason = "ロスカット"
                
                if should_close:
                    self.close_enhanced_position(method, symbol, current_price, current_date, close_reason)
                    
            except Exception as e:
                logging.error(f"ポジション管理エラー {symbol}: {e}")
    
    def close_enhanced_position(self, method: DemoMethod, symbol: str, exit_price: float, 
                               exit_date: date, reason: str):
        """拡張ポジション決済"""
        try:
            session = self.sessions[method]
            position = session['positions'][symbol]
            
            # 損益計算
            if position.side == 'buy':
                realized_pnl = (exit_price - position.entry_price) * position.quantity
            else:
                realized_pnl = (position.entry_price - exit_price) * position.quantity
            
            # 資金回収
            recovered_capital = exit_price * position.quantity
            session['capital'] += recovered_capital
            
            # ポジション更新
            position.exit_price = exit_price
            position.exit_date = exit_date
            position.realized_pnl = realized_pnl
            position.close_reason = reason
            position.status = "closed"
            
            # 取引履歴に追加
            session['trade_history'].append(position)
            
            return_rate = realized_pnl / (position.entry_price * position.quantity)
            
            logging.info(f"決済: {symbol} {reason} 損益: {realized_pnl:,.0f}円 (利回り: {return_rate:.1%})")
            
            # ポジション削除
            del session['positions'][symbol]
            
        except Exception as e:
            logging.error(f"決済エラー {symbol}: {e}")
    
    def close_all_positions_enhanced(self, method: DemoMethod, final_date: date):
        """全ポジション強制決済"""
        session = self.sessions[method]
        
        for symbol in list(session['positions'].keys()):
            try:
                market_data = self.strategy.get_market_data_for_date(symbol, final_date)
                if market_data:
                    self.close_enhanced_position(method, symbol, market_data['close_price'], 
                                                final_date, "シミュレーション終了")
            except Exception as e:
                logging.error(f"強制決済エラー {symbol}: {e}")
    
    def analyze_method_results(self, method: DemoMethod):
        """方法別結果分析"""
        session = self.sessions[method]
        trade_history = session['trade_history']
        
        logging.info(f"\n{'='*50}")
        logging.info(f"{method.value} 結果分析")
        logging.info(f"{'='*50}")
        
        if not trade_history:
            logging.info("取引履歴なし")
            return
        
        # 基本統計
        total_trades = len(trade_history)
        winning_trades = len([t for t in trade_history if t.realized_pnl > 0])
        total_pnl = sum([t.realized_pnl for t in trade_history])
        
        initial_capital = self.config.get('capital', 2000000)
        final_capital = session['capital']
        total_return = (final_capital - initial_capital) / initial_capital
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_confidence = np.mean([t.confidence for t in trade_history])
        
        logging.info(f"📊 {method.value} 統計:")
        logging.info(f"   総取引数: {total_trades}")
        logging.info(f"   勝率: {win_rate:.1%} ({winning_trades}/{total_trades})")
        logging.info(f"   総損益: {total_pnl:,.0f}円")
        logging.info(f"   利回り: {total_return:.2%}")
        logging.info(f"   平均信頼度: {avg_confidence:.3f}")
        
        if total_trades > 0:
            profits = [t.realized_pnl for t in trade_history if t.realized_pnl > 0]
            losses = [t.realized_pnl for t in trade_history if t.realized_pnl < 0]
            
            avg_profit = np.mean(profits) if profits else 0
            avg_loss = np.mean(losses) if losses else 0
            
            logging.info(f"   平均利益: {avg_profit:,.0f}円")
            logging.info(f"   平均損失: {avg_loss:,.0f}円")
            
            if avg_loss != 0:
                profit_factor = abs(sum(profits) / sum(losses)) if losses else float('inf')
                logging.info(f"   プロフィットファクター: {profit_factor:.2f}")
    
    def compare_methods(self, methods: List[DemoMethod]):
        """方法比較"""
        logging.info(f"\n{'='*60}")
        logging.info("方法比較結果")
        logging.info(f"{'='*60}")
        
        comparison_data = []
        
        for method in methods:
            session = self.sessions[method]
            trade_history = session['trade_history']
            
            if trade_history:
                total_pnl = sum([t.realized_pnl for t in trade_history])
                win_rate = len([t for t in trade_history if t.realized_pnl > 0]) / len(trade_history)
                avg_confidence = np.mean([t.confidence for t in trade_history])
                
                initial_capital = self.config.get('capital', 2000000)
                return_rate = (session['capital'] - initial_capital) / initial_capital
                
                comparison_data.append({
                    'method': method.value,
                    'trades': len(trade_history),
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'return_rate': return_rate,
                    'avg_confidence': avg_confidence
                })
        
        # 比較表示
        for data in comparison_data:
            logging.info(f"\n【{data['method']}】")
            logging.info(f"  取引数: {data['trades']}")
            logging.info(f"  勝率: {data['win_rate']:.1%}")
            logging.info(f"  総損益: {data['total_pnl']:,.0f}円")
            logging.info(f"  利回り: {data['return_rate']:.2%}")
            logging.info(f"  平均信頼度: {data['avg_confidence']:.3f}")
        
        # 最適方法判定
        if len(comparison_data) > 1:
            best_method = max(comparison_data, key=lambda x: x['return_rate'])
            logging.info(f"\n🏆 最優秀方法: {best_method['method']} (利回り: {best_method['return_rate']:.2%})")
    
    def save_session_results_enhanced(self, method: DemoMethod):
        """拡張セッション結果保存"""
        try:
            session = self.sessions[method]
            trade_history = session['trade_history']
            
            conn = sqlite3.connect(self.data_collector.db_path)
            cursor = conn.cursor()
            
            # セッション情報保存
            if trade_history:
                start_date = min([t.entry_date for t in trade_history])
                end_date = max([t.exit_date for t in trade_history if t.exit_date])
                total_trades = len(trade_history)
                winning_trades = len([t for t in trade_history if t.realized_pnl > 0])
                total_pnl = sum([t.realized_pnl for t in trade_history])
                avg_confidence = np.mean([t.confidence for t in trade_history])
                
                cursor.execute('''
                    INSERT OR REPLACE INTO enhanced_demo_sessions
                    (session_name, start_date, end_date, initial_capital, final_capital,
                     method, total_trades, winning_trades, total_pnl, avg_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session['name'], start_date, end_date,
                    self.config.get('capital', 2000000), session['capital'],
                    method.value, total_trades, winning_trades, total_pnl, avg_confidence
                ))
                
                # 取引履歴保存
                for trade in trade_history:
                    cursor.execute('''
                        INSERT INTO enhanced_demo_trades VALUES
                        (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        trade.symbol, trade.side, trade.quantity, trade.entry_price,
                        trade.entry_date, trade.entry_vwap, trade.exit_price, trade.exit_date,
                        trade.realized_pnl, trade.method.value, trade.entry_reason,
                        trade.confidence, trade.close_reason, trade.ml_hourly_prediction,
                        trade.ml_minute_prediction, trade.strategy_signals, session['name'],
                        datetime.now()
                    ))
            
            conn.commit()
            conn.close()
            
            logging.info(f"{method.value} セッション結果保存完了")
            
        except Exception as e:
            logging.error(f"セッション結果保存エラー: {e}")

def load_enhanced_config():
    """拡張設定読み込み"""
    default_config = {
        'capital': 2000000,
        'max_positions': 3,
        'demo_mode': True,
        'methods': ['ml_factcheck', 'strategy_based']
    }
    
    try:
        with open('enhanced_demo_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        return {**default_config, **config}
    except Exception as e:
        logging.warning(f"設定ファイル読み込みエラー: {e}")
        return default_config

def main():
    """メイン関数"""
    print("=" * 70)
    print("統合デモトレードシステム - 方法1、方法2対応版")
    print("=" * 70)
    print("1. メインDBからデータインポート")
    print("2. 方法1（ML）デモ実行")
    print("3. 方法2（戦略）デモ実行")
    print("4. 両方法比較実行")
    print("5. 過去結果確認")
    print("6. データベース状況確認")
    print("7. 終了")
    
    # 初期化
    config = load_enhanced_config()
    data_collector = EnhancedHistoricalDataCollector()
    
    while True:
        try:
            choice = input("\n選択してください (1-7): ").strip()
            
            if choice == '1':
                print("📥 メインデータベースからインポート中...")
                data_collector.import_from_main_database()
                
                min_date, max_date = data_collector.get_available_data_range()
                if min_date and max_date:
                    print(f"✅ インポート完了: {min_date} - {max_date}")
                else:
                    print("❌ データがありません")
            
            elif choice in ['2', '3', '4']:
                min_date, max_date = data_collector.get_available_data_range()
                if not min_date or not max_date:
                    print("❌ データが不足しています。まずインポートを実行してください")
                    continue
                
                # 実行期間設定（最新の5日間）
                end_date = max_date
                start_date = max(min_date, end_date - timedelta(days=5))
                
                print(f"📊 シミュレーション期間: {start_date} - {end_date}")
                
                simulator = EnhancedDemoTradingSimulator(config, data_collector)
                
                if choice == '2':
                    methods = [DemoMethod.METHOD_1_ML]
                elif choice == '3':
                    methods = [DemoMethod.METHOD_2_STRATEGY]
                else:  # choice == '4'
                    methods = [DemoMethod.METHOD_1_ML, DemoMethod.METHOD_2_STRATEGY]
                
                simulator.run_enhanced_demo_simulation(start_date, end_date, methods)
                print("✅ シミュレーション完了")
            
            elif choice == '5':
                # 過去結果表示
                try:
                    conn = sqlite3.connect(data_collector.db_path)
                    
                    query = '''
                        SELECT session_name, method, start_date, end_date, 
                               total_trades, winning_trades, total_pnl, avg_confidence
                        FROM enhanced_demo_sessions
                        ORDER BY created_at DESC
                        LIMIT 10
                    '''
                    
                    df = pd.read_sql_query(query, conn)
                    conn.close()
                    
                    if not df.empty:
                        print("\n📋 過去のデモ結果:")
                        for _, row in df.iterrows():
                            win_rate = row['winning_trades'] / row['total_trades'] if row['total_trades'] > 0 else 0
                            print(f"  {row['session_name']} ({row['method']})")
                            print(f"    期間: {row['start_date']} - {row['end_date']}")
                            print(f"    成績: {win_rate:.1%} ({row['winning_trades']}/{row['total_trades']})")
                            print(f"    損益: {row['total_pnl']:,.0f}円")
                            print(f"    信頼度: {row['avg_confidence']:.3f}")
                            print()
                    else:
                        print("過去のデモ結果がありません")
                        
                except Exception as e:
                    print(f"❌ 結果表示エラー: {e}")
            
            elif choice == '6':
                # データベース状況確認
                try:
                    conn = sqlite3.connect(data_collector.db_path)
                    
                    # データ件数確認
                    five_min_count = conn.execute("SELECT COUNT(*) FROM five_minute_data").fetchone()[0]
                    symbols_count = conn.execute("SELECT COUNT(DISTINCT symbol) FROM five_minute_data").fetchone()[0]
                    
                    min_date, max_date = data_collector.get_available_data_range()
                    
                    conn.close()
                    
                    print(f"\n📊 データベース状況:")
                    print(f"  5分足データ: {five_min_count:,}件")
                    print(f"  銘柄数: {symbols_count}")
                    print(f"  データ期間: {min_date} - {max_date}")
                    
                except Exception as e:
                    print(f"❌ 状況確認エラー: {e}")
            
            elif choice == '7':
                print("👋 終了します")
                break
            
            else:
                print("❌ 無効な選択です")
                
        except KeyboardInterrupt:
            print("\n👋 終了します")
            break
        except Exception as e:
            print(f"❌ エラー: {e}")

if __name__ == "__main__":
    main()
