"""
ファンダメンタルズデータ取得システム
財務データ、業績データ、バリュエーション指標を取得・管理
"""

import pandas as pd
import numpy as np
import requests
import sqlite3
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yfinance as yf
import json
import time
from pathlib import Path

@dataclass
class FundamentalData:
    """ファンダメンタルズデータ"""
    symbol: str
    # バリュエーション指標
    per: float = 0.0
    pbr: float = 0.0
    psr: float = 0.0  # 株価売上高倍率
    dividend_yield: float = 0.0
    
    # 財務指標
    roe: float = 0.0
    roa: float = 0.0
    debt_ratio: float = 0.0
    current_ratio: float = 0.0
    
    # 業績指標
    revenue_growth: float = 0.0
    profit_growth: float = 0.0
    operating_margin: float = 0.0
    
    # 市場データ
    market_cap: float = 0.0
    beta: float = 0.0
    
    # 更新日時
    updated_at: datetime = None

class FundamentalDataCollector:
    """ファンダメンタルズデータ収集器"""
    
    def __init__(self, db_path: str = "fundamental_data.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.init_database()
        
        # 日本株のsuffix
        self.japan_suffix = ".T"
        
    def init_database(self):
        """データベース初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fundamental_data (
                symbol TEXT,
                date DATE,
                per REAL,
                pbr REAL,
                psr REAL,
                dividend_yield REAL,
                roe REAL,
                roa REAL,
                debt_ratio REAL,
                current_ratio REAL,
                revenue_growth REAL,
                profit_growth REAL,
                operating_margin REAL,
                market_cap REAL,
                beta REAL,
                updated_at DATETIME,
                PRIMARY KEY (symbol, date)
            )
        ''')
        
        # 業界別平均テーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sector_averages (
                sector TEXT,
                date DATE,
                avg_per REAL,
                avg_pbr REAL,
                avg_roe REAL,
                avg_debt_ratio REAL,
                updated_at DATETIME,
                PRIMARY KEY (sector, date)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        self.logger.info("ファンダメンタルズデータベース初期化完了")
    
    def get_fundamental_data_yfinance(self, symbol: str) -> Optional[FundamentalData]:
        """Yahoo Financeからファンダメンタルズデータ取得"""
        try:
            # 日本株のティッカー形式に変換
            ticker = symbol + self.japan_suffix
            stock = yf.Ticker(ticker)
            
            # 基本情報
            info = stock.info
            
            # 財務諸表
            try:
                financials = stock.financials
                balance_sheet = stock.balance_sheet
                cash_flow = stock.cashflow
            except:
                financials = pd.DataFrame()
                balance_sheet = pd.DataFrame()
                cash_flow = pd.DataFrame()
            
            # データ抽出
            data = FundamentalData(
                symbol=symbol,
                per=info.get('trailingPE', 0.0) or 0.0,
                pbr=info.get('priceToBook', 0.0) or 0.0,
                psr=info.get('priceToSalesTrailing12Months', 0.0) or 0.0,
                dividend_yield=info.get('dividendYield', 0.0) or 0.0,
                roe=info.get('returnOnEquity', 0.0) or 0.0,
                roa=info.get('returnOnAssets', 0.0) or 0.0,
                market_cap=info.get('marketCap', 0.0) or 0.0,
                beta=info.get('beta', 0.0) or 0.0,
                updated_at=datetime.now()
            )
            
            # 財務諸表から計算
            if not balance_sheet.empty:
                try:
                    # 流動比率
                    current_assets = balance_sheet.loc['Current Assets'].iloc[0] if 'Current Assets' in balance_sheet.index else 0
                    current_liabilities = balance_sheet.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance_sheet.index else 0
                    data.current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
                    
                    # 負債比率
                    total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
                    total_equity = balance_sheet.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in balance_sheet.index else 0
                    data.debt_ratio = total_debt / total_equity if total_equity > 0 else 0
                    
                except Exception as e:
                    self.logger.warning(f"財務比率計算エラー {symbol}: {e}")
            
            # 成長率計算
            if not financials.empty:
                try:
                    # 売上成長率
                    revenues = financials.loc['Total Revenue'] if 'Total Revenue' in financials.index else pd.Series()
                    if len(revenues) >= 2:
                        data.revenue_growth = (revenues.iloc[0] - revenues.iloc[1]) / revenues.iloc[1] if revenues.iloc[1] > 0 else 0
                    
                    # 営業利益率
                    operating_income = financials.loc['Operating Income'] if 'Operating Income' in financials.index else pd.Series()
                    if len(revenues) > 0 and len(operating_income) > 0:
                        data.operating_margin = operating_income.iloc[0] / revenues.iloc[0] if revenues.iloc[0] > 0 else 0
                        
                except Exception as e:
                    self.logger.warning(f"成長率計算エラー {symbol}: {e}")
            
            self.logger.info(f"ファンダメンタルズデータ取得完了: {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"ファンダメンタルズデータ取得エラー {symbol}: {e}")
            return None
    
    def get_fundamental_data_excel(self, symbol: str) -> Optional[FundamentalData]:
        """Excel/MarketSpeedからファンダメンタルズデータ取得"""
        try:
            # 楽天証券MarketSpeedⅡのデータ取得
            # 実際の実装では、ExcelのMSGETDATA関数を使用
            
            # デモ用の模擬データ
            return FundamentalData(
                symbol=symbol,
                per=np.random.uniform(5, 30),
                pbr=np.random.uniform(0.5, 3.0),
                psr=np.random.uniform(0.5, 5.0),
                dividend_yield=np.random.uniform(0.01, 0.05),
                roe=np.random.uniform(0.05, 0.25),
                roa=np.random.uniform(0.02, 0.15),
                debt_ratio=np.random.uniform(0.2, 0.8),
                current_ratio=np.random.uniform(1.0, 3.0),
                revenue_growth=np.random.uniform(-0.1, 0.3),
                profit_growth=np.random.uniform(-0.2, 0.5),
                operating_margin=np.random.uniform(0.02, 0.20),
                market_cap=np.random.uniform(1000000000, 10000000000),
                beta=np.random.uniform(0.5, 2.0),
                updated_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Excel データ取得エラー {symbol}: {e}")
            return None
    
    def collect_fundamental_data(self, symbols: List[str], use_excel: bool = False) -> Dict[str, FundamentalData]:
        """複数銘柄のファンダメンタルズデータ収集"""
        results = {}
        
        for symbol in symbols:
            try:
                if use_excel:
                    data = self.get_fundamental_data_excel(symbol)
                else:
                    data = self.get_fundamental_data_yfinance(symbol)
                
                if data:
                    results[symbol] = data
                    self.save_fundamental_data(data)
                
                # API制限対策
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"データ収集エラー {symbol}: {e}")
                continue
        
        self.logger.info(f"ファンダメンタルズデータ収集完了: {len(results)}/{len(symbols)} 銘柄")
        return results
    
    def save_fundamental_data(self, data: FundamentalData):
        """ファンダメンタルズデータをDBに保存"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO fundamental_data 
                (symbol, date, per, pbr, psr, dividend_yield, roe, roa, debt_ratio, 
                 current_ratio, revenue_growth, profit_growth, operating_margin, 
                 market_cap, beta, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.symbol, date.today(), data.per, data.pbr, data.psr, data.dividend_yield,
                data.roe, data.roa, data.debt_ratio, data.current_ratio,
                data.revenue_growth, data.profit_growth, data.operating_margin,
                data.market_cap, data.beta, data.updated_at
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"データ保存エラー: {e}")
    
    def get_fundamental_data_from_db(self, symbol: str, target_date: date = None) -> Optional[FundamentalData]:
        """DBからファンダメンタルズデータ取得"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if target_date is None:
                target_date = date.today()
            
            cursor.execute('''
                SELECT * FROM fundamental_data 
                WHERE symbol = ? AND date <= ?
                ORDER BY date DESC
                LIMIT 1
            ''', (symbol, target_date))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return FundamentalData(
                    symbol=row[0],
                    per=row[2], pbr=row[3], psr=row[4], dividend_yield=row[5],
                    roe=row[6], roa=row[7], debt_ratio=row[8], current_ratio=row[9],
                    revenue_growth=row[10], profit_growth=row[11], operating_margin=row[12],
                    market_cap=row[13], beta=row[14], updated_at=datetime.fromisoformat(row[15])
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"DB取得エラー {symbol}: {e}")
            return None
    
    def calculate_sector_averages(self):
        """業界別平均値計算"""
        try:
            # prime_symbols.csvから業界情報取得
            symbols_df = pd.read_csv("prime_symbols.csv")
            
            conn = sqlite3.connect(self.db_path)
            
            # 最新データ取得
            query = '''
                SELECT f.*, s.sector
                FROM fundamental_data f
                JOIN (
                    SELECT symbol, MAX(date) as max_date
                    FROM fundamental_data
                    GROUP BY symbol
                ) latest ON f.symbol = latest.symbol AND f.date = latest.max_date
                LEFT JOIN ... -- prime_symbols.csvとの結合は実装時に調整
            '''
            
            # 業界別集計
            sector_averages = {}
            
            for sector in symbols_df['sector'].unique():
                sector_symbols = symbols_df[symbols_df['sector'] == sector]['symbol'].astype(str).tolist()
                
                # 各業界の平均値計算
                fundamental_data = []
                for symbol in sector_symbols:
                    data = self.get_fundamental_data_from_db(symbol)
                    if data:
                        fundamental_data.append({
                            'per': data.per,
                            'pbr': data.pbr,
                            'roe': data.roe,
                            'debt_ratio': data.debt_ratio
                        })
                
                if fundamental_data:
                    df = pd.DataFrame(fundamental_data)
                    sector_averages[sector] = {
                        'avg_per': df['per'].mean(),
                        'avg_pbr': df['pbr'].mean(),
                        'avg_roe': df['roe'].mean(),
                        'avg_debt_ratio': df['debt_ratio'].mean()
                    }
            
            # 業界平均をDBに保存
            cursor = conn.cursor()
            for sector, avgs in sector_averages.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO sector_averages 
                    (sector, date, avg_per, avg_pbr, avg_roe, avg_debt_ratio, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (sector, date.today(), avgs['avg_per'], avgs['avg_pbr'], 
                      avgs['avg_roe'], avgs['avg_debt_ratio'], datetime.now()))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"業界別平均計算完了: {len(sector_averages)} 業界")
            return sector_averages
            
        except Exception as e:
            self.logger.error(f"業界別平均計算エラー: {e}")
            return {}
    
    def get_relative_valuation(self, symbol: str) -> Dict:
        """相対評価指標計算"""
        try:
            # 銘柄データ取得
            data = self.get_fundamental_data_from_db(symbol)
            if not data:
                return {}
            
            # 業界取得
            symbols_df = pd.read_csv("prime_symbols.csv")
            symbol_info = symbols_df[symbols_df['symbol'] == int(symbol)]
            if symbol_info.empty:
                return {}
            
            sector = symbol_info['sector'].iloc[0]
            
            # 業界平均取得
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT avg_per, avg_pbr, avg_roe, avg_debt_ratio
                FROM sector_averages
                WHERE sector = ?
                ORDER BY date DESC
                LIMIT 1
            ''', (sector,))
            
            sector_avg = cursor.fetchone()
            conn.close()
            
            if not sector_avg:
                return {}
            
            # 相対評価計算
            relative_metrics = {
                'per_ratio': data.per / sector_avg[0] if sector_avg[0] > 0 else 0,
                'pbr_ratio': data.pbr / sector_avg[1] if sector_avg[1] > 0 else 0,
                'roe_ratio': data.roe / sector_avg[2] if sector_avg[2] > 0 else 0,
                'debt_ratio_ratio': data.debt_ratio / sector_avg[3] if sector_avg[3] > 0 else 0,
                'sector': sector
            }
            
            return relative_metrics
            
        except Exception as e:
            self.logger.error(f"相対評価計算エラー {symbol}: {e}")
            return {}

def main():
    """メイン実行"""
    logging.basicConfig(level=logging.INFO)
    
    # サンプル銘柄
    symbols = ["7203", "9984", "6758", "8306", "6501"]
    
    collector = FundamentalDataCollector()
    
    # データ収集
    print("ファンダメンタルズデータ収集開始...")
    data = collector.collect_fundamental_data(symbols)
    
    # 業界別平均計算
    print("業界別平均計算...")
    sector_averages = collector.calculate_sector_averages()
    
    # 相対評価
    print("相対評価計算...")
    for symbol in symbols:
        relative = collector.get_relative_valuation(symbol)
        if relative:
            print(f"{symbol}: PER比 {relative['per_ratio']:.2f}, PBR比 {relative['pbr_ratio']:.2f}")

if __name__ == "__main__":
    main()
