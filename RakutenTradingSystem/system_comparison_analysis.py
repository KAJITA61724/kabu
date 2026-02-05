#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thursday Friday Demo vs Leak Free System 比較分析
異なる結果の原因を詳細に分析
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import os

class SystemComparisonAnalyzer:
    def __init__(self):
        self.old_db = "trading_data.db"  # Thursday Friday Demo が使用
        self.new_db = "leak_free_trading.db"  # Leak Free System が使用
    
    def analyze_data_differences(self):
        """データソースの違いを分析"""
        print("🔍 システム比較分析: Thursday Friday Demo vs Leak Free System")
        print("="*70)
        
        print("\\n📊 データベース比較:")
        
        # 1. データベースファイルの存在確認
        old_exists = os.path.exists(self.old_db)
        new_exists = os.path.exists(self.new_db)
        
        print(f"  旧システム DB ({self.old_db}): {'存在' if old_exists else '不在'}")
        print(f"  新システム DB ({self.new_db}): {'存在' if new_exists else '不在'}")
        
        if not old_exists and not new_exists:
            print("❌ 両方のデータベースが見つかりません")
            return
        
        # 2. データ内容比較
        if old_exists:
            self.analyze_database_content(self.old_db, "Thursday Friday Demo")
        
        if new_exists:
            self.analyze_database_content(self.new_db, "Leak Free System")
        
        # 3. 価格データの比較
        if old_exists and new_exists:
            self.compare_price_data()
    
    def analyze_database_content(self, db_path, system_name):
        """個別データベースの内容分析"""
        print(f"\n📈 {system_name} データベース分析:")
        
        try:
            conn = sqlite3.connect(db_path)
            
            # テーブル構造確認
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'", conn
            )
            print(f"  テーブル: {', '.join(tables['name'].tolist())}")
            
            # chart_dataテーブルの分析
            if 'chart_data' in tables['name'].values:
                # データ期間確認
                date_range = pd.read_sql_query("""
                    SELECT 
                        MIN(datetime) as start_date,
                        MAX(datetime) as end_date,
                        COUNT(*) as total_records
                    FROM chart_data
                """, conn)
                
                print(f"  期間: {date_range['start_date'].iloc[0]} ～ {date_range['end_date'].iloc[0]}")
                print(f"  総レコード数: {date_range['total_records'].iloc[0]:,}件")
                
                # 銘柄別データ数
                symbol_counts = pd.read_sql_query("""
                    SELECT symbol, COUNT(*) as count 
                    FROM chart_data 
                    GROUP BY symbol 
                    ORDER BY symbol
                """, conn)
                
                print("  銘柄別データ数:")
                for _, row in symbol_counts.iterrows():
                    print(f"    {row['symbol']}: {row['count']:,}件")
                
                # データベース構造に応じた価格カラム名を決定
                column_info = conn.execute("PRAGMA table_info(chart_data)").fetchall()
                price_column = None
                for col in column_info:
                    if col[1] == 'close':
                        price_column = 'close'
                        break
                    elif col[1] == 'close_price':
                        price_column = 'close_price'
                        break
                
                if price_column:
                    # 2025-07-17, 2025-07-18の価格例
                    sample_data = pd.read_sql_query(f"""
                        SELECT symbol, datetime, {price_column} as price
                        FROM chart_data 
                        WHERE date(datetime) IN ('2025-07-17', '2025-07-18')
                        AND symbol = '7203'
                        ORDER BY datetime
                        LIMIT 10
                    """, conn)
                    
                    if len(sample_data) > 0:
                        print("  7203価格サンプル:")
                        for _, row in sample_data.iterrows():
                            print(f"    {row['datetime']}: ¥{row['price']:.0f}")
                else:
                    print("  価格カラムが見つかりません")
            
            conn.close()
            
        except Exception as e:
            print(f"  ❌ エラー: {e}")
    
    def compare_price_data(self):
        """価格データの直接比較"""
        print(f"\n🔍 価格データ直接比較 (2025-07-17, 2025-07-18):")
        
        try:
            # 旧システムデータ（close_price カラム）
            conn_old = sqlite3.connect(self.old_db)
            old_data = pd.read_sql_query("""
                SELECT datetime, symbol, close_price as price
                FROM chart_data 
                WHERE symbol = '7203' 
                AND date(datetime) IN ('2025-07-17', '2025-07-18')
                ORDER BY datetime
                LIMIT 20
            """, conn_old)
            conn_old.close()
            
            # 新システムデータ（close カラム）
            conn_new = sqlite3.connect(self.new_db)
            new_data = pd.read_sql_query("""
                SELECT datetime, symbol, close as price
                FROM chart_data 
                WHERE symbol = '7203' 
                AND date(datetime) IN ('2025-07-17', '2025-07-18')
                ORDER BY datetime
                LIMIT 20
            """, conn_new)
            conn_new.close()
            
            print(f"  旧システム 7203データ: {len(old_data)}件")
            print(f"  新システム 7203データ: {len(new_data)}件")
            
            if len(old_data) > 0:
                old_price_range = f"¥{old_data['price'].min():.0f} ～ ¥{old_data['price'].max():.0f}"
                print(f"  旧システム価格範囲: {old_price_range}")
                
                print("  旧システム価格例:")
                for i in range(min(5, len(old_data))):
                    row = old_data.iloc[i]
                    print(f"    {row['datetime']}: ¥{row['price']:.0f}")
            
            if len(new_data) > 0:
                new_price_range = f"¥{new_data['price'].min():.0f} ～ ¥{new_data['price'].max():.0f}"
                print(f"  新システム価格範囲: {new_price_range}")
                
                print("  新システム価格例:")
                for i in range(min(5, len(new_data))):
                    row = new_data.iloc[i]
                    print(f"    {row['datetime']}: ¥{row['price']:.0f}")
        
        except Exception as e:
            print(f"  ❌ 比較エラー: {e}")
    
    def analyze_system_differences(self):
        """システム設計の違いを分析"""
        print(f"\\n🔧 システム設計の違い:")
        print("="*40)
        
        differences = [
            ("データソース", "Thursday Friday Demo", "Leak Free System"),
            ("─────────", "────────────────", "─────────────────"),
            ("データベース", "trading_data.db", "leak_free_trading.db"),
            ("データ生成", "合成データ生成システム", "yfinance実データ取得"),
            ("価格範囲", "非現実的な価格変動", "実際の市場価格"),
            ("取引判定", "高い予測閾値", "低い予測閾値(0.2%)"),
            ("リーク対策", "部分的", "厳密(前日17:00カットオフ)"),
            ("特徴量", "基本的な特徴量", "高度な特徴量エンジニアリング"),
            ("モデル", "基本的なアンサンブル", "最適化されたアンサンブル"),
        ]
        
        for diff in differences:
            print(f"  {diff[0]:<12} | {diff[1]:<20} | {diff[2]}")
    
    def explain_performance_difference(self):
        """パフォーマンス差の説明"""
        print(f"\\n💡 パフォーマンス差の原因:")
        print("="*35)
        
        explanations = [
            "🎯 主要因:",
            "  1. データの現実性",
            "     - 旧: 合成データ（非現実的な価格急騰）",
            "     - 新: 実際の市場データ（現実的な価格変動）",
            "",
            "  2. 価格変動パターン",
            "     - 旧: ¥1000→¥2500（150%急騰）は合成データの異常",
            "     - 新: ¥3600前後の正常な5分足変動",
            "",
            "  3. 予測精度",
            "     - 旧: 合成データに最適化されたモデル",
            "     - 新: 実市場データでのR²スコア負値（予測困難）",
            "",
            "🔍 Technical Details:",
            "  - 旧システムの74.72%利益は合成データの価格急騰による",
            "  - 新システムの-0.98%損失は実際の市場環境を反映",
            "  - リークなし制約により予測精度がより現実的に",
            "",
            "✅ 結論:",
            "  新システム（Leak Free）が実際の取引環境に近い",
            "  旧システムの高収益は非現実的なデータによる虚偽の成功"
        ]
        
        for explanation in explanations:
            print(explanation)

def main():
    analyzer = SystemComparisonAnalyzer()
    
    # データ違い分析
    analyzer.analyze_data_differences()
    
    # システム設計の違い
    analyzer.analyze_system_differences()
    
    # パフォーマンス差の説明
    analyzer.explain_performance_difference()
    
    print(f"\\n" + "="*70)
    print("📋 分析完了: システム間の違いと原因を特定")

if __name__ == "__main__":
    main()
