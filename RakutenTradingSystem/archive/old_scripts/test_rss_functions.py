"""
MarketSpeedⅡ RSS関数テストスクリプト
正しいMSGETDATA関数の仕様を確認
"""

import win32com.client
import time
import logging
from datetime import datetime

# ログ設定
logging.basicConfig(level=logging.INFO)

def test_rss_functions():
    """RSS関数の正しい仕様をテスト"""
    
    try:
        # Excel起動
        excel_app = win32com.client.Dispatch("Excel.Application")
        excel_app.Visible = True  # 表示して確認
        excel_app.DisplayAlerts = False
        
        workbook = excel_app.Workbooks.Add()
        sheet = workbook.Worksheets(1)
        
        print("Excel起動完了")
        
        # ヘッダー設定
        headers = [
            "銘柄コード", "テスト項目", "関数形式", "結果", "エラー情報"
        ]
        
        for i, header in enumerate(headers, 1):
            sheet.Cells(1, i).Value = header
            sheet.Cells(1, i).Font.Bold = True
        
        # テスト銘柄
        test_symbols = ["1301", "7203", "6758"]
        
        # テスト関数のパターン
        test_functions = [
            # 数値形式のテスト
            ("現在値_数値", '=MSGETDATA("{symbol}",1)'),
            ("始値_数値", '=MSGETDATA("{symbol}",2)'),
            ("高値_数値", '=MSGETDATA("{symbol}",3)'),
            ("安値_数値", '=MSGETDATA("{symbol}",4)'),
            ("出来高_数値", '=MSGETDATA("{symbol}",5)'),
            
            # 文字列形式のテスト
            ("現在値_文字列", '=MSGETDATA("{symbol}","現在値")'),
            ("始値_文字列", '=MSGETDATA("{symbol}","始値")'),
            ("高値_文字列", '=MSGETDATA("{symbol}","高値")'),
            ("安値_文字列", '=MSGETDATA("{symbol}","安値")'),
            ("出来高_文字列", '=MSGETDATA("{symbol}","出来高")'),
            
            # 別の形式のテスト
            ("現在値_別形式1", '=MSGETDATA("{symbol}","CP")'),
            ("現在値_別形式2", '=MSGETDATA("{symbol}","PRICE")'),
            ("出来高_別形式", '=MSGETDATA("{symbol}","VOLUME")'),
            
            # 板情報のテスト
            ("買気配1_数値", '=MSGETDATA("{symbol}",51)'),
            ("買気配1_文字列", '=MSGETDATA("{symbol}","買気配1")'),
            ("買気配1_別形式", '=MSGETDATA("{symbol}","BID1")'),
        ]
        
        row = 2
        
        for symbol in test_symbols:
            for test_name, formula_template in test_functions:
                sheet.Cells(row, 1).Value = symbol
                sheet.Cells(row, 2).Value = test_name
                
                # 関数設定
                formula = formula_template.format(symbol=symbol)
                sheet.Cells(row, 3).Value = formula
                
                try:
                    sheet.Cells(row, 4).Formula = formula
                    print(f"関数設定完了: {symbol} - {test_name}")
                except Exception as e:
                    sheet.Cells(row, 5).Value = f"設定エラー: {e}"
                    print(f"関数設定エラー: {symbol} - {test_name} - {e}")
                
                row += 1
                
                # 処理速度調整
                time.sleep(0.1)
        
        # 計算実行
        print("Excel計算実行中...")
        excel_app.Calculate()
        time.sleep(10)
        
        # 結果確認
        print("\n=== 結果確認 ===")
        for i in range(2, row):
            symbol = sheet.Cells(i, 1).Value
            test_name = sheet.Cells(i, 2).Value
            formula = sheet.Cells(i, 3).Value
            result = sheet.Cells(i, 4).Value
            
            print(f"{symbol} - {test_name}: {result}")
            
            # エラー状態の確認
            if result and str(result).startswith('#'):
                sheet.Cells(i, 5).Value = f"Excelエラー: {result}"
                print(f"  -> エラー: {result}")
            elif isinstance(result, (int, float)) and result != 0:
                sheet.Cells(i, 5).Value = "正常"
                print(f"  -> 正常取得: {result}")
            else:
                sheet.Cells(i, 5).Value = "データなし"
                print(f"  -> データなし: {result}")
        
        # ファイル保存
        try:
            workbook.SaveAs(r"C:\Users\akane\RakutenTradingSystem\RSS_Function_Test.xlsx")
            print("\nテスト結果保存完了: RSS_Function_Test.xlsx")
        except Exception as e:
            print(f"ファイル保存エラー: {e}")
        
        # 手動確認用待機
        print("\n=== 手動確認用待機 ===")
        print("Excelファイルでテスト結果を確認してください")
        print("正常に動作する関数形式を特定してください")
        input("確認完了後、Enterキーを押してください...")
        
        # クリーンアップ
        workbook.Close(False)
        excel_app.Quit()
        
        return True
        
    except Exception as e:
        print(f"テストエラー: {e}")
        return False

if __name__ == "__main__":
    print("=== MarketSpeedⅡ RSS関数テスト ===")
    print("複数の関数形式をテストして正しい仕様を確認します")
    print("MarketSpeedⅡにログインした状態で実行してください")
    
    success = test_rss_functions()
    
    if success:
        print("テスト完了。RSS_Function_Test.xlsxで結果を確認してください。")
    else:
        print("テスト失敗。MarketSpeedⅡの状態を確認してください。")
