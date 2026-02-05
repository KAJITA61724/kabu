"""
Excelデータ収集のデバッグ用スクリプト
MarketSpeedⅡのMSGETDATA関数の動作確認
"""

import win32com.client
import pandas as pd
import time
from datetime import datetime
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)

def debug_excel_connection():
    """Excel接続とMSGETDATA関数のデバッグ"""
    
    try:
        # Excel接続
        excel_app = win32com.client.Dispatch("Excel.Application")
        excel_app.Visible = True  # デバッグ用に表示
        excel_app.DisplayAlerts = False
        
        workbook = excel_app.Workbooks.Add()
        sheet = workbook.Worksheets(1)
        sheet.Name = "MarketSpeedDebug"
        
        print("Excel接続成功")
        
        # ヘッダー設定
        headers = ["銘柄", "現在値", "始値", "高値", "安値", "出来高", "エラー情報"]
        for i, header in enumerate(headers, 1):
            sheet.Cells(1, i).Value = header
            sheet.Cells(1, i).Font.Bold = True
        
        # テスト銘柄（プライム銘柄から数銘柄）
        test_symbols = ["7203", "6758", "9984", "8306", "4503"]
        
        print(f"テスト銘柄: {test_symbols}")
        
        # データ取得テスト
        for i, symbol in enumerate(test_symbols, 2):
            print(f"データ取得テスト: {symbol}")
            
            # 銘柄コード
            sheet.Cells(i, 1).Value = symbol
            
            # MSGETDATA関数の設定
            try:
                sheet.Cells(i, 2).Formula = f'=MSGETDATA("{symbol}","現在値")'
                sheet.Cells(i, 3).Formula = f'=MSGETDATA("{symbol}","始値")'
                sheet.Cells(i, 4).Formula = f'=MSGETDATA("{symbol}","高値")'
                sheet.Cells(i, 5).Formula = f'=MSGETDATA("{symbol}","安値")'
                sheet.Cells(i, 6).Formula = f'=MSGETDATA("{symbol}","出来高")'
                
                print(f"数式設定完了: {symbol}")
                
            except Exception as e:
                sheet.Cells(i, 7).Value = f"数式設定エラー: {e}"
                print(f"数式設定エラー {symbol}: {e}")
        
        # 計算実行
        print("Excel計算実行中...")
        excel_app.Calculate()
        time.sleep(10)  # 計算完了待機
        
        # 結果確認
        print("\n=== 結果確認 ===")
        for i, symbol in enumerate(test_symbols, 2):
            current_price = sheet.Cells(i, 2).Value
            open_price = sheet.Cells(i, 3).Value
            high_price = sheet.Cells(i, 4).Value
            low_price = sheet.Cells(i, 5).Value
            volume = sheet.Cells(i, 6).Value
            
            print(f"{symbol}: 現在値={current_price}, 始値={open_price}, 高値={high_price}, 安値={low_price}, 出来高={volume}")
            
            # エラーチェック
            if current_price and str(current_price).startswith("#"):
                sheet.Cells(i, 7).Value = f"エラー: {current_price}"
                print(f"  -> エラー発生: {current_price}")
            elif current_price and isinstance(current_price, (int, float)) and current_price > 0:
                sheet.Cells(i, 7).Value = "OK"
                print(f"  -> データ取得成功")
            else:
                sheet.Cells(i, 7).Value = "データなし"
                print(f"  -> データ取得失敗")
        
        # 保存
        try:
            workbook.SaveAs(r"C:\Users\akane\RakutenTradingSystem\MarketSpeedDebug.xlsx")
            print("デバッグファイル保存完了: MarketSpeedDebug.xlsx")
        except Exception as e:
            print(f"ファイル保存エラー: {e}")
        
        # 5分待機（手動確認用）
        print("5分間待機中 - Excelファイルを手動で確認してください")
        time.sleep(300)
        
        # クリーンアップ
        workbook.Close(False)
        excel_app.Quit()
        
    except Exception as e:
        print(f"Excel接続エラー: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=== MarketSpeedⅡ Excel接続デバッグ ===")
    print("現在時刻:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    success = debug_excel_connection()
    
    if success:
        print("デバッグ完了")
    else:
        print("デバッグ失敗")
