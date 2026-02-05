"""
MarketSpeedⅡの状態確認スクリプト
"""

import win32com.client
import time
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)

def check_marketspeed_status():
    """MarketSpeedⅡの状態確認"""
    
    try:
        # Excel接続
        excel_app = win32com.client.Dispatch("Excel.Application")
        excel_app.Visible = True  # 表示して確認
        excel_app.DisplayAlerts = False
        
        workbook = excel_app.Workbooks.Add()
        sheet = workbook.Worksheets(1)
        
        print("Excel接続成功")
        
        # ヘッダー設定
        sheet.Cells(1, 1).Value = "テスト項目"
        sheet.Cells(1, 2).Value = "結果"
        sheet.Cells(1, 3).Value = "詳細"
        
        # 基本的な関数テスト
        test_cases = [
            ("基本関数", '=NOW()', "現在時刻"),
            ("MSGETDATA関数-日経平均", '=MSGETDATA("NK225","現在値")', "日経平均現在値"),
            ("MSGETDATA関数-銘柄1", '=MSGETDATA("7203","現在値")', "トヨタ自動車"),
            ("MSGETDATA関数-銘柄2", '=MSGETDATA("6758","現在値")', "ソニー"),
            ("MSGETDATA関数-銘柄3", '=MSGETDATA("9984","現在値")', "SBG"),
            ("MSGETDATA関数-出来高", '=MSGETDATA("7203","出来高")', "トヨタ出来高"),
            ("MSGETDATA関数-VWAP", '=MSGETDATA("7203","VWAP")', "トヨタVWAP"),
        ]
        
        for i, (test_name, formula, description) in enumerate(test_cases, 2):
            sheet.Cells(i, 1).Value = test_name
            sheet.Cells(i, 3).Value = description
            
            try:
                sheet.Cells(i, 2).Formula = formula
                print(f"数式設定: {test_name} = {formula}")
            except Exception as e:
                sheet.Cells(i, 2).Value = f"エラー: {e}"
                print(f"数式設定エラー: {test_name} - {e}")
        
        # 計算実行
        print("Excel計算実行中...")
        excel_app.Calculate()
        time.sleep(15)  # 計算完了とデータ取得待機
        
        # 結果確認
        print("\n=== 結果確認 ===")
        for i, (test_name, formula, description) in enumerate(test_cases, 2):
            try:
                result = sheet.Cells(i, 2).Value
                print(f"{test_name}: {result} ({description})")
                
                # エラーチェック
                if result and str(result).startswith('#'):
                    print(f"  -> Excelエラー: {result}")
                elif result == -2146826259:
                    print(f"  -> COMエラー: MarketSpeedⅡ未接続またはログイン失敗")
                elif isinstance(result, (int, float)) and result > 0:
                    print(f"  -> 正常取得")
                else:
                    print(f"  -> 異常値: {result}")
                    
            except Exception as e:
                print(f"結果取得エラー {test_name}: {e}")
        
        # 保存
        workbook.SaveAs(r"C:\Users\akane\RakutenTradingSystem\MarketSpeedStatusCheck.xlsx")
        print("\n状態確認ファイル保存: MarketSpeedStatusCheck.xlsx")
        
        # 10分待機（手動確認用）
        print("10分間待機 - MarketSpeedⅡの状態を確認してください")
        print("- MarketSpeedⅡがログインしているか確認")
        print("- データ配信が有効になっているか確認")
        print("- Excelファイルで結果を確認")
        
        time.sleep(600)  # 10分待機
        
        # クリーンアップ
        workbook.Close(False)
        excel_app.Quit()
        
    except Exception as e:
        print(f"エラー: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=== MarketSpeedⅡ状態確認 ===")
    check_marketspeed_status()
    print("状態確認完了")
