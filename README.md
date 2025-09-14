# TDCC Tools - 台股集保戶股權分散分析工具

這是一個用於抓取和分析台灣集中保管結算所(TDCC)股權分散資料的工具包，並結合週K線/成交量數據進行分析。

## 功能特色

### Program 1: `fetch_tdcc_modified.py` - TDCC資料抓取器
- **本地列表驅動**：從本地的 `stock_list.csv` 讀取股票列表，穩定可靠。
- **首次執行**：可設定抓取過去N週的資料。
- **後續執行**：增量更新，自動跳過已下載的日期，只抓取遺漏的資料。
- **並行下載**：支援多執行緒加速下載。
- **錯誤處理**：自動處理網站的SSL憑證問題。

### Program 2: `query_tddc_modified.py` - 區間查詢與資料整合
- **參數化查詢**：可指定單一股票代號、起始日期、結束日期。
- **資料整合**：整合多個日期的股權分散資料，並從`wearn.com`抓取對應區間的日K線與成交量。
- **日期對齊**：自動將股權分散的日期與K線的週五日期對齊。
- **Excel輸出**：產生包含人數、股數、佔比，以及對齊後的「周收盤價」與「周成交量」的 Excel 報表。

### Program 3: `plottingTrends_tddc_modified.py` - 趨勢彙總與製圖
- **三種分組方案**：
  - **股數定義 (`shares`)**：散戶(1-40萬)、中實戶(40-100萬)、大戶(>100萬)。
  - **金額定義 (`amount`)**：<500萬、500-1000萬、1000-3000萬、>3000萬。
  - **自訂級距 (`custom`)**：可自由定義持有股數區間。
- **智慧繪圖**：
  - **主圖**：繪製所有彙總後類別的趨勢線。
  - **細節圖**：自動偵測變化平緩的線條，並產生一張專門放大其微小變化的 `_screenIn` 細節圖。
- **雙重輸出**：同時產生彙總後的 Excel 數據表和 PNG 趨勢圖。

## 安裝需求

### 系統需求
- Python 3.10+
- Windows 10/11 (已在Windows環境測試)

### 安裝依賴
```bash
cd "F:\Investment and Finance\tdcc_tools"
python -m pip install -r requirements.txt
```

## 使用方式

### 0. (僅需一次) 產生股票列表 (可選)
如果您需要重新產生或客製化股票列表，請執行：
```bash
python "C:\Users\Wayne\Documents\create_stock_list.py"
```

### 1. 首次資料抓取
```bash
# 抓取`stock_list.csv`中所有股票最近一年的資料
python "C:\Users\Wayne\Documents\fetch_tdcc_modified.py" --base "F:\Investment and Finance\tdcc_tools"

# 抓取最近10週的資料進行測試
python "C:\Users\Wayne\Documents\fetch_tdcc_modified.py" --base "F:\Investment and Finance\tdcc_tools" --first-run-weeks 10

# 當前目錄
python fetch_tdcc_modified.py
```



**參數說明：**
- `--base`：(必要) 專案根目錄路徑。
- `--stock-list`：(可選) 股票列表CSV檔案路徑。預設為 `C:\Users\Wayne\Documents\stock_list.csv`。
- `--max-workers`：(可選) 並行下載數量（預設8）。
- `--first-run-weeks`：(可選) **首次執行**時要抓取的週數（預設52）。

### 2. 區間查詢與資料整合
```bash
# 查詢台積電(2330)在指定期間的股權分散，並整合K線資料
python "C:\Users\Wayne\Documents\query_tddc_modified.py" --base "F:\Investment and Finance\tdcc_tools" --ticker 2330 --start 2023-01-01 --end 2023-12-31

# 當前目錄
python query_tddc_modified.py --ticker 2330 --start 2023-01-01 --end 2023-12-31

```

**參數說明：**
- `--base`：(必要) 專案根目錄路徑。
- `--ticker`：(必要) 股票代號。
- `--start`：(必要) 起始日期 (YYYY-MM-DD)。
- `--end`：(必要) 結束日期 (YYYY-MM-DD)。

### 3. 趨勢彙總與製圖
```bash
# 對上一步產生的2330 Excel檔，使用股數分組方案繪圖
python "C:\Users\Wayne\Documents\plottingTrends_tddc_modified.py" --base "F:\Investment and Finance\tdcc_tools" --input "F:\Investment and Finance\tdcc_tools\data\query_excel\2330_2023-01-01_2023-12-31.xlsx" --scheme shares

# 使用金額分組方案（需提供股價）
python "C:\Users\Wayne\Documents\plottingTrends_tddc_modified.py" --base "F:\Investment and Finance\tdcc_tools" --input "F:\Investment and Finance\tdcc_tools\data\query_excel\2330_2023-01-01_2023-12-31.xlsx" --scheme amount --price 600


# 當前目錄
python plottingTrends_tddc_modified.py --input "F:\Investment and Finance\tdcc_tools\data\query_excel\2330_2023-01-01_2023-12-31.xlsx" 


```
**參數說明：**
- `--base`：(必要) 專案根目錄路徑。
- `--input`：(必要) 第二步產生的Excel檔案路徑。
- `--scheme`：(可選) 分組方案 (shares/amount/custom)，預設為 `shares`。
- `--price`：(金額分組時必要) 股價。
- `--custom-bins`：(自訂分組時必要) 自訂級距，例如 `0-1000,1001-10000,>10000`。

## 輸出檔案結構

```
F:\Investment and Finance\tdcc_tools\
├── data\
│   ├── tdcc_raw\{ticker}\{date}.csv       # 原始TDCC資料
│   ├── tdcc_index.csv                     # 下載索引
│   ├── query_excel\{ticker}_{start}_{end}.xlsx # 程式2輸出：整合K線的報表
│   └── trends_aggregated\{file}_{scheme}.xlsx # 程式3輸出：彙總後的數據
├── output\
│   └── trends\{file}_{scheme}_*.png       # 程式3輸出：最終趨勢圖
├── C:\Users\Wayne\Documents\
│   ├── fetch_tdcc_modified.py             # 程式 1
│   ├── query_tddc_modified.py             # 程式 2
│   ├── plottingTrends_tddc_modified.py    # 程式 3
│   ├── create_stock_list.py               # (工具) 產生股票列表
│   └── stock_list.csv                     # 股票列表來源
├── requirements.txt                       # 依賴套件
└── README.md                              # 說明文件

```

## 資料來源

- **TDCC股權分散**：`https://www.tdcc.com.tw/portal/zh/smWeb/qryStock`
- **週K線/成交量**：`https://tw.stock.yahoo.com/class/`
- **股票列表**：台灣證券交易所 & 櫃檯買賣中心

## 注意事項

1.  **網路連線**：執行程式時需要穩定的網路連線。
2.  **執行時間**：首次抓取大量資料時可能需要較長時間。
3.  **資料更新**：TDCC資料通常每週五晚上或週六更新。

## 故障排除

1.  **SSL錯誤**：程式已內建處理，如仍有問題請檢查您的防火牆或網路設定。
2.  **下載失敗**：可嘗試降低 `--max-workers` 數量，或檢查網路連線。
3.  **找不到檔案**：請確認 `--base` 和 `--input` 等路徑參數是否正確。

## 版本資訊

- **版本**：1.0.0
- **Python**：3.10+
- **最後更新**：2024年8月
- **支援平台**：Windows 10/11
