import argparse
import os
from datetime import datetime, date, timedelta
from typing import List, Tuple, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
import urllib3

# Disable SSL warnings for wearn.com
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


WEARN_URL = "https://stock.wearn.com/cdata.asp"


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def find_available_dates(raw_root: str) -> List[str]:
    """Find all available dates from the directory structure of fetch_tdcc.py"""
    available_dates = []
    if not os.path.exists(raw_root):
        return []
    for stock_dir in os.listdir(raw_root):
        stock_path = os.path.join(raw_root, stock_dir)
        if os.path.isdir(stock_path):
            for file_name in os.listdir(stock_path):
                if file_name.endswith('.csv'):
                    date_str = file_name.replace('.csv', '')
                    if len(date_str) == 8 and date_str.isdigit():
                        available_dates.append(date_str)
    return sorted(list(set(available_dates)))


def find_start_date(dates: List[str], start: date) -> Tuple[str, bool]:
    """Find the start date from available dates"""
    dt_dates = []
    for d in dates:
        try:
            dt = datetime.strptime(d, "%Y%m%d").date()
            dt_dates.append((d, dt))
        except Exception:
            continue
    
    dt_dates.sort(key=lambda x: x[1])
    for d_str, dt in dt_dates:
        if dt >= start:
            return d_str, True
    
    prev = None
    for d_str, dt in dt_dates:
        if dt <= start:
            prev = d_str
        else:
            break
    if prev:
        return prev, False
    
    return dates[0] if dates else "", False


def fetch_wearn_month(year: int, month: int, kind: str) -> pd.DataFrame:
    params = {"Year": f"{year:03d}", "month": f"{month:02d}", "kind": kind}
    r = requests.get(WEARN_URL, params=params, timeout=30, verify=False)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find("table")
    if not table:
        return pd.DataFrame()
    rows = []
    for tr in table.find_all("tr"):
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cols) >= 7 and cols[0]:
            rows.append(cols[:7])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "chg", "vol"])
    
    def parse_roc(s: str) -> Optional[date]:
        try:
            parts = s.split("/")
            roc_y = int(parts[0])
            y = roc_y + 1911
            return date(y, int(parts[1]), int(parts[2]))
        except Exception:
            return None
            
    df["gdate"] = df["date"].apply(parse_roc)
    df = df.dropna(subset=["gdate"])
    for col in ["open", "high", "low", "close", "vol"]:
        df[col] = pd.to_numeric(df[col].str.replace(",", ""), errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close", "vol"])


def fetch_wearn_range(start: date, end: date, kind: str) -> pd.DataFrame:
    dfs = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        roc_year = cur.year - 1911
        month_df = fetch_wearn_month(roc_year, cur.month, kind)
        if not month_df.empty:
            dfs.append(month_df)
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    if "gdate" not in df.columns:
        return pd.DataFrame()
    df["gdate"] = pd.to_datetime(df["gdate"])
    df = df[(df["gdate"] >= pd.Timestamp(start)) & (df["gdate"] <= pd.Timestamp(end))]
    return df.sort_values("gdate")


def get_ticker_data_from_tdcc_files(file_paths: List[str], ticker: str) -> pd.DataFrame:
    records = []
    for csv_path in file_paths:
        try:
            df = pd.read_csv(csv_path)
            # The file name is the date
            date_str = os.path.basename(csv_path).replace('.csv', '')
            df["週期日期"] = date_str
            records.append(df)
        except Exception as e:
            print(f"Warning: Could not read {csv_path}: {e}")
            continue
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)

def main():
    parser = argparse.ArgumentParser(description="Query TDCC distribution, merge with K-line/Volume, and save to Excel.")
    parser.add_argument("--base", required=True, help="Base directory where data is stored.")
    parser.add_argument("--ticker", required=True, help="Stock ticker to process.")
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD format.")
    args = parser.parse_args()

    base_dir = args.base
    data_root = os.path.join(base_dir, "data")
    raw_root = os.path.join(data_root, "tdcc_raw")
    out_query_dir = os.path.join(data_root, "query_excel")
    os.makedirs(out_query_dir, exist_ok=True)

    start_d = parse_date(args.start)
    end_d = parse_date(args.end)
    
    ticker_tdcc_path = os.path.join(raw_root, args.ticker)
    if not os.path.exists(ticker_tdcc_path):
        print(f"Error: No data found for ticker {args.ticker} in {raw_root}.")
        print("Please run fetch_tdcc.py first.")
        return

    available_files = [os.path.join(ticker_tdcc_path, f) for f in os.listdir(ticker_tdcc_path) if f.endswith('.csv')]
    available_dates = sorted([os.path.basename(f).replace('.csv', '') for f in available_files])

    if not available_dates:
        print(f"No TDCC CSV files found for ticker {args.ticker}.")
        return
    
    print(f"Found {len(available_dates)} available dates for ticker {args.ticker}")
    
    start_date_str, is_ge = find_start_date(available_dates, start_d)
    if not is_ge:
        print("Warning: No data >= start date; using nearest <= week.")

    def to_date(s: str) -> Optional[date]:
        try: return datetime.strptime(s, "%Y%m%d").date()
        except: return None

    selected_dates = [d for d in available_dates if start_d <= (to_date(d) or date(1900,1,1)) <= end_d]
    if not selected_dates:
        sdt = to_date(start_date_str)
        if sdt:
            selected_dates = [start_date_str]

    print(f"Selected {len(selected_dates)} TDCC dates for processing.")
    selected_files = [os.path.join(ticker_tdcc_path, f"{d}.csv") for d in selected_dates]

    full_tdcc_data = get_ticker_data_from_tdcc_files(selected_files, args.ticker)
    if full_tdcc_data.empty:
        print("Could not load any TDCC data for the specified range.")
        return

    print(f"Combined TDCC data shape: {full_tdcc_data.shape}")

    col_level = "持股/單位數分級"
    col_people = "人\u3000\u3000\u3000數"
    col_shares = "股\u3000\u3000\u3000數/單位數"
    col_ratio = "占集保庫存數比例 (%)"

    # --- 1. Create Pivot Tables (transposed) ---
    people_tbl = full_tdcc_data.pivot_table(index=col_level, columns="週期日期", values=col_people, aggfunc="first")
    shares_tbl = full_tdcc_data.pivot_table(index=col_level, columns="週期日期", values=col_shares, aggfunc="first")
    ratio_tbl = full_tdcc_data.pivot_table(index=col_level, columns="週期日期", values=col_ratio, aggfunc="first")

    # --- 2. Fetch and Prepare K-line/Volume Data ---
    print("Fetching price data from wearn.com...")
    df_price_daily = fetch_wearn_range(start_d, end_d, args.ticker)
    
    if df_price_daily.empty:
        print("Warning: No price data available from wearn.com for this range.")
    else:
        print(f"Daily price data shape: {df_price_daily.shape}")
        df_price_daily.set_index('gdate', inplace=True)
        
        # Resample to weekly, anchored on Friday
        weekly_agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'vol': 'sum'
        }
        df_price_weekly = df_price_daily.resample('W-FRI').agg(weekly_agg).dropna()
        print(f"Resampled weekly price data shape: {df_price_weekly.shape}")

        # --- 3. Align Dates and Create New Rows ---
        close_prices = {}
        volumes = {}

        for str_date in people_tbl.columns:
            tdcc_date = to_date(str_date)
            if not tdcc_date: continue
            
            # Align TDCC date (can be Sat/Sun) to its preceding Friday
            aligned_friday = tdcc_date - timedelta(days=(tdcc_date.weekday() - 4) % 7)
            
            if aligned_friday in df_price_weekly.index:
                close_prices[str_date] = df_price_weekly.loc[aligned_friday, 'close']
                volumes[str_date] = df_price_weekly.loc[aligned_friday, 'vol']
            else:
                close_prices[str_date] = None
                volumes[str_date] = None

        # Create new rows as DataFrames to append
        close_row = pd.DataFrame([close_prices]).rename(index={0: '周收盤價'})
        volume_row = pd.DataFrame([volumes]).rename(index={0: '周成交量'})

        # --- 4. Append New Rows to Each Table ---
        people_tbl = pd.concat([people_tbl, close_row, volume_row])
        shares_tbl = pd.concat([shares_tbl, close_row, volume_row])
        ratio_tbl = pd.concat([ratio_tbl, close_row, volume_row])

    # --- 5. Save to Excel ---
    out_xlsx = os.path.join(out_query_dir, f"{args.ticker}_{args.start}_{args.end}.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        people_tbl.to_excel(writer, sheet_name="People")
        shares_tbl.to_excel(writer, sheet_name="Shares")
        ratio_tbl.to_excel(writer, sheet_name="RatioPct")

    print(f"\nProcessing complete. Saved results to {out_xlsx}")


if __name__ == "__main__":
    main()
