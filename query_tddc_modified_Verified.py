import argparse
import os
from datetime import datetime, date, timedelta
from typing import List, Optional

import pandas as pd
import yfinance as yf

def parse_date(value: str) -> date:
    """Parses a date string in YYYY-MM-DD format."""
    return datetime.strptime(value, "%Y-%m-%d").date()

def to_date_obj(s: str) -> Optional[date]:
    """Converts a date string in YYYYMMDD format to a date object."""
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except (ValueError, TypeError):
        return None

def find_closest_past_date(target_date: date, available_dates: List[date]) -> Optional[date]:
    """Finds the latest date in the list that is on or before the target date."""
    past_dates = [d for d in available_dates if d <= target_date]
    if not past_dates:
        return None
    return max(past_dates)

def fetch_yahoo_finance_range(start: date, end: date, ticker: str) -> pd.DataFrame:
    """
    Fetches stock data from Yahoo Finance, automatically trying both .TW and .TWO suffixes.
    """
    end_adjusted = end + timedelta(days=1)
    tickers_to_try = [f"{ticker}.TW", f"{ticker}.TWO"]
    df_price = pd.DataFrame()

    for yahoo_ticker in tickers_to_try:
        try:
            print(f"Attempting to fetch daily data for {yahoo_ticker} from Yahoo Finance...")
            current_df = yf.download(yahoo_ticker, start=start, end=end_adjusted, progress=False, auto_adjust=False, interval="1d")

            if not current_df.empty and 'Close' in current_df.columns:
                print(f"Successfully fetched daily data for {yahoo_ticker}.")
                df_price = current_df
                break
            
            print(f"Daily data not found for {yahoo_ticker}. Falling back to fetch weekly data...")
            current_df_weekly = yf.download(yahoo_ticker, start=start, end=end_adjusted, progress=False, auto_adjust=False, interval="1wk")
            
            if not current_df_weekly.empty and 'Close' in current_df_weekly.columns:
                print(f"Successfully fetched weekly data for {yahoo_ticker}.")
                df_price = current_df_weekly
                break
        except Exception as e:
            print(f"An error occurred while fetching {yahoo_ticker}: {e}")
            continue
    
    if df_price.empty:
        return pd.DataFrame()

    if isinstance(df_price.columns, pd.MultiIndex):
        df_price.columns = df_price.columns.get_level_values(0)
        
    df_price.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'vol', 'Adj Close': 'adj_close'
    }, inplace=True, errors='ignore')
    
    df_price.index.name = 'gdate'
    return df_price

def get_ticker_data_from_tdcc_files(file_paths: List[str], ticker: str) -> pd.DataFrame:
    records = []
    for csv_path in file_paths:
        try:
            df = pd.read_csv(csv_path)
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
    parser = argparse.ArgumentParser(description="Query TDCC distribution for a date range, merging with K-line/Volume from Yahoo Finance.")
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

    user_start_d = parse_date(args.start)
    user_end_d = parse_date(args.end)
    
    ticker_tdcc_path = os.path.join(raw_root, args.ticker)
    if not os.path.exists(ticker_tdcc_path):
        print(f"Error: No data found for ticker {args.ticker} in {raw_root}.")
        return

    available_files = [os.path.join(ticker_tdcc_path, f) for f in os.listdir(ticker_tdcc_path) if f.endswith('.csv')]
    available_dates_str = sorted([os.path.basename(f).replace('.csv', '') for f in available_files])

    if not available_dates_str:
        print(f"No TDCC CSV files found for ticker {args.ticker}.")
        return

    all_possible_dates = sorted([d for d in (to_date_obj(s) for s in available_dates_str) if d is not None])
    actual_start_date = find_closest_past_date(user_start_d, all_possible_dates)
    actual_end_date = find_closest_past_date(user_end_d, all_possible_dates)

    if not actual_start_date or not actual_end_date or actual_start_date > actual_end_date:
        print(f"Error: Could not determine a valid data range based on your inputs.")
        return

    print(f"User provided range: {user_start_d} to {user_end_d}")
    print(f"Found actual data range: {actual_start_date} to {actual_end_date}")

    selected_dates_obj = [d for d in all_possible_dates if actual_start_date <= d <= actual_end_date]
    selected_dates_str = [d.strftime('%Y%m%d') for d in selected_dates_obj]

    print(f"Selected {len(selected_dates_str)} TDCC dates for processing.")
    selected_files = [os.path.join(ticker_tdcc_path, f"{d}.csv") for d in selected_dates_str]
    full_tdcc_data = get_ticker_data_from_tdcc_files(selected_files, args.ticker)

    if full_tdcc_data.empty:
        print("Could not load any TDCC data for the specified range.")
        return

    def find_col(df_cols, keyword):
        for col in df_cols:
            normalized_col = col.replace('\u3000', '').replace(' ', '')
            if keyword in normalized_col: return col
        raise KeyError(f"Keyword '{keyword}' not found in any column. Available columns: {list(df_cols)}")

    try:
        actual_cols = full_tdcc_data.columns
        col_level = find_col(actual_cols, "持股/單位數分級")
        col_people = find_col(actual_cols, "人數")
        col_shares = find_col(actual_cols, "股數")
        col_ratio = find_col(actual_cols, "占集保庫存數比例")
    except KeyError as e:
        print(f"Error: Could not automatically identify required columns. {e}")
        return

    people_tbl = full_tdcc_data.pivot_table(index=col_level, columns="週期日期", values=col_people, aggfunc="first")
    shares_tbl = full_tdcc_data.pivot_table(index=col_level, columns="週期日期", values=col_shares, aggfunc="first")
    ratio_tbl = full_tdcc_data.pivot_table(index=col_level, columns="週期日期", values=col_ratio, aggfunc="first")

    fetch_start_date = actual_start_date - timedelta(days=7)
    df_price = fetch_yahoo_finance_range(fetch_start_date, actual_end_date, args.ticker)
    
    expected_cols = ['open', 'high', 'low', 'close', 'vol']
    if df_price.empty or not all(col in df_price.columns for col in expected_cols):
        print("Warning: No valid price data available from Yahoo Finance. The output will not contain price/volume info.")
    else:
        print("Valid price data found. Processing and merging...")
        
        is_daily_data = pd.Series(df_price.index).diff().min() <= timedelta(days=5)
        if is_daily_data:
            weekly_agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'vol': 'sum'}
            df_price_weekly = df_price.resample('W-FRI').agg(weekly_agg).dropna()
        else:
            df_price_weekly = df_price

        # --- MODIFIED: Prepare a dictionary for ALL OHLCV data ---
        price_data_rows = {
            '周開盤價': {}, '周最高價': {}, '周最低價': {}, '周收盤價': {}, '周成交量': {}
        }

        for str_date in people_tbl.columns:
            tdcc_date = to_date_obj(str_date)
            if not tdcc_date: continue
            
            tdcc_timestamp = pd.Timestamp(tdcc_date)
            past_price_dates = df_price_weekly.index[df_price_weekly.index <= tdcc_timestamp]

            if not past_price_dates.empty:
                closest_price_date = past_price_dates.max()
                week_data = df_price_weekly.loc[closest_price_date]
                price_data_rows['周開盤價'][str_date] = week_data['open']
                price_data_rows['周最高價'][str_date] = week_data['high']
                price_data_rows['周最低價'][str_date] = week_data['low']
                price_data_rows['周收盤價'][str_date] = week_data['close']
                price_data_rows['周成交量'][str_date] = week_data['vol']
            else:
                for key in price_data_rows:
                    price_data_rows[key][str_date] = None
        
        # --- MODIFIED: Create and append all new rows ---
        rows_to_append = [pd.DataFrame([data]).rename(index={0: name}) for name, data in price_data_rows.items()]
        
        if rows_to_append:
            people_tbl = pd.concat([people_tbl] + rows_to_append)
            shares_tbl = pd.concat([shares_tbl] + rows_to_append)
            ratio_tbl = pd.concat([ratio_tbl] + rows_to_append)

    start_str = actual_start_date.strftime('%Y-%m-%d')
    end_str = actual_end_date.strftime('%Y-%m-%d')
    out_xlsx = os.path.join(out_query_dir, f"{args.ticker}_{start_str}_to_{end_str}.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        people_tbl.to_excel(writer, sheet_name="People")
        shares_tbl.to_excel(writer, sheet_name="Shares")
        ratio_tbl.to_excel(writer, sheet_name="RatioPct")

    print(f"\nProcessing complete. Saved results to {out_xlsx}")


if __name__ == "__main__":
    main()