import argparse
import os
import re
from typing import List, Tuple, Dict

import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf


# Set a font that supports Chinese characters to prevent garbled text in charts
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Heiti TC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False # Fix for minus sign display issue

    # For mplfinance specific use
    s = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.family': 'Microsoft JhengHei'})
except Exception as e:
    print(f"Font setting warning: {e}")
    s = 'yahoo' # Fallback to default style if font is not found

def is_line_flat(series: pd.Series, threshold: float = 0.01) -> bool:
    """Determines if a line is 'flat' based on its standard deviation relative to its mean."""
    if series.dropna().empty: return False
    mean_val = series.mean()
    if pd.isna(mean_val): return False
    std_val = series.std()
    if pd.isna(std_val): return False
    if abs(mean_val) < 1e-9 and std_val < 1e-9: return True
    if abs(mean_val) > 1e-9 and (std_val / abs(mean_val)) < threshold: return True
    return False

def validate_data_for_plotting(df: pd.DataFrame, data_name: str) -> bool:
    """
    Validates if the DataFrame contains valid data for plotting.
    Returns True if data is valid, False otherwise.
    """
    if df.empty:
        print(f"Warning: {data_name} DataFrame is empty, skipping chart generation.")
        return False
    
    # Check if all columns contain only NaN or zero values
    valid_columns = []
    for col in df.columns:
        col_data = df[col].dropna()
        if not col_data.empty and not (col_data == 0).all():
            valid_columns.append(col)
    
    if not valid_columns:
        print(f"Warning: {data_name} contains no valid non-zero data, skipping chart generation.")
        return False
    
    # Check for infinite values
    if df.isin([float('inf'), float('-inf')]).any().any():
        print(f"Warning: {data_name} contains infinite values, cleaning data...")
        df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    
    return True

def parse_custom_bins(bins_str: str) -> List[Tuple[float, float]]:
    """Parses a custom bin string like '0-30,30-100,>1000' into a list of tuples."""
    bins: List[Tuple[float, float]] = []
    for part in bins_str.split(","):
        part = part.strip()
        if not part: continue
        if part.startswith(">"):
            lo = float(part[1:])
            bins.append((lo, float("inf")))
        else:
            lo, hi = part.split("-")
            bins.append((float(lo), float(hi)))
    return bins

def get_label_to_cols_map(scheme: str, price: float, custom_bins_str: str, all_cols: List[str]) -> Tuple[Dict[str, List[str]], str]:
    """Determines which original columns map to which new aggregated group label."""
    def parse_col_range(label: str) -> Tuple[float, float]:
        nums = [float(s.replace(",", "")) for s in re.findall(r"[\d,]+", label)]
        if len(nums) == 2: return nums[0], nums[1]
        if len(nums) == 1:
            if ">" in label or "以上" in label: return nums[0], float("inf")
            return nums[0], nums[0]
        return 0.0, 0.0

    label_to_cols: Dict[str, List[str]] = {}
    scheme_name = scheme

    if scheme == "shares":
        scheme_name = "General_Definition"
        bins = {"散戶 (1-400K)": (1, 400000), "中實戶 (400K-1M)": (400001, 1000000), "大戶 (>1M)": (1000001, float("inf"))}
        label_to_cols = {label: [] for label in bins}
        for col in all_cols:
            lo, hi = parse_col_range(col)
            if lo == 0: continue
            col_mid = (lo + hi) / 2 if hi != float("inf") else lo * 1.5
            for label, (bin_lo, bin_hi) in bins.items():
                if bin_lo <= col_mid < bin_hi:
                    label_to_cols[label].append(col)
                    break
    elif scheme == "amount":
        scheme_name = "Amount_Definition"
        if price <= 0: raise ValueError("--price is required for amount scheme")
        bins = {"< 5M": (0, 5_000_000), "5M-10M": (5_000_000, 10_000_000), "10M-30M": (10_000_000, 30_000_000), "> 30M": (30_000_000, float("inf"))}
        label_to_cols = {label: [] for label in bins}
        for col in all_cols:
            lo, hi = parse_col_range(col)
            mid_shares = (lo + hi) / 2 if hi != float("inf") else lo * 1.5
            amount = mid_shares * price
            for label, (bin_lo, bin_hi) in bins.items():
                if bin_lo <= amount < bin_hi:
                    label_to_cols[label].append(col)
                    break
    elif scheme == "custom":
        scheme_name = "Custom_Definition"
        custom_bins = parse_custom_bins(custom_bins_str)
        if not custom_bins: raise ValueError("--custom-bins required for custom scheme")
        labels = [f"{int(lo)}-{int(hi) if hi != float('inf') else 'inf'}" for lo, hi in custom_bins]
        label_to_cols = {label: [] for label in labels}
        for col in all_cols:
            lo, hi = parse_col_range(col)
            mid_shares = (lo + hi) / 2 if hi != float("inf") else lo * 1.5
            for (bin_lo, bin_hi), label in zip(custom_bins, labels):
                if bin_lo <= mid_shares < bin_hi:
                    label_to_cols[label].append(col)
                    break
    return label_to_cols, scheme_name

def aggregate_data(df: pd.DataFrame, label_to_cols: Dict[str, List[str]]) -> pd.DataFrame:
    """Aggregates a DataFrame based on the provided column mapping."""
    frames = []
    for label, cols in label_to_cols.items():
        if not cols or not all(c in df.columns for c in cols): continue
        sub = df[cols].sum(axis=1)
        frames.append(sub.rename(label))
    if not frames: return pd.DataFrame(index=df.index)
    return pd.concat(frames, axis=1)

def plot_and_save(df: pd.DataFrame, weekly_ohlc: pd.DataFrame, title: str, path: str, ylabel: str):
    """
    Plots the trend data with a candlestick chart background and volume overlay.
    """
    if df.empty or weekly_ohlc.empty:
        print(f"Skipping chart generation for '{title}' due to missing data.")
        return
    
    # Validate data before plotting
    if not validate_data_for_plotting(df, title):
        return
    
    # Clean the data - remove any columns that are all NaN or all zeros
    clean_df = df.copy()
    columns_to_drop = []
    
    for col in clean_df.columns:
        col_data = clean_df[col].dropna()
        if col_data.empty or (col_data == 0).all() or col_data.isna().all():
            columns_to_drop.append(col)
    
    if columns_to_drop:
        print(f"Dropping empty/zero columns for {title}: {columns_to_drop}")
        clean_df = clean_df.drop(columns=columns_to_drop)
    
    if clean_df.empty:
        print(f"No valid data remaining for {title} after cleaning, skipping chart generation.")
        return

    try:
        ohlc_data = weekly_ohlc[['open', 'high', 'low', 'close', 'vol']].copy()
        ohlc_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Create addplot lines only for columns with valid data
        ap_lines = []
        for col in clean_df.columns:
            col_data = clean_df[col].dropna()
            if not col_data.empty and not (col_data == 0).all():
                ap_lines.append(mpf.make_addplot(clean_df[col], panel=0, ylabel=ylabel))
        
        if not ap_lines:
            print(f"No valid addplot lines for {title}, skipping chart generation.")
            return
        
        fig, axes = mpf.plot(ohlc_data, 
                             type='candle', 
                             style=s, # Use the new style with the Chinese font
                             title=f'\n{title}',
                             volume=True, 
                             addplot=ap_lines,
                             figsize=(20, 10),
                             panel_ratios=(3, 1),
                             returnfig=True)

        axes[0].legend([col.replace('_', ' ') for col in clean_df.columns if col not in columns_to_drop], fontsize='small')
        
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved main chart to: {path}")

        flat_cols = [col for col in clean_df.columns if is_line_flat(clean_df[col])]
        if flat_cols:
            print(f"Detected {len(flat_cols)} near-flat lines. Creating detail chart...")
            # Temporarily set font parameters for this specific chart
            with plt.rc_context({'font.sans-serif': ['Microsoft JhengHei', 'Heiti TC', 'sans-serif'], 
                                'axes.unicode_minus': False}):
                fig_detail, ax_detail = plt.subplots(figsize=(15, 8))
                for col in flat_cols:
                    ax_detail.plot(clean_df.index, clean_df[col], label=str(col), marker='.', markersize=4, linestyle='-')
                
                ax_detail.set_title(f"{title} (Detail View)", fontsize=16)
                ax_detail.set_ylabel(ylabel, fontsize=12)
                ax_detail.set_xlabel("Date", fontsize=12)
                ax_detail.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)
                ax_detail.tick_params(axis='x', labelrotation=45)
                ax_detail.grid(True)
                fig_detail.tight_layout()
                
                detail_path = path.replace('.png', '_screenIn.png')
                plt.savefig(detail_path, dpi=150)
                plt.close(fig_detail)
                print(f"Saved detail chart to: {detail_path}")
            
    except Exception as e:
        print(f"Error generating chart for {title}: {e}")
        print(f"DataFrame shape: {clean_df.shape}")
        print(f"DataFrame columns: {list(clean_df.columns)}")
        print(f"Data types: {clean_df.dtypes}")
        return


def main():
    parser = argparse.ArgumentParser(description="Generate detailed and aggregated TDCC trend charts with candlestick overlay.")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--base", default=script_dir, help=f"專案根目錄路徑。預設為本檔案所在目錄: {script_dir}")
    parser.add_argument("--input", required=True, help="Path to the Excel file from query_tddc.")
    parser.add_argument("--scheme", choices=["shares", "amount", "custom"], default="shares", help="Aggregation scheme for the 4th chart.")
    parser.add_argument("--price", type=float, default=0.0, help="Stock price, required for 'amount' scheme.")
    parser.add_argument("--custom-bins", type=str, default="", help="Custom bins for 'custom' scheme.")
    args = parser.parse_args()

    out_dir_detailed = os.path.join(args.base, "output", "trends_detailed")
    os.makedirs(out_dir_detailed, exist_ok=True)
    out_dir_aggregated = os.path.join(args.base, "output", "trends_aggregated")
    os.makedirs(out_dir_aggregated, exist_ok=True)
    data_dir_aggregated = os.path.join(args.base, "data", "trends_aggregated")
    os.makedirs(data_dir_aggregated, exist_ok=True)


    try:
        # --- FINAL, MOST ROBUST "GLOBAL SEARCH" PARSING LOGIC V2 ---
        # Read the excel file without any header or index assumptions
        raw_df = pd.read_excel(args.input, sheet_name="Combined_Report", header=None)

        # Function to find the exact coordinates (row, col) of a keyword
        def find_keyword_coords(df, keyword):
            for r_idx, row in df.iterrows():
                for c_idx, cell_value in enumerate(row):
                    if isinstance(cell_value, str) and keyword in cell_value:
                        return (r_idx, c_idx)
            return (None, None)

        # Find coordinates of all section headers
        coords = {
            "People": find_keyword_coords(raw_df, "人數"),
            "Shares": find_keyword_coords(raw_df, "股數"),
            "RatioPct": find_keyword_coords(raw_df, "佔比"),
            "OHLCV_Header": find_keyword_coords(raw_df, "周開盤價") # Note the different name
        }

        # Sort found sections by their row number
        sorted_sections = sorted([ (name, coord) for name, coord in coords.items() if coord[0] is not None ], key=lambda item: item[1][0])
        
        # Initialize empty dataframes
        people, shares, ratio, weekly_ohlcv = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Extract each section by finding its end point (the start of the next section)
        for i, (name, start_coord) in enumerate(sorted_sections):
            start_row = start_coord[0]
            end_row = None
            if i + 1 < len(sorted_sections):
                end_row = sorted_sections[i+1][1][0]
            
            # --- MODIFIED: Different logic for TDCC data vs OHLCV data ---
            if name in ["People", "Shares", "RatioPct"]:
                # Data is from the row after the header to the row before the next header
                section_data = raw_df.iloc[start_row + 1 : end_row]
                # First column is the index, the rest are data columns
                section_data = section_data.set_index(section_data.columns[0])
                # The header is in the same row as the keyword
                section_data.columns = raw_df.iloc[start_row, 1:]
                
                if name == "People": people = section_data
                elif name == "Shares": shares = section_data
                elif name == "RatioPct": ratio = section_data
            
            elif name == "OHLCV_Header":
                # For OHLCV, the section STARTS AT the header row
                ohlcv_raw = raw_df.iloc[start_row : end_row]
                ohlcv_raw = ohlcv_raw.set_index(ohlcv_raw.columns[0])
                ohlcv_raw.columns = raw_df.iloc[0, 1:] # Dates are in the first row
                
                if not ohlcv_raw.empty:
                    weekly_ohlcv = pd.DataFrame({
                        'open': ohlcv_raw.loc['周開盤價'], 'high': ohlcv_raw.loc['周最高價'],
                        'low': ohlcv_raw.loc['周最低價'], 'close': ohlcv_raw.loc['周收盤價'],
                        'vol': ohlcv_raw.loc['周成交量']
                    })
                    weekly_ohlcv.index = pd.to_datetime(weekly_ohlcv.index, errors='coerce')
                    weekly_ohlcv = weekly_ohlcv.apply(pd.to_numeric, errors='coerce').dropna()
                    print("Found and parsed weekly OHLCV data.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input}")
        return
    except Exception as e:
        print(f"Error reading or parsing the combined Excel file: {e}")
        return



    # Check if we have any data to process
    if people.empty and shares.empty and ratio.empty:
        print("No valid TDCC data sections found in the Excel file. Cannot generate charts.")
        return

    dfs_to_process = {"People": people, "Shares": shares, "RatioPct": ratio}
    dfs_processed = {}

    for name, df in dfs_to_process.items():
        if df.empty:
            dfs_processed[name] = pd.DataFrame()
            continue

        # --- FINAL FIX: Robustly convert all data to numeric, handling both ',' and '%' ---
        # First, set the column headers (dates) to datetime objects
        df.columns = pd.to_datetime(df.columns, errors='coerce')
        # Transpose the dataframe so dates become the index
        transposed_df = df.transpose()
        # Now, apply the numeric conversion to all columns at once, also removing the '%' sign for the ratio data
        transposed_df = transposed_df.apply(
            lambda x: pd.to_numeric(x.astype(str).str.replace(',', '').str.replace('%', ''), errors='coerce')
        )
        dfs_processed[name] = transposed_df


    people, shares, ratio = dfs_processed.get("People"), dfs_processed.get("Shares"), dfs_processed.get("RatioPct")
    ticker_stub = os.path.splitext(os.path.basename(args.input))[0].replace('_Combined', '')




    # Generate DETAILED charts
    print("\n--- Generating Detailed Charts (All Levels) ---")
    plot_and_save(people, weekly_ohlcv, f"{ticker_stub} 人數趨勢 (全級距)", os.path.join(out_dir_detailed, f"{ticker_stub}_detailed_people.png"), "人數")
    plot_and_save(shares, weekly_ohlcv, f"{ticker_stub} 股數趨勢 (全級距)", os.path.join(out_dir_detailed, f"{ticker_stub}_detailed_shares.png"), "股數")
    plot_and_save(ratio, weekly_ohlcv, f"{ticker_stub} 佔比趨勢 (全級距)", os.path.join(out_dir_detailed, f"{ticker_stub}_detailed_ratio.png"), "佔比 (%)")
    
    # Generate AGGREGATED charts
    print("\n--- Generating Aggregated Charts ---")
    label_to_cols, scheme_name = get_label_to_cols_map(args.scheme, args.price, args.custom_bins, people.columns.astype(str))
    agg_people = aggregate_data(people, label_to_cols)
    agg_shares = aggregate_data(shares, label_to_cols)
    agg_ratio = aggregate_data(ratio, label_to_cols)

    out_xlsx = os.path.join(data_dir_aggregated, f"{ticker_stub}_{scheme_name}.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        agg_people.to_excel(writer, sheet_name="Agg_People")
        agg_shares.to_excel(writer, sheet_name="Agg_Shares")
        agg_ratio.to_excel(writer, sheet_name="Agg_RatioPct")
    print(f"Saved aggregated Excel to: {out_xlsx}")

    plot_and_save(agg_people, weekly_ohlcv, f"{ticker_stub} 人數趨勢 ({scheme_name})", os.path.join(out_dir_aggregated, f"{ticker_stub}_{scheme_name}_people.png"), "人數")
    plot_and_save(agg_shares, weekly_ohlcv, f"{ticker_stub} 股數趨勢 ({scheme_name})", os.path.join(out_dir_aggregated, f"{ticker_stub}_{scheme_name}_shares.png"), "股數")
    plot_and_save(agg_ratio, weekly_ohlcv, f"{ticker_stub} 佔比趨勢 ({scheme_name})", os.path.join(out_dir_aggregated, f"{ticker_stub}_{scheme_name}_ratio.png"), "佔比 (%)")

    print(f"\nProcessing complete.")

if __name__ == "__main__":
    main()