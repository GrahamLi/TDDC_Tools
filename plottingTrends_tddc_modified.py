import argparse
import os
import re
from typing import List, Tuple, Dict

import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf # Import the new library for candlestick charts

# Set a font that supports Chinese characters
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Heiti TC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"Font setting warning: {e}")


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
    if df.empty: return

    # --- Prepare data for mplfinance ---
    ohlc_data = weekly_ohlc[['open', 'high', 'low', 'close', 'vol']].copy()
    ohlc_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Create overlays (trend lines) for the main plot
    ap_lines = [mpf.make_addplot(df[col], panel=0, ylabel=ylabel) for col in df.columns]
    
    # Create plot using mplfinance
    fig, axes = mpf.plot(ohlc_data, 
                         type='candle', 
                         style='yahoo',
                         title=f'\n{title}',
                         volume=True, 
                         addplot=ap_lines,
                         figsize=(20, 10),
                         panel_ratios=(3, 1),
                         returnfig=True)
    
    # Customize legend
    axes[0].legend([col.replace('_', ' ') for col in df.columns], fontsize='small')
    
    # Save the main chart
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved main chart to: {path}")

    # --- Detail Plot for flat lines ---
    flat_cols = [col for col in df.columns if is_line_flat(df[col])]
    if flat_cols:
        print(f"Detected {len(flat_cols)} near-flat lines. Creating detail chart...")
        fig_detail, ax_detail = plt.subplots(figsize=(15, 8))
        for col in flat_cols:
            ax_detail.plot(df.index, df[col], label=str(col), marker='.', markersize=4, linestyle='-')
        
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


def main():
    parser = argparse.ArgumentParser(description="Generate detailed and aggregated TDCC trend charts with candlestick overlay.")
    parser.add_argument("--base", required=True, help="Base directory.")
    parser.add_argument("--input", required=True, help="Path to the Excel file from query_tddc.")
    parser.add_argument("--scheme", choices=["shares", "amount", "custom"], default="shares", help="Aggregation scheme for the 4th chart.")
    parser.add_argument("--price", type=float, default=0.0, help="Stock price, required for 'amount' scheme.")
    parser.add_argument("--custom-bins", type=str, default="", help="Custom bins for 'custom' scheme.")
    args = parser.parse_args()

    # Define output directories
    out_dir_detailed = os.path.join(args.base, "output", "trends_detailed")
    os.makedirs(out_dir_detailed, exist_ok=True)
    out_dir_aggregated = os.path.join(args.base, "output", "trends_aggregated")
    os.makedirs(out_dir_aggregated, exist_ok=True)
    data_dir_aggregated = os.path.join(args.base, "data", "trends_aggregated")
    os.makedirs(data_dir_aggregated, exist_ok=True)

    try:
        xls = pd.ExcelFile(args.input)
        people = pd.read_excel(xls, sheet_name="People", index_col=0)
        shares = pd.read_excel(xls, sheet_name="Shares", index_col=0)
        ratio = pd.read_excel(xls, sheet_name="RatioPct", index_col=0)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input}")
        return
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # --- Prepare Weekly OHLCV Data ---
    if '周收盤價' in people.index:
        # Extract OHLCV from the respective rows
        weekly_ohlcv = pd.DataFrame({
            'open': people.loc['周開盤價'] if '周開盤價' in people.index else None,
            'high': people.loc['周最高價'] if '周最高價' in people.index else None,
            'low': people.loc['周最低價'] if '周最低價' in people.index else None,
            'close': people.loc['周收盤價'],
            'vol': people.loc['周成交量']
        })
        weekly_ohlcv.index = pd.to_datetime(weekly_ohlcv.index, format='%Y%m%d')
        weekly_ohlcv = weekly_ohlcv.apply(pd.to_numeric, errors='coerce').dropna()
        print("Found weekly OHLCV data.")
    else:
        weekly_ohlcv = pd.DataFrame()
        print("Weekly OHLCV data not found. Charts will not include price/volume.")

    dfs_raw = {"People": people, "Shares": shares, "RatioPct": ratio}
    dfs_processed = {}

    for name, df in dfs_raw.items():
        rows_to_drop = ['周開盤價', '周最高價', '周最低價', '周收盤價', '周成交量', '合計']
        df.drop(rows_to_drop, errors='ignore', inplace=True)
        df.columns = pd.to_datetime(df.columns, format='%Y%m%d')
        transposed_df = df.transpose()

        for col in transposed_df.columns:
            transposed_df[col] = pd.to_numeric(transposed_df[col].astype(str).str.replace(',', ''), errors='coerce')
        dfs_processed[name] = transposed_df

    people, shares, ratio = dfs_processed["People"], dfs_processed["Shares"], dfs_processed["RatioPct"]
    ticker_stub = os.path.splitext(os.path.basename(args.input))[0]

    # --- 1. Generate and save DETAILED (15+ levels) charts ---
    print("\n--- Generating Detailed Charts (All Levels) ---")
    plot_and_save(people, weekly_ohlcv, f"{ticker_stub} 人數趨勢 (全級距)", os.path.join(out_dir_detailed, f"{ticker_stub}_detailed_people.png"), "人數")
    plot_and_save(shares, weekly_ohlcv, f"{ticker_stub} 股數趨勢 (全級距)", os.path.join(out_dir_detailed, f"{ticker_stub}_detailed_shares.png"), "股數")
    plot_and_save(ratio, weekly_ohlcv, f"{ticker_stub} 佔比趨勢 (全級距)", os.path.join(out_dir_detailed, f"{ticker_stub}_detailed_ratio.png"), "佔比 (%)")
    
    # --- 2. Generate and save AGGREGATED charts ---
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