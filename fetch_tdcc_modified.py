import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


TDCC_BASE_URL = "https://www.tdcc.com.tw"
TDCC_QRY_PATH = "/portal/zh/smWeb/qryStock"


def get_stocks_from_csv(file_path: str) -> List[Tuple[str, str]]:
    """Reads stock codes and names from a local CSV file."""
    if not os.path.exists(file_path):
        print(f"Error: Stock list file not found at {file_path}")
        return []
    try:
        # Use default UTF-8 encoding as the user has saved the file in this format
        df = pd.read_csv(file_path)
        df['stock_code'] = df['stock_code'].astype(str)
        df['stock_name'] = df['stock_name'].astype(str)
        stock_list = list(zip(df['stock_code'], df['stock_name']))
        print(f"Loaded {len(stock_list)} stocks from {os.path.basename(file_path)}.")
        return stock_list
    except Exception as e:
        print(f"Error reading stock list from {file_path}: {e}")
        return []

class TdccClient:
    def __init__(self, session: Optional[requests.Session] = None, timeout: int = 30):
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
            }
        )
        self.session.verify = False
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self.timeout = timeout

    def get_query_page(self) -> BeautifulSoup:
        resp = self.session.get(TDCC_BASE_URL + TDCC_QRY_PATH, timeout=self.timeout)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")

    @staticmethod
    def _parse_options(select_el: BeautifulSoup) -> List[Tuple[str, str]]:
        options: List[Tuple[str, str]] = []
        for opt in select_el.find_all("option"):
            value = (opt.get("value") or "").strip()
            text = opt.get_text(strip=True)
            if value:
                options.append((value, text))
        return options

    def list_available_dates(self) -> List[Tuple[str, str]]:
        soup = self.get_query_page()
        form = soup.find("form")
        if not form:
            raise RuntimeError("TDCC query form not found; site structure may have changed.")

        selects = form.find_all("select")
        date_options: List[Tuple[str, str]] = []
        for i, sel in enumerate(selects):
            name = (sel.get("name") or "").lower()
            if any(k in name for k in ["date", "week", "sca", "ym"]):
                date_options = self._parse_options(sel)
                print(f"Found {len(date_options)} date options.")
                break
        return date_options

    def submit_query(self, date_value: str, stock_value: str) -> BeautifulSoup:
        soup = self.get_query_page()
        form = soup.find("form")
        if not form: raise RuntimeError("TDCC query form not found.")
        action = form.get("action") or TDCC_QRY_PATH
        method = (form.get("method") or "GET").upper()

        payload: Dict[str, str] = {}
        for inp in form.find_all(["input", "select"]):
            name = inp.get("name")
            if not name: continue
            if inp.name == "input":
                if name == "stockNo":
                    payload[name] = stock_value
                elif name == "sqlMethod":
                    payload[name] = "sqlStockNo"
                else:
                    payload[name] = inp.get("value") or payload.get(name, "")
            elif inp.name == "select":
                if any(k in name.lower() for k in ["date", "week", "sca", "ym"]):
                    payload[name] = date_value
                else:
                    opts = inp.find_all("option")
                    if opts: payload[name] = (opts[0].get("value") or "").strip()

        url = action if action.startswith("http") else (TDCC_BASE_URL + action)
        resp = self.session.post(url, data=payload, timeout=self.timeout) if method == "POST" else self.session.get(url, params=payload, timeout=self.timeout)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")

def parse_distribution_table(soup: BeautifulSoup) -> Optional[pd.DataFrame]:
    tables = soup.find_all("table")
    target = next((tbl for tbl in tables if any(k in " ".join(th.get_text(strip=True) for th in tbl.find_all("th")) for k in ["人數", "股數", "占集保庫存數比例"])) , None)
    if target is None: return None
    headers = [th.get_text(strip=True) for th in target.find_all("tr")[0].find_all(["th", "td"])]
    rows = [[td.get_text(strip=True) for td in tr.find_all(["th", "td"])] for tr in target.find_all("tr")[1:]]
    return pd.DataFrame([row for row in rows if len(row) == len(headers)], columns=headers)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_index(index_path: str) -> pd.DataFrame:
    if os.path.exists(index_path):
        try: return pd.read_csv(index_path, dtype=str)
        except Exception: return pd.DataFrame(columns=["date", "ticker", "saved_path"])
    return pd.DataFrame(columns=["date", "ticker", "saved_path"])

def save_index(index_path: str, df: pd.DataFrame) -> None:
    df.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"]).to_csv(index_path, index=False)

def is_first_run(raw_root: str) -> bool:
    if not os.path.exists(raw_root): return True
    return not any(os.path.isdir(os.path.join(raw_root, item)) and re.fullmatch(r"\d{4,6}", item) for item in os.listdir(raw_root))

def get_date_range_for_first_run(date_options: List[Tuple[str, str]], weeks: int = 52) -> List[str]:
    return [v for v, _ in date_options][:weeks]

def main():
    # Correctly expand the user's home directory with a single tilde
    default_stock_list_path = os.path.join(os.path.expanduser("~"), "Documents", "stock_list.csv")
    parser = argparse.ArgumentParser(description="Fetch TDCC shareholder distribution data from a local stock list CSV.")
    parser.add_argument("--base", required=True, help="Base directory to store the data.")
    parser.add_argument("--stock-list", default=default_stock_list_path, help=f"Path to the stock list CSV file. Defaults to {default_stock_list_path}")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--sleep", type=float, default=0.6, help="Seconds to sleep between requests")
    parser.add_argument("--first-run-weeks", type=int, default=52, help="How many weeks to fetch on first run (default: 52 weeks = 1 year)")
    args = parser.parse_args()

    base_dir = args.base
    data_root = os.path.join(base_dir, "data")
    raw_root = os.path.join(data_root, "tdcc_raw")
    ensure_dir(data_root)
    ensure_dir(raw_root)

    stock_options = get_stocks_from_csv(args.stock_list)
    if not stock_options:
        print("No stocks to process. Exiting.")
        return

    index_path = os.path.join(data_root, "tdcc_index.csv")
    index_df = load_index(index_path)
    have = set((row["date"], row["ticker"]) for _, row in index_df.iterrows())

    client = TdccClient()
    print("Fetching available dates from TDCC...")
    date_options = client.list_available_dates()
    if not date_options: 
        print("Could not fetch date options from TDCC. Exiting.")
        return

    if is_first_run(raw_root):
        print("First run detected. Will fetch data for the past year...")
        date_values = get_date_range_for_first_run(date_options, args.first_run_weeks)
        print(f"First run: Using {len(date_values)} dates (past {args.first_run_weeks} weeks)")
    else:
        date_values = [v for v, _ in date_options]
        print(f"Subsequent run: Checking against all {len(date_values)} available dates to backfill missing data.")

    tasks = [(date_value, ticker) for date_value in date_values for ticker, _ in stock_options if (date_value, ticker) not in have]

    if not tasks:
        print("No new tasks. Index up-to-date.")
        return

    print(f"Planned downloads: {len(tasks)}")

    def worker(date_value: str, ticker: str) -> Optional[Tuple[str, str, str]]:
        try:
            soup = client.submit_query(date_value=date_value, stock_value=ticker)
            df = parse_distribution_table(soup)
            if df is None or df.empty: return None
            stock_dir = os.path.join(raw_root, ticker)
            ensure_dir(stock_dir)
            out_path = os.path.join(stock_dir, f"{date_value}.csv")
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            return (date_value, ticker, out_path)
        except Exception as e:
            # print(f"Failed to fetch {ticker} for {date_value}: {e}") # Optional: for debugging
            return None

    results: List[Tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(worker, d, t): (d, t) for d, t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetching"):
            res = fut.result()
            if res: results.append(res)
            time.sleep(args.sleep)

    if results:
        add_df = pd.DataFrame(results, columns=["date", "ticker", "saved_path"])
        index_df = pd.concat([index_df, add_df], ignore_index=True)
        save_index(index_path, index_df)
        print(f"Saved {len(results)} new items. Index at {index_path}")
    else:
        print("No new data saved. Check if site structure changed or filters too strict.")

if __name__ == "__main__":
    main()