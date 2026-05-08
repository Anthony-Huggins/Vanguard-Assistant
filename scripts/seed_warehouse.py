"""Seed the fund-warehouse SQLite file used by the local ``db_query`` tool.

Run from the project root:

    ./.venv/Scripts/python.exe scripts/seed_warehouse.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "fund_warehouse.sqlite"


# Synthetic data — values match the corpus's "Style Bible" expense ratios
# but the return numbers are made up.
FUNDS = [
    # (ticker, share_class, name, expense_ratio, asset_class, inception_year, net_assets_billions)
    ("VTMSI", "Investor", "Total Stock Market Index Fund", 0.14, "US Equity", 1992, 412.0),
    ("VTMSA", "Admiral", "Total Stock Market Index Fund", 0.04, "US Equity", 2000, 803.0),
    ("VTMS", "ETF", "Total Stock Market ETF", 0.03, "US Equity", 2001, 412.0),
    ("V500I", "Investor", "500 Index Fund", 0.14, "US Equity", 1976, 270.0),
    ("V500A", "Admiral", "500 Index Fund", 0.04, "US Equity", 2000, 715.0),
    ("V500", "ETF", "500 Index ETF", 0.03, "US Equity", 2010, 460.0),
    ("VTBMI", "Investor", "Total Bond Market Index Fund", 0.15, "US Bonds", 1986, 95.0),
    ("VTBMA", "Admiral", "Total Bond Market Index Fund", 0.05, "US Bonds", 2001, 240.0),
    ("VTBM", "ETF", "Total Bond Market ETF", 0.04, "US Bonds", 2007, 100.0),
    ("VTISI", "Investor", "Total International Stock Index Fund", 0.17, "Intl Equity", 1996, 60.0),
    ("VTISA", "Admiral", "Total International Stock Index Fund", 0.10, "Intl Equity", 2010, 200.0),
    ("VTIS", "ETF", "Total International Stock ETF", 0.07, "Intl Equity", 2011, 90.0),
    ("VTR50", "Investor", "Target Retirement 2050 Fund", 0.08, "Multi-Asset", 2006, 58.4),
    ("VTR30", "Investor", "Target Retirement 2030 Fund", 0.08, "Multi-Asset", 2006, 89.0),
    ("VTRIN", "Investor", "Target Retirement Income Fund", 0.08, "Multi-Asset", 2003, 32.0),
    ("VHDY", "ETF", "High Dividend Yield ETF", 0.06, "US Equity", 2006, 63.7),
    ("VFMMF", "Investor", "Federal Money Market Fund", 0.11, "Money Market", 1981, 290.0),
]

# (ticker, year, total_return)
RETURNS = [
    ("V500", 2025, 12.04), ("V500", 2024, 26.10), ("V500", 2023, 26.20), ("V500", 2022, -18.05),
    ("VTMS", 2025, 11.50), ("VTMS", 2024, 25.30), ("VTMS", 2023, 25.90), ("VTMS", 2022, -19.50),
    ("VTBM", 2025,  3.20), ("VTBM", 2024,  1.30), ("VTBM", 2023,  5.60), ("VTBM", 2022, -13.10),
    ("VTIS", 2025,  9.40), ("VTIS", 2024,  6.10), ("VTIS", 2023, 15.40), ("VTIS", 2022, -16.20),
    ("VTR50", 2025, 10.80), ("VTR50", 2024, 19.20), ("VTR50", 2023, 19.80), ("VTR50", 2022, -16.40),
    ("VHDY", 2025,  9.20), ("VHDY", 2024, 14.80), ("VHDY", 2023,  6.40), ("VHDY", 2022, -1.00),
    ("VFMMF", 2025, 4.30), ("VFMMF", 2024, 5.20), ("VFMMF", 2023, 4.90), ("VFMMF", 2022, 1.50),
]


def main() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(
            """
            CREATE TABLE funds (
                ticker TEXT PRIMARY KEY,
                share_class TEXT NOT NULL,
                name TEXT NOT NULL,
                expense_ratio REAL NOT NULL,
                asset_class TEXT NOT NULL,
                inception_year INTEGER,
                net_assets_billions REAL
            );

            CREATE TABLE fund_returns (
                ticker TEXT NOT NULL,
                year INTEGER NOT NULL,
                total_return REAL NOT NULL,
                PRIMARY KEY (ticker, year),
                FOREIGN KEY (ticker) REFERENCES funds(ticker)
            );
            """
        )
        conn.executemany(
            "INSERT INTO funds VALUES (?, ?, ?, ?, ?, ?, ?)",
            FUNDS,
        )
        conn.executemany(
            "INSERT INTO fund_returns VALUES (?, ?, ?)",
            RETURNS,
        )
        conn.commit()
    finally:
        conn.close()
    print(f"Seeded {DB_PATH} with {len(FUNDS)} funds and {len(RETURNS)} return rows.")


if __name__ == "__main__":
    main()
