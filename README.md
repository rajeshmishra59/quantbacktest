Quantitative Backtesting Engine "Formula 1"
Ek high-performance, professional-grade backtesting engine jo vectorbt ki raftaar aur ek custom event-driven portfolio simulator ki sachchai ko jodta hai.

Vartamaan Sthiti (Current Status)
Last Updated: August 6, 2025

Project ne apna initial setup aur refactoring phase poora kar liya hai. Humne ek purane system ko ek saaf, modular, aur professional "Formula 1" engine mein badal diya hai. Engine ab poori tarah se taiyaar hai jismein hum apni custom trading strategies ko integrate aur test kar sakte hain.

Ab Tak Poore Kiye Gaye Milestones:

✅ Project Structure: Ek saaf-suthra, industry-standard folder structure banaya gaya hai.

✅ Hybrid Engine Core: vectorbt (fast signal generation) aur UpgradedPortfolio (realistic, event-driven simulation) ko safaltapoorvak joda gaya hai.

✅ Advanced Risk Management: Portfolio simulator mein 2% per-trade risk aur 5% max-daily-loss jaise zaroori risk niyam laagu kiye gaye hain.

✅ Modular & Flexible: Engine ab alag-alag symbols, timeframes, aur strategies ke liye poori tarah se taiyaar hai.

✅ Persistent Results: Saare backtest ke results ab ek dedicated SQLite database (data/backtest_results.db) mein save hote hain, jisse alag-alag runs ko compare karna aasan hai.

Project Structure
quantbacktest/
│
├── config.py                 # Sabhi mukhya settings (DB path, risk params)
├── hybrid_backtest_runner.py # Engine ko chalaane waali mukhya file
├── requirements.txt          # Zaroori Python packages
├── .gitignore                # Git ke liye ignore ki jaane waali files
│
├── data/                     # Saara data (market data, result DBs)
├── engine/                   # Core simulation logic (UpgradedPortfolio)
├── strategies/               # Aapki saari trading strategies yahan aayengi
└── utils/                    # Helper functions (metrics, loaders, etc.)

Quick Start Guide
1. Setup
a. Environment Banayein:

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows

b. Dependencies Install Karein:

pip install -r requirements.txt

c. Configuration Set Karein:
config.py file ko kholein aur MARKET_DATA_DB_PATH variable mein apne 10-saal ke data waale database ka sahi path daalein.

2. Backtest Chalayein
Apne terminal se, project ke root folder mein yeh command chalayein:

python hybrid_backtest_runner.py

Runner aapse symbols aur timeframe ke baare mein poochhega.

Agla Kadam (Next Steps)
Engine ki neenv (foundation) ab taiyaar hai. Hamara agla aur sabse mahatvapurna kaam hai:

🎯 Integrate Custom Strategies: Apne strategies/ folder mein maujood asli strategies (AlphaOne, Apex, NumeroUno, etc.) ko naye "signal generator" format mein convert karna aur unhe is engine par backtest karna.

Is README file ko hum har bade update ke baad modify karenge.