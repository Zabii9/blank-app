# Salesflo End Stock Summary Bot

Automated daily bot that fetches **End Stock Summary** (Daily | Summary | Value)
from `engrofoods.salesflo.com` and saves it to MySQL.

---

## How It Works

1. Runs every day at **10:00 PM**
2. Checks MySQL for the **last saved date**
3. Fetches all missing dates from `last_date + 1` up to **yesterday (N-1)**
4. Logs into Salesflo, sets filters, waits for report to complete, downloads CSV
5. Parses and upserts rows into MySQL

---

## Setup

### 1. Clone / copy files
```bash
mkdir salesflo_bot && cd salesflo_bot
# copy bot.py, requirements.txt, .env, salesflo_bot.service here
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
playwright install chromium
playwright install-deps chromium
```

### 4. Configure `.env`
Edit `.env` and fill in your actual MySQL credentials:
```
DB_HOST=your-mysql-host.example.com
DB_PORT=3306
DB_USER=your_db_username
DB_PASS=your_db_password
DB_NAME=salesflo_data
```

> ⚠️ **Important:** Change your Salesflo password and update `.env` accordingly.

### 5. Test run manually
```bash
source venv/bin/activate
python bot.py
```

---

## Running as a Background Service (Linux systemd)

### 1. Edit the service file
Update `salesflo_bot.service`:
- Replace `YOUR_LINUX_USER` with your Linux username
- Replace `/path/to/salesflo_bot` with the actual folder path

### 2. Install & enable the service
```bash
sudo cp salesflo_bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable salesflo_bot
sudo systemctl start salesflo_bot
```

### 3. Check status & logs
```bash
sudo systemctl status salesflo_bot
tail -f /var/log/salesflo_bot.log
# or
tail -f salesflo_bot.log   # local log file in bot directory
```

---

## Database Tables Created Automatically

### `end_stock_summary`
| Column | Type | Description |
|---|---|---|
| id | INT PK | Auto increment |
| report_date | DATE | Date of the report |
| distributor | VARCHAR | Distributor name |
| sku | VARCHAR | Product SKU |
| brand | VARCHAR | Brand name |
| opening_stock | DECIMAL | Opening stock value |
| closing_stock | DECIMAL | Closing stock value |
| sold_qty | DECIMAL | Sold quantity |
| value | DECIMAL | Stock value |
| unit | VARCHAR | Unit type (Value) |
| fetched_at | DATETIME | When row was saved |

### `bot_run_log`
| Column | Type | Description |
|---|---|---|
| id | INT PK | Auto increment |
| run_date | DATE | Date the bot ran for |
| status | ENUM | success / failed / no_data |
| rows_saved | INT | Number of rows saved |
| message | TEXT | Error message if failed |
| created_at | DATETIME | When log entry was created |

---

## Troubleshooting

| Issue | Solution |
|---|---|
| Login fails | Check credentials in `.env`, site may have changed login page |
| Report stays Pending | Increase timeout in `click_view_report_and_wait()` (default 5 min) |
| Download not found | Inspect site with DevTools, update CSS selectors in `download_and_parse()` |
| MySQL connection error | Verify DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME in `.env` |
| CSV columns don't match | Open a downloaded CSV, check column headers, update `download_and_parse()` |

---

## File Structure
```
salesflo_bot/
├── bot.py                  # Main bot script
├── requirements.txt        # Python dependencies
├── .env                    # Credentials & config (keep secret!)
├── salesflo_bot.service    # systemd service file
├── salesflo_bot.log        # Runtime log (auto-created)
└── downloads/              # Temp downloaded CSVs (auto-created)
```
