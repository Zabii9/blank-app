# Bazaar Prime Analytics — End User Guide

This guide explains the latest dashboard behavior, filters, and calculation logic for all reports.

## 1) Access

- Open the Streamlit app URL provided by your admin.
- Login with your assigned username/password.
- Use Logout from sidebar when needed.

---

## 2) Global Filters (Apply across tabs)

### Date / Period filter
Available options:
- Last 7 Days
- Last 30 Days
- This Month
- Last Month
- Last 3 Months
- YTD
- Custom Range

### Account / Location filter
Distributor selection (example mappings):
- Karachi (`D70002202`)
- Lahore (`D70002246`)

### How period comparison works
- KPI deltas compare against a prior base period (previous month / previous year / previous snapshot depending on metric).
- If comparison base is missing, delta may show `-`.

---

## 3) Tab Structure (Current)

1. 📈 Sales Growth Analysis  
2. 🎯 Booker Performance  
3. 🧭 Booker & Field Force Deep Analysis  
4. 📦 Inventory  
5. 🧪 Custom Query Runner

---

## 4) Filters by Report Area

## Tab 1 — Sales Growth Analysis

Common controls in this tab:
- Value vs Litre basis toggles on growth/performance visuals
- Channel filters (where applicable)
- Brand/DM drill controls in hierarchical charts

What filters affect:
- Most charts respect global period + account.
- Additional local controls (channel, basis, brand) apply only to the relevant chart section.

## Tab 2 — Booker Performance

Common controls:
- Period selector (available periods)
- Channel filter
- Achievement cutoff filters
- Brand multiselect
- Flow direction and Top/Bottom controls in Sankey

What filters affect:
- Treemap/Sankey/heatmaps respect global period + account, plus local controls.

## Tab 3 — Booker & Field Force Deep Analysis

Common controls:
- Booker multiselect
- Channel multiselect
- Additional local chart/table controls (where shown)

What filters affect:
- Route, calls, cohort, segmentation, scoring, and leaderboard modules use global filters plus their local selectors.

## Tab 4 — Inventory

Common controls:
- Global period + account (required)
- Stock Cover Bucket Settings (B1..B5) for matrix bucketing

What filters affect:
- Inventory KPIs, trend charts, and health metrics use latest available stock snapshot up to selected end date.
- Sales-linked inventory logic uses trailing windows (mostly 30-day and 90-day).
- Bucket settings affect only the Stock Cover Days matrix.

## Tab 5 — Custom Query Runner

Controls:
- SQL editor
- Sample query loader
- Schema browser + column search

Rules:
- Only single read-only `SELECT` statement allowed.
- Mutating operations are blocked.

---

## 5) Core Metric Logic (Business Definitions)

## A) Sales / Order KPIs

- **Revenue** = `SUM(Delivered Amount + Total Discount)`
- **Orders** = distinct invoice count
- **AOV** = `Revenue / Orders`

## B) Inventory KPI Cards

- **Current Inventory**  
  Latest stock snapshot total value (YTD month-end sparkline shows last stock day of each month).

- **Total SKUs**  
  Distinct SKU count in current stock snapshot.

- **In-Stock Rate (%)**  
  `In-stock SKUs / Total SKUs × 100`

- **OOS SKUs**  
  SKU count where stock value <= 0.

- **Avg Days Cover** (weighted by SKU contribution; 90-day basis)  
  For each SKU:  
  `days_cover_i = stock_i / (sales90_i / 90)`  
  `contribution_i = sales90_i / total_sales90`  
  Final:  
  `Avg Days Cover = Σ(days_cover_i × contribution_i)`

- **Slow Movers**  
  SKUs where `stock > 0` and `sales_30 <= 0`.

## C) Inventory Trend Charts

- **Stock Movement Trend (30 days)**
  - Inflow = positive day-over-day stock increase
  - Outflow = daily delivered sales value

- **Days Cover by Category**
  - Category cover days = `Category Stock / (Category Sales30 / 30)`
  - Visual thresholds:
    - Red: `<7`
    - Amber: `7–14`
    - Green: `>14`

## D) Stock Cover Days Matrix (as per Salesflo)

For each SKU:
- `cover_days = stock_value / (sales90 / 90)` when `sales90 > 0`
- `cover_days = ∞` when `sales90 = 0` (goes to Zero Sale)

Bucket columns are dynamic using B1..B5:
- `Inventory <B1`
- `Inventory B1-B2`
- `Inventory B2-B3`
- `Inventory B3-B4`
- `Inventory B4-B5`
- `Inventory B5+`
- `Zero Sale`

Matrix rules:
- Rows are account-wise (no total row).
- Top percentage row = each bucket share of total inventory across displayed accounts.

## E) Inventory Health

- **Inventory Turnover**  
  `Sales_90 / Avg_Stock_90`

- **Dead Stock Value**  
  Sum of SKU stock where:
  - sales contribution in 90-day sales `< 2%`, and
  - cover days `> 25`

- **Wastage Rate (%)**  
  `Dead Stock Value / Total Current Stock × 100`

- **Avg Replenishment Cycle (days)**  
  Average gap between positive stock refill events.

- **GMROII (proxy)**  
  `Sales_30 / Avg_Stock_30`

- **Safety Stock Coverage (days)**  
  `Total Stock / (Sales_30 / 30)`

- **Dead Stock SKU-wise Detail**
  Includes SKU Name, Stock Value, Sales 90D, Sales Contribution %, Cover Days.

---

## 6) Custom Query Logic

Allowed:
- One `SELECT` query only

Blocked:
- `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `CREATE`, multi-statement scripts

Output:
- Preview table + CSV download

---

## 7) Reading Deltas Correctly

- Arrow/Color indicates movement vs comparison base.
- Some KPIs use inverted-good logic (example: OOS lower is better).
- If base period has no data, delta may be neutral/blank.

---

## 8) Troubleshooting

### No data shown
- Expand period range.
- Verify account filter.
- Clear local filters in tab section.

### Inventory values look unusually high/low
- Check selected account and end date.
- Check bucket settings (B1..B5) for matrix interpretation.

### Custom Query blocked
- Ensure query is a single read-only `SELECT`.

---

## 9) Support Checklist

When reporting an issue, share:
- Selected period
- Selected account/distributor
- Tab and report section name
- Screenshot
- If SQL related: the exact query used

---

## 10) Business Glossary

- **AOV (Average Order Value)**: Average value per invoice (`Revenue / Orders`).
- **Cover Days**: Estimated days current stock can sustain at recent average daily sales.
- **Dead Stock**: Low-contribution SKUs (`<2%` of 90-day sales) with high cover (`>25 days`).
- **GMROII (Proxy)**: Return-on-inventory proxy using sales and average stock for recent period.
- **In-Stock Rate**: Share of SKUs with positive stock.
- **Inventory Turnover**: How fast inventory converts into sales (`Sales_90 / Avg_Stock_90`).
- **OOS (Out of Stock)**: SKUs with stock value less than or equal to zero.
- **Par Level Inventory**: Target inventory level based on desired cover days and daily sales run-rate.
- **Productive Calls**: Planned calls that converted into productive outcomes as per visit logic.
- **Safety Stock Coverage**: Buffer days available from current stock at 30-day average sales rate.
- **Sales Contribution %**: SKU share of total sales in the selected lookback window.
- **Slow Movers**: SKUs in stock but with no sales in last 30 days.
- **Stock Snapshot**: Latest available stock state (value/units) on a specific report date.
- **YTD (Year to Date)**: Period from start of current year to selected end date.
