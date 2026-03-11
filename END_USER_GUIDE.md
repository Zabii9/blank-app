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

1. 🗂️ Summary
2. 📈 Sales Growth Analysis
3. 🎯 Booker Performance
4. 🧭 Booker & Field Force Deep Analysis
5. 📦 Inventory
6. 🧪 Custom Query
7. 🤖 Bot Runner
8. 🧩 Missing Data (red indicator appears if issue detected)

Notes:
- **Brand Coverage — Per Visit, Per Shop** Summary tab ke andar render hota hai (separate tab nahi hai).
- **Missing Data** tab label red (`🔴`) ho sakta hai jab unknown brand SKUs, missing targets, ya missing shops detect hon.

---

## 4) Filters by Report Area

## Summary Tab

Summary tab me high-level executive view + brand coverage module dono available hain.

Common controls in this area:
- Global period + location filters
- Brand Coverage local filters:
  - Booker multiselect
  - Focus Brands multiselect (runtime add/remove)
  - Table rows selector
  - `Missed shops only` toggle
  - Sunburst detail table filters: Date and Brand

What filters affect:
- Summary KPI cards, under-performer view, funnel, SKU/Bill and brand focus charts global period/account follow karte hain.
- Brand Coverage visuals visit-date logic follow karte hain and include:
  - coverage donut, miss-rate heatmap, booker-wise brand bars
  - daily brand sunburst + shop detail table
  - daily coverage matrix (Booker x Date)
  - shop-wise and booker drill-down tables

## Tab 1 — Sales Growth Analysis

Common controls in this tab:
- Value vs Litre basis toggles on growth/performance visuals
- Channel filters (where applicable)
- Deliveryman growth basis toggles (`vs LM` / `vs LY`) for DM comparison charts

What filters affect:
- Most charts respect global period + account.
- Additional local controls (channel, basis, brand) apply only to the relevant chart section.

### Deliveryman Growth Panels (Tab 1)

- Left panel: **Deliveryman-wise Growth Percentage**
  - Basis switch: `vs LY` (default) or `vs LM`
  - Metric basis follows tab filter (`Value` or `Litre`)
- Right panel: **Deliveryman-wise Unique Productivity Growth**
  - Basis switch: `vs LY` (default) or `vs LM`
  - Uses unique productive shops (distinct shops billed by DM)
- Both panels are aligned with the same deliveryman order for easy row-by-row comparison.

## Tab 2 — Booker Performance

Common controls:
- Period selector (available periods)
- Channel filter
- Achievement cutoff filters
- Brand filter (segmented multi-select)
- Treemap hierarchy toggle (Booker→Brand or Brand→Booker)
- Flow direction and Top/Bottom controls in Sankey
- Calendar month window + booker filter for GMV calendar heatmap

What filters affect:
- Treemap, Sankey, and GMV calendar heatmap respect global period + account plus local controls.

## Tab 3 — Booker & Field Force Deep Analysis

Common controls:
- Booker multiselect
- Channel multiselect
- Additional local chart/table controls (where shown)
- Leaderboard view toggle (Top 5 / Bottom 5 / All)
- Booker/segment table filters in activity segmentation subsection

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

## Tab 5 — Custom Query

Controls:
- SQL editor
- Sample query loader
- Schema browser + column search

Rules:
- Only single read-only `SELECT` statement allowed.
- Mutating operations are blocked.

## Tab 6 — Bot Runner

Controls:
- Password unlock for Bot Runner tab
- Start/Stop bot
- Refresh logs
- Log lines slider
- Live log popup
- `RefreshData` (password protected, selected period refresh)

Behavior:
- Bot status with PID shown when running.
- RefreshData selected period ke liye bot process run karke DB refresh trigger karta hai.

## Tab 7 — Missing Data

Checks:
- Unknown Brand SKU (missing in master or brand mapping issue)
- Current month target missing / missing Booker+Brand targets against sales
- Missing shops in universe (present in order data, missing in universe master)

Outputs:
- On-screen issue tables
- CSV export buttons for each check

---

## 5) Core Metric Logic (Business Definitions)

## A) Summary (Top Cards)

Summary cards dashboard ka quick health snapshot dete hain selected period + distributor ke basis par.

- **Current Value**: selected period ka actual metric value.
- **Delta vs LY**: same period last year ke against percentage change.
- **Delta vs LM**: previous month comparable window ke against percentage change.
- **Arrow/Color Meaning**:
  - Green = positive movement
  - Red = negative movement
  - Neutral/blank = comparison base missing ya zero-base condition
- **Sparkline (agar card par show ho)**: recent trend direction ko highlight karta hai, exact totals nahi.

Common summary metrics:
- **Revenue** = `SUM(Delivered Amount + Total Discount)`
- **Litres/Volume** = delivered litres + kg-equivalent volume (model logic ke mutabiq)
- **Orders** = distinct invoices
- **AOV** = `Revenue / Orders`

## B) Sales / Order KPIs

- **Revenue** = `SUM(Delivered Amount + Total Discount)`
- **Orders** = distinct invoice count
- **AOV** = `Revenue / Orders`

## C) Inventory KPI Cards

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

## D) Inventory Trend Charts

- **Stock Movement Trend (30 days)**
  - Inflow = positive day-over-day stock increase
  - Outflow = daily delivered sales value

- **Days Cover by Category**
  - Category cover days = `Category Stock / (Category Sales30 / 30)`
  - Visual thresholds:
    - Red: `<7`
    - Amber: `7–14`
    - Green: `>14`

## E) Stock Cover Days Matrix (as per Salesflo)

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

## F) Inventory Health

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

## G) Field Force / Brand Coverage Metrics

- **Assigned Visits**: Distinct planned/executed visit IDs in selected scope.
- **Assigned Shops**: Distinct shops assigned/visited in selected scope.
- **Productive Visits**: Visits marked productive under visit/order logic.
- **Productive Shops**: Distinct shops with productive visits.
- **Shops Sold**: Distinct shops where selected brand had sales on visit date.
- **Unique Productivity (DM)**: Distinct shops billed by deliveryman in selected period.
- **Growth vs LY / LM**:
  - `((Current - Base) / Base) × 100`
  - Base is Last Year or Last Month for the same date window.
  - If base is `0` and current is positive, growth is treated as `100%` in chart view.

Brand Coverage visuals include:
- Daily sunburst split by selected Focus Brands + `Other`
- Daily coverage matrix (Booker x Date)
- Right-side drill table with full assigned shops including no-order rows

---

## 6) Custom Query Logic

Allowed:
- One `SELECT` query only

Blocked:
- `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `CREATE`, multi-statement scripts

Output:
- Preview table + CSV download

Related operational tabs:
- **Bot Runner**: password-protected automation controls for bot start/stop/logs and period refresh.
- **Missing Data**: diagnostics tab for unknown brand SKU, missing targets, and missing universe shops.

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

### Growth % looks too high/low
- Check selected growth basis (`vs LY` or `vs LM`).
- Confirm comparison period has non-zero base values.
- For new/returning DMs with zero base, chart may show capped positive growth behavior.

### Inventory values look unusually high/low
- Check selected account and end date.
- Check bucket settings (B1..B5) for matrix interpretation.

### Custom Query blocked
- Ensure query is a single read-only `SELECT`.

### Bot Runner unlock/start issue
- Verify Bot Runner password from admin.
- Ensure bot is not already running before `RefreshData`.
- If logs are empty, click refresh and increase log lines.

### Missing Data tab showing red indicator
- Open `🧩 Missing Data` tab and review all three checks.
- Export CSV and share with data/master-data owner for correction.

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
