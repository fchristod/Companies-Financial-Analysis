# Companies-Financial-Analysis
Navigating the Financial Landscape through Python &amp; SQL

🎯 Objectives:
Utilize SQL queries to extract and summarize specific portions of the financial data.

Conduct in-depth data analysis using Python.

Use libraries such as pandas, numpy, matplotlib, seaborn, and plotly to scrutinize the data further and generate meaningful visualizations.

# Dataset before SQL query: "company_financial_data.csv"
📊 Columns in the Dataset:

The dataset comprises several key columns, each providing specific financial data points:

Symbol: The stock symbol under which the company is traded. 📈

Name: The official name of the company. 🏢

Sector: The sector of the economy to which the company belongs. 🌐

Price: The current trading price of the company’s stock. 💲

Price/Earnings: The ratio for valuing a company that measures its current share price relative to its per-share earnings. 🔍

Dividend Yield: A financial ratio that shows how much a company pays out in dividends each year relative to its share price. 💸

Earnings/Share: The portion of a company's profit allocated to each outstanding share of common stock. 🧾

52 Week Low: The lowest share price of the company in the last 52 weeks. 🔽

52 Week High: The highest share price of the company in the last 52 weeks. 🔼

Market Cap: The total market value of a company's outstanding shares. 🏦

EBITDA: Earnings before interest, taxes, depreciation, and amortization, which is an indicator of a company's financial performance. 📊

Price/Sales: The ratio that compares a company's stock price to its revenues. 📉

Price/Book: A ratio used to compare a firm's market value to its book value. 📚

# Dataset after SQL query: "company_financial.csv"
📊 New Columns in the Dataset:

PB_PE Ratio (Price/Book to Price/Earnings ratio)

Price_Level: CASE statement to categorize each company based on its current stock price in relation to its 52-week high and low. Categories include 'High', 'Moderate', or 'Low'

Market_Cap_RanK: Window function to rank companies within their respective sectors based on their market capitalization

# Filtering for Significant Performers: 
Applying filters to focus on companies with an EBITDA greater than $1 billion and a PE Ratio between 15 and 30. This criterion helps in isolating companies that are financially significant and moderately valued

# Sorting the Data
Arranging the results by Sector and Market Cap Rank

# Python Section

1: Interactive Financial Correlation Analysis

2: Financial Data Clustering and 3D Visualization

3: Sector-wise Financial Ratio Analysis and Visualization


