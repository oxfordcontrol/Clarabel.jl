import yfinance as yf
import pandas as pd
################################################################
#YC: We use this file to download the initial data,
#    but we finally remove partially missing columns later on
################################################################

# List of 500 selected stocks from the S&P 500
top_100_stocks = [
    "AAPL", "MSFT", "AMZN", "META", "GOOGL", "GOOG", "BRK-B", "JNJ", "V", "PG",
    "NVDA", "JPM", "HD", "UNH", "MA", "PFE", "VZ", "MRK", "INTC", "KO",
    "T", "PEP", "DIS", "BAC", "CSCO", "XOM", "CMCSA", "CVX", "WFC", "BA",
    "MCD", "ABBV", "AMGN", "MDT", "TMO", "HON", "ACN", "AVGO", "IBM", "ADBE",
    "GS", "TXN", "CRM", "C", "LLY", "QCOM", "GE", "MMM", "PYPL", "NKE",
    "LOW", "NEE", "ORCL", "DHR", "WMT", "CVS", "NVO", "INTU", "AMD",
    "SPGI", "MDLZ", "PM", "SYK", "SCHW", "AXP", "TGT", "GS", "BLK", "COST",
    "USB", "AMT", "TMUS", "MS", "BKNG", "CAT", "LMT", "AMAT", "MU", "GILD",
    "RTX", "ISRG", "NOW", "LRCX", "SBUX", "CI", "ADI", "TJX", "MRNA", "ADP",
    "ZTS", "MMM", "BSX", "LULU", "DE", "FDX", "PGR", "F", "GM", "ITW",
    "SYF", "EBAY", "DXCM", "VRSN", "XEL", "WBA", "MPC", "TRV", "TT",
    "MO", "TFC", "HUM", "DG", "WM", "LHX", "BK", "EPAM", "ETSY", "CMG",
    "DD", "CDNS", "MNST", "WELL", "ROP", "CL", "FTNT", "STZ", "AEP",
    "IQV", "D", "BMY", "EL", "FIS", "RSG", "AON", "PSX", "APD",
    "ORLY", "ADM", "EXC", "CCI", "IDXX", "MSI", "MCO", "ED",
    "ILMN", "CARR", "BAX", "JCI", "BK", "TTWO", "HCA", "WST", "F",
    "KDP", "OXY", "CTSH", "CLX", "AWK", "PCAR", "LYV", "ECL", "KHC",
    "GIS", "ZBH", "ALL", "SBAC", "CNC", "SRE", "CTVA", "WMB", "PAYC", "VLO",
    "HAL", "VTR", "OKE", "EMR", "MTD", "MTCH", "EFX", "ATO", "GPC",
    "HSIC", "CHD", "OTIS", "WYNN", "DD", "CZR", "RMD", "HPE", "IR", "BKR",
    "IFF", "STE", "J", "ABMD", "LKQ", "GWW", "MKTX", "ANSS", "NTRS", "FMC",
    "FFIV", "AIZ", "XYL", "TER", "FTV", "XRAY", "WRK", "SLG", "IRM",
    "TPR", "CHRW", "TRMB", "MAS", "IP", "SNA", "RHI", "WY", "HST", "SEE",
    "BXP", "NDAQ", "STX", "PWR", "LEG", "CHTR", "FANG", "PFG",
    "EXPD", "HIG", "KEYS", "JBHT", "LNC", "HRL", "CF", "TXT", "DTE",
    "NUE", "KMI", "LW", "VFC", "CBOE", "WU", "HAS", "RF", "CFG",
    "FITB", "HBAN", "LUV", "TAP", "CMA", "PVH", "L", "UNM", "AIV",
    "HII", "VNO", "NRG", "REG", "OMC", "NWL", "MOS", "HES", "AOS", "ALK",
    "AEE", "CNP", "JBHT", "UAA", "UA", "BWA", "DXC", "KSS",
    "APA", "KIM", "ALB", "PRGO", "JWN", "NCLH", "COTY", "CPB", "SEE", "IRM",
    "HRB", "RHI", "KMX", "J", "SYY", "SWK", "AAP", "IEX", "EMN", "AVY",
    "MKC", "NVR", "PKG", "STZ", "MLM", "XYL", "QRVO", "HSIC", "HRL", "PGR",
    "POOL", "ROK", "ROL", "ROP", "SJM", "TXT", "TDY", "ZBRA", "CMS",
    "CE", "CINF", "CHD", "COO", "CMS", "CBRE", "CDW", "CEG", "CLX", "CPT",
    "CTAS", "DOV", "DXCM", "EIX", "EXPE", "EXR", "FAST", "FIS", "GL",
    "GRMN", "HES", "HII", "HST", "JKHY", "LDOS", "LW", "MAA", "MKC", "MKTX",
    "NDSN", "NWL", "PAYX", "PNR", "PPL", "RCL", "REG", "RMD",
    "RSG", "STE", "SNPS", "STT", "SUI", "TECH", "TDY", "TER", "TFX",
    "TROW", "TRMB", "TSCO", "ULTA", "WAB", "WRK", "WST", "ZBRA"
]

# Function to fetch data in batches
def fetch_data_in_batches(stocks, batch_size=100):
    all_returns = pd.DataFrame()
    
    for i in range(0, len(stocks), batch_size):
        batch_stocks = stocks[i:i + batch_size]
        try:
            data = yf.download(batch_stocks, period="5y")
            returns = data['Adj Close'].pct_change().dropna()
            all_returns = pd.concat([all_returns, returns], axis=1)
        except Exception as e:
            print(f"Error fetching data for batch {i // batch_size + 1}: {e}")
            
    all_returns.dropna(axis=1, how='any')
    
    return all_returns

# Fetch historical price data for the last 5 years in batches
returns = fetch_data_in_batches(top_100_stocks, batch_size=10)

# Save the returns to a CSV file
csv_file_path = "sp500_stock_returns_5y.csv"
returns.to_csv(csv_file_path)

print(f"Returns data saved to {csv_file_path}")