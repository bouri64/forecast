from fuzzywuzzy import process
import matplotlib.pyplot as plt
import pandas as pd
from supabase import create_client, Client
import io
import base64
import yfinance as yf
from companies import companies_list

# url = "https://upbgpsqskumjhfrqefbt.supabase.co"
# public_anon_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVwYmdwc3Fza3VtamhmcnFlZmJ0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDcwODE5MDcsImV4cCI6MjA2MjY1NzkwN30.sK_OPvxn2S0L4UeCUkjRV1jJm7bwfc55SRP7tRQ4FhQ"
# supabase: Client = create_client(url, public_anon_key)

# !pip install fuzzywuzzy
# !pip install python-Levenshtein

# company_name = 'Apple'

def find_closest_company(input_name, df, column_name='Name'):
    # Use fuzzywuzzy's process.extractOne to find the closest match
    closest_match = process.extractOne(input_name, df[column_name])
    
    if closest_match:
        return closest_match[0]  # Return the best match (name)
    else:
        return None  # In case no match is found
    
def find_closest_company_from_dict(input_name, companies: dict):
    # Use fuzzywuzzy's process.extractOne to find the closest match
    closest_match = process.extractOne(input_name, companies.keys())
    
    if closest_match:
        return closest_match[0]  # Return the best match (name)
    else:
        return None  # In case no match is found
    

def adjust_sp(sp_df, minDate, maxDate):
    # sp_df['Date'] = pd.to_datetime(sp_df.index.tz_localize(None))
    # sp_limited = sp_df[sp_df['Date']>= minDate]
    # sp_limited = sp_limited[sp_limited['Date']<= maxDate]
    return sp_df

def load_company(company_name, financials_path, min_date, max_date, verbose = 1):
    financials = pd.read_csv(financials_path, index_col=False)
    closest_company = find_closest_company(company_name, financials)
    symbol = financials[financials['Name'] == closest_company].Symbol.iloc[0]
    if (verbose):
        print(f'The closest match to "{company_name}" is: "{closest_company}" with symbol: "{symbol}"')
    df = yf.Ticker(symbol).history(start=min_date, end=max_date)
    # df['Date'] = pd.to_datetime(df.index.tz_localize(None))
    return df, closest_company

def load_sp(minDate, maxDate):
    sp = yf.Ticker("^GSPC").history(start=minDate, end=maxDate)
    return adjust_sp(sp, minDate, maxDate)

def plot_company_vs_sp_df(name, df, sp_df, display = True, save = False, period='M', from_files = False, verbose = 1):
    # Group by period and aggregate
    agg_funcs = {'Low': 'min', 'High': 'max'}

    df_grouped = df.resample(period).agg(agg_funcs).sort_index()
    sp_grouped = sp_df.resample(period).agg(agg_funcs).sort_index()

    # Convert to numpy arrays for plotting
    time_df = df_grouped.index.to_numpy()
    time_sp = sp_grouped.index.to_numpy()

    df_low, df_high = df_grouped['Low'].to_numpy(), df_grouped['High'].to_numpy()
    sp_low, sp_high = sp_grouped['Low'].to_numpy(), sp_grouped['High'].to_numpy()

    # Create plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(time_df, df_low, label=f'{name} Low', color='blue')
    ax1.plot(time_df, df_high, label=f'{name} High', color='red')
    ax1.set_ylabel(f'{name} Price')
    ax1.set_xlabel('Date')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    # Twin axis for S&P 500
    ax2 = ax1.twinx()
    ax2.plot(time_sp, sp_low, label='S&P 500 Low', color='purple', linestyle='--')
    ax2.plot(time_sp, sp_high, label='S&P 500 High', color='green', linestyle='--')
    ax2.set_ylabel('S&P 500 Index')
    ax2.legend(loc='upper right')

    plt.title(f'{name} vs. S&P 500: "{period}" High and Low Prices')
    plt.tight_layout()

    # Save to buffer and encode as base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{img_base64}" alt="Plot">'


def read_symbols():
    # response = supabase.table("symbols").select("Symbol,Name").execute()
    # df_symbols = pd.DataFrame(response.data)
    # return df_symbols
    return companies_list

def read_company(symbol, min_date, max_date):
    df = yf.Ticker(symbol).history(start=min_date, end=max_date)
    df['Date'] = pd.to_datetime(df.index.tz_localize(None))
    return df

# plot_company_vs_sp_df(company_name, df_company, df_sp, display = True, save = True, period = period)