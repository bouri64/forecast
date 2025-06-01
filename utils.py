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
period = 'Y'
def find_closest_company(input_name, df, column_name='Name'):
    # Use fuzzywuzzy's process.extractOne to find the closest match
    closest_match = process.extractOne(input_name, df[column_name])
    
    if closest_match:
        return closest_match[0]  # Return the best match (name)
    else:
        return None  # In case no match is found
    

def adjust_sp(sp_df, minDate, maxDate):
    sp_df['Date'] = pd.to_datetime(sp_df.index.tz_localize(None))
    sp_limited = sp_df[sp_df['Date']>= minDate]
    sp_limited = sp_limited[sp_limited['Date']<= maxDate]
    return sp_limited

def load_company(company_name, financials_path, verbose = 1):
    financials = pd.read_csv(financials_path, index_col=False)
    closest_company = find_closest_company(company_name, financials)
    symbol = financials[financials['Name'] == closest_company].Symbol.iloc[0]
    if (verbose):
        print(f'The closest match to "{company_name}" is: "{closest_company}" with symbol: "{symbol}"')
    df = yf.Ticker(symbol).history(period="max")
    df['Date'] = pd.to_datetime(df.index.tz_localize(None))
    return df, closest_company

def load_sp(minDate, maxDate):
    sp = yf.Ticker("^GSPC").history(period="max")
    return adjust_sp(sp, minDate, maxDate)

def plot_company_vs_sp_df(name, df, sp_df, display = True, save = False, period='M', from_files = False, verbose = 1):
    # Process: Compute monthly average of High and Low for Company
    df['Time'] = df['Date'].dt.to_period(period)
    sp_df['Time'] = sp_df['Date'].dt.to_period(period)
    df_grouped = df.groupby('Time').agg({'Low': 'min', 'High': 'max'}).reset_index()
    sp_grouped = sp_df.groupby('Time').agg({'Low': 'min', 'High': 'max'}).reset_index()

    # Sort both DataFrames by date
    df_grouped = df_grouped.sort_values('Time')
    sp_grouped = sp_grouped.sort_values('Time')

    df_grouped['Time'] = df_grouped['Time'].dt.to_timestamp()
    sp_grouped['Time'] = sp_grouped['Time'].dt.to_timestamp()
    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Company
    ax1.plot(df_grouped['Time'].to_numpy(), df_grouped['Low'].to_numpy(), label=f'{name} Low', color='blue')
    ax1.plot(df_grouped['Time'].to_numpy(), df_grouped['High'].to_numpy(), label=f'{name} High', color='red')
    ax1.set_ylabel(f'{name} Price')
    ax1.set_xlabel('Date')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Create second y-axis for S&P 500
    ax2 = ax1.twinx()
    ax2.plot(sp_grouped['Time'].to_numpy(), sp_grouped['Low'].to_numpy(), label='S&P 500 Low', color='purple', linestyle='--')
    ax2.plot(sp_grouped['Time'].to_numpy(), sp_grouped['High'].to_numpy(), label='S&P 500 High', color='green', linestyle='--')
    ax2.set_ylabel('S&P 500 Index')
    ax2.legend(loc='upper right')

    plt.title(f'{name} vs. S&P 500: "{period}" High and Low Prices')
    plt.tight_layout()
    # if (save):
    #     plt.savefig(f".\output\{name}.png")  # Save as PNG
    # if (display):
    #     plt.show()
    # Save to memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    # Encode as base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{img_base64}" alt="Plot">'


def read_symbols():
    # response = supabase.table("symbols").select("Symbol,Name").execute()
    # df_symbols = pd.DataFrame(response.data)
    df_symbols = pd.DataFrame(data=companies_list, columns=['Symbol','Name'])
    return df_symbols

def read_company(symbol):
    df = yf.Ticker(symbol).history(period="max")
    df['Date'] = pd.to_datetime(df.index.tz_localize(None))
    return df

# plot_company_vs_sp_df(company_name, df_company, df_sp, display = True, save = True, period = period)