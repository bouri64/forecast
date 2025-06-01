# your_script.py
import sys
from utils import *

# Check if a name was provided
if len(sys.argv) < 2:
    print("No name provided!")
    sys.exit(1)

name = sys.argv[1]
df_symbols = read_symbols()
# closest_company = find_closest_company(name, df_symbols)
# symbol = df_symbols[df_symbols['Name'] == closest_company].Symbol.iloc[0]
closest_company = find_closest_company_from_dict(name, df_symbols)
symbol = df_symbols[closest_company]
print(closest_company, "---", symbol)
df_company = read_company(symbol)
minDate = df_company['Date'].min()
maxDate = df_company['Date'].max()
df_sp = load_sp(minDate, maxDate)
image = plot_company_vs_sp_df(name, df_company, df_sp, display = True, save = True, period = period)
# print(f"Plot saved to: output/{name}.png")
print(image)