# your_script.py
import sys
from utils import *
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Check if a name was provided
if len(sys.argv) < 3:
    print("Not enough arguments!")
    sys.exit(1)

name = sys.argv[1]
period = sys.argv[2]
min_date = pd.Timestamp(f"{sys.argv[3]}-01-01")
max_date = pd.Timestamp(f"{sys.argv[4]}-12-31")

df_symbols = read_symbols()
# closest_company = find_closest_company(name, df_symbols)
# symbol = df_symbols[df_symbols['Name'] == closest_company].Symbol.iloc[0]

start_time = time.time()

print(name)
if name in df_symbols.keys():
    closest_company = name
else:
    closest_company = find_closest_company_from_dict(name, df_symbols)

symbol = df_symbols[closest_company]
logger.info(closest_company, "---", symbol)
df_company = read_company(symbol, min_date, max_date)
min_date = max(df_company['Date'].min(), min_date)
max_date = min(df_company['Date'].max(), max_date)
df_sp = load_sp(min_date, max_date)
image = plot_company_vs_sp_df(name, df_company, df_sp, display = True, save = True, period = period)
logger.info(f"Plot saved to: output/{name}.png")
end_time = time.time()
elapsed = end_time - start_time
logger.info(f"Script took {elapsed:.3f} seconds")

print(image)