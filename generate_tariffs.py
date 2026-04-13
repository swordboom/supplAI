import pandas as pd
import random
from pathlib import Path

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent
SUPPLY_PATH = PROJECT_ROOT / "data" / "supply_chain.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "wits_tariffs.csv"

def generate_tariffs():
    if not SUPPLY_PATH.exists():
        print(f"Error: {SUPPLY_PATH} not found.")
        return

    df = pd.read_csv(SUPPLY_PATH)
    
    # Extract unique countries and their regions
    country_regions = df[['country', 'region']].drop_duplicates()
    
    countries_list = list(country_regions['country'])
    region_mapping = dict(zip(country_regions['country'], country_regions['region']))
    
    tariffs = []
    
    # Generate pairwise tariffs
    for src in countries_list:
        for dst in countries_list:
            if src == dst:
                rate = 0.0 # Domestic is 0%
            else:
                src_region = region_mapping[src]
                dst_region = region_mapping[dst]
                
                # Intra-region simulates Free Trade Agreements (FTA) like EU, USMCA, ASEAN
                if src_region == dst_region:
                    # High chance of 0%, else small tariff
                    rate = 0.0 if random.random() < 0.8 else round(random.uniform(1.0, 3.0), 1)
                else:
                    # Inter-region trade (MFN baseline)
                    # Varies based on random factors, simulating different product codes
                    # Base rate MFN
                    base = round(random.uniform(2.0, 15.0), 1)
                    
                    # Some specific realistic high-tension corridors
                    if (src == "China" and dst == "USA") or (src == "USA" and dst == "China"):
                        base += round(random.uniform(15.0, 25.0), 1) # Section 301 simulation
                        
                    rate = base
                    
            tariffs.append({
                "Source_Country": src,
                "Destination_Country": dst,
                "Tariff_Rate_Pct": rate
            })
            
    out_df = pd.DataFrame(tariffs)
    OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Generated {len(out_df)} pairwise tariff rates mapped to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_tariffs()
