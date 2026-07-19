#mcandrew

"""
Extract ILI (Influenza-Like Illness) data from Delphi Epidata API
for all US states from 2009 to present.
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
from epiweeks import Week

# Install epidata if needed: pip install delphi-epidata
try:
    from delphi_epidata import Epidata
except ImportError:
    print("ERROR: delphi-epidata package not installed.")
    print("Install it with: pip install delphi-epidata")
    sys.exit(1)


# State FIPS codes mapping
STATE_FIPS = {
    1: 'AL', 2: 'AK', 4: 'AZ', 5: 'AR', 6: 'CA', 8: 'CO', 9: 'CT', 10: 'DE',
    11: 'DC', 12: 'FL', 13: 'GA', 15: 'HI', 16: 'ID', 17: 'IL', 18: 'IN', 19: 'IA',
    20: 'KS', 21: 'KY', 22: 'LA', 23: 'ME', 24: 'MD', 25: 'MA', 26: 'MI', 27: 'MN',
    28: 'MS', 29: 'MO', 30: 'MT', 31: 'NE', 32: 'NV', 33: 'NH', 34: 'NJ', 35: 'NM',
    36: 'NY', 37: 'NC', 38: 'ND', 39: 'OH', 40: 'OK', 41: 'OR', 42: 'PA', 44: 'RI',
    45: 'SC', 46: 'SD', 47: 'TN', 48: 'TX', 49: 'UT', 50: 'VT', 51: 'VA', 53: 'WA',
    54: 'WV', 55: 'WI', 56: 'WY', -1:"nat",72:"PR"
}

STATE_NAMES = {
    1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 6: 'California',
    8: 'Colorado', 9: 'Connecticut', 10: 'Delaware', 11: 'District of Columbia',
    12: 'Florida', 13: 'Georgia', 15: 'Hawaii', 16: 'Idaho', 17: 'Illinois',
    18: 'Indiana', 19: 'Iowa', 20: 'Kansas', 21: 'Kentucky', 22: 'Louisiana',
    23: 'Maine', 24: 'Maryland', 25: 'Massachusetts', 26: 'Michigan', 27: 'Minnesota',
    28: 'Mississippi', 29: 'Missouri', 30: 'Montana', 31: 'Nebraska', 32: 'Nevada',
    33: 'New Hampshire', 34: 'New Jersey', 35: 'New Mexico', 36: 'New York',
    37: 'North Carolina', 38: 'North Dakota', 39: 'Ohio', 40: 'Oklahoma', 41: 'Oregon',
    42: 'Pennsylvania', 44: 'Rhode Island', 45: 'South Carolina', 46: 'South Dakota',
    47: 'Tennessee', 48: 'Texas', 49: 'Utah', 50: 'Vermont', 51: 'Virginia',
    53: 'Washington', 54: 'West Virginia', 55: 'Wisconsin', 56: 'Wyoming', -1:"nat",72:"Puerto Rico"
}


def get_epiweek_range(start_year, end_year):
    """
    Generate list of epiweeks from start_year to end_year.
    
    Args:
        start_year: Starting year (e.g., 2009)
        end_year: Ending year (e.g., 2024)
    
    Returns:
        List of epiweek integers in YYYYWW format
    """
    epiweeks = []
    for year in range(start_year, end_year + 1):
        # Get number of weeks in this year
        #try:
        #    last_week = Week(year, 52).week
        #    max_week = 52
        #except:
        try:
            last_week = Week(year, 53).week
            max_week = 53
        except:
            max_week = 52
        
        for week in range(1, max_week + 1):
            epiweek = int(f"{year}{week:02d}")
            epiweeks.append(epiweek)
    
    return epiweeks


def fetch_ili_data_for_state(state_abbr, epiweeks):
    """
    Fetch ILI data for a specific state using Epidata API.
    
    Args:
        state_abbr: State abbreviation (e.g., 'pa', 'ny')
        epiweeks: List of epiweeks to fetch
    
    Returns:
        DataFrame with ILI data
    """
    # Epidata expects lowercase state abbreviations
    state_abbr_lower = state_abbr.lower()
    
    # Fetch data using Epidata.fluview
    # Can fetch ranges: Epidata.range(start, end)
    epiweek_start = min(epiweeks)
    epiweek_end   = max(epiweeks)
    
    print(f"  Fetching {state_abbr} data for epiweeks {epiweek_start} to {epiweek_end}...")
    
    response = Epidata.fluview(regions=state_abbr_lower, epiweeks=Epidata.range(epiweek_start, epiweek_end))
    
    if response['result'] != 1:
        print(f"  WARNING: No data returned for {state_abbr}. Result code: {response['result']}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    data = pd.DataFrame(response['epidata'])
    return data


def process_ili_data(df, state_fips, state_abbr, state_name):
    """
    Process raw ILI data and add useful columns.
    
    Args:
        df: Raw data from Epidata
        state_fips: State FIPS code
        state_abbr: State abbreviation
        state_name: State full name
    
    Returns:
        Processed DataFrame
    """
    if len(df) == 0:
        return df
    
    # Add state information
    df['state_fips'] = state_fips
    df['state_abbr'] = state_abbr
    df['state_name'] = state_name
    
    # Extract year and week from epiweek
    df['year'] = df['epiweek'].astype(str).str[:4].astype(int)
    df['week'] = df['epiweek'].astype(str).str[4:].astype(int)
    
    # Add season column
    def get_season(row):
        if row['week'] >= 40:
            return f"{row['year']}/{row['year']+1}"
        else:
            return f"{row['year']-1}/{row['year']}"
    
    df['season'] = df.apply(get_season, axis=1)
    
    # Select and rename key columns
    cols_to_keep = [
        'epiweek', 'year', 'week', 'season', 
        'state_fips', 'state_abbr', 'state_name',
        'wili', 'ili',  # Weighted ILI and ILI
        'num_ili', 'num_patients', 'num_providers',  # Raw counts
        'num_age_0', 'num_age_1', 'num_age_2', 'num_age_3', 'num_age_4', 'num_age_5',  # Age groups
        'release_date', 'issue'  # Metadata
    ]
    
    # Only keep columns that exist
    cols_available = [col for col in cols_to_keep if col in df.columns]
    df = df[cols_available]
    
    return df

def main():
    """Main function to extract ILI data for all states."""
    
    # Configuration
    START_YEAR = 2021
    END_YEAR = datetime.now().year
    OUTPUT_FILE = "./analysis_data/ili_data_all_states_2021_present.csv"
    
    print("="*70)
    print(f"Extracting ILI Data from Delphi Epidata")
    print("="*70)
    print(f"Year range: {START_YEAR} - {END_YEAR}")
    print(f"Number of states: {len(STATE_FIPS)}")
    print()
    
    # Generate epiweek range
    epiweeks = get_epiweek_range(START_YEAR, END_YEAR)

    print(epiweeks)
    
    print(f"Epiweeks range: {min(epiweeks)} to {max(epiweeks)} ({len(epiweeks)} weeks total)")
    print()
    
    # Collect data for all states
    all_data = []
    
    for state_fips, state_abbr in STATE_FIPS.items():
        state_name = STATE_NAMES[state_fips]
        print(f"Processing {state_name} ({state_abbr})...")
        
        try:
            # Fetch data
            state_data = fetch_ili_data_for_state(state_abbr, epiweeks)
            
            if len(state_data) > 0:
                # Process data
                state_data = process_ili_data(state_data, state_fips, state_abbr, state_name)
                all_data.append(state_data)
                print(f"  ✓ Retrieved {len(state_data)} records")
            else:
                print(f"  ✗ No data available")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
        
        print()
    
    # Combine all data
    if all_data:
        print("Combining all state data...")
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Sort by state and epiweek
        combined_data = combined_data.sort_values(['state_fips', 'epiweek'])
        
        # Save to CSV
        print(f"Saving to {OUTPUT_FILE}...")
        combined_data.to_csv(OUTPUT_FILE, index=False)
        
        # Summary statistics
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Total records: {len(combined_data):,}")
        print(f"States with data: {combined_data['state_abbr'].nunique()}")
        print(f"Date range: {combined_data['epiweek'].min()} to {combined_data['epiweek'].max()}")
        print(f"Seasons covered: {sorted(combined_data['season'].unique())}")
        print(f"\nOutput saved to: {OUTPUT_FILE}")
        
        # Show sample
        print("\nSample data:")
        print(combined_data.head(10))
        
        print("\n✓ Complete!")
    else:
        print("ERROR: No data retrieved for any state.")


if __name__ == "__main__":
    main()

