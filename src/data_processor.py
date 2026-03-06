"""
Data processing module for World Bank indicators.
Handles extraction, cleaning, imputation, and scaling of global development data.
"""

import pandas as pd
import numpy as np
import wbgapi as wb
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


# Curated list of 18 important World Bank indicators
CURATED_INDICATORS = {
    # Economic
    "NY.GDP.PCAP.CD": "GDP per Capita (USD)",
    "NY.GNP.PCAP.CD": "GNI per Capita (USD)",
    "NE.TRD.GNFS.CD": "Trade (% of GDP)",
    "FP.CPI.TOTL.ZG": "Inflation (annual %)",
    
    # Demographic
    "SP.DYN.LE00.IN": "Life Expectancy (years)",
    "SP.POP.GROW": "Population Growth (annual %)",
    "SP.DYN.TFRT.IN": "Fertility Rate (births per woman)",
    "SP.URB.TOTL.IN.ZS": "Urban Population (% of total)",
    
    # Education
    "SE.ADT.LITR.ZS": "Literacy Rate (% age 15+)",
    "SE.PRM.ENRR": "Primary School Enrollment (gross %)",
    "SE.SEC.ENRR": "Secondary School Enrollment (gross %)",
    "SE.TER.ENRR": "Tertiary School Enrollment (gross %)",
    
    # Health
    "SP.DYN.IMRT.IN": "Infant Mortality (per 1,000 live births)",
    "SP.DYN.CDRT.IN": "Child Mortality (per 1,000 children)",
    "SH.XPD.CHEX.GD.ZP": "Health Expenditure (% of GDP)",
    
    # Environment
    "EN.ATM.CO2E.PC": "CO2 Emissions (metric tons per capita)",
    "AG.LND.FRST.ZS": "Forest Area (% of land)",
    "EG.ELC.ACCS.ZS": "Electricity Access (% of population)",
}


def get_curated_indicators() -> Dict[str, str]:
    """
    Returns the curated list of important World Bank indicators.
    
    Returns:
        dict: {indicator_code: indicator_name}
    """
    return CURATED_INDICATORS.copy()


def get_all_countries() -> List[str]:
    """
    Retrieves all countries available in World Bank database.
    
    Returns:
        list: List of country codes
    """
    try:
        countries = wb.data.list('NY.GDP.PCAP.CD', time=2023)
        country_codes = [item[0] for item in countries]
        return country_codes
    except Exception as e:
        print(f"Error fetching countries: {e}")
        return []


def fetch_indicator_data(
    indicator_code: str,
    year: int = 2023,
) -> pd.Series:
    """
    Fetches data for a single indicator across all countries for a specific year.
    
    Args:
        indicator_code: World Bank indicator code
        year: Year to fetch (default: 2023)
    
    Returns:
        pd.Series: Country codes as index, values as data
    """
    try:
        # Fetch data for indicator across all years
        df = wb.data.DataFrame(indicator_code, time=range(year-2, year+1))
        
        if df.empty:
            return pd.Series(dtype='float64')
        
        # Get most recent available data (try current year first, then backwards)
        result = pd.Series(dtype='float64')
        for yr_col in [f'YR{year}', f'YR{year-1}', f'YR{year-2}']:
            if yr_col in df.columns:
                series = df[yr_col].dropna()
                if len(series) > 0:
                    result = series
                    break
        
        if result.empty:
            return pd.Series(dtype='float64')
        
        result.index.name = 'country'
        return result
    
    except Exception as e:
        print(f"  Error fetching {indicator_code}: {str(e)[:50]}")
        return pd.Series(dtype='float64')


def fetch_indicators(
    indicator_codes: Optional[List[str]] = None,
    use_curated: bool = True,
    all_countries: bool = True,
    most_recent_year_only: bool = True
) -> pd.DataFrame:
    """
    Fetches multiple indicators and merges them into a single DataFrame.
    
    Args:
        indicator_codes: List of indicator codes (None = use curated list)
        use_curated: If True, use predefined curated indicators
        all_countries: If True, include all countries from World Bank
        most_recent_year_only: If True, use only latest available year
    
    Returns:
        pd.DataFrame: Countries x Indicators matrix
    """
    if use_curated:
        indicators = CURATED_INDICATORS
    elif indicator_codes:
        indicators = {code: code for code in indicator_codes}
    else:
        indicators = CURATED_INDICATORS
    
    print(f"Fetching {len(indicators)} indicators...")
    
    dfs = []
    for i, (code, name) in enumerate(indicators.items()):
        try:
            print(f"  [{i+1}/{len(indicators)}] Fetching {name}...")
            
            # Fetch data from World Bank (latest available year)
            series = fetch_indicator_data(code, year=2023)
            
            if series.empty:
                print(f"    [!] No data available for {code}")
                continue
            
            # Rename to indicator name
            series.name = name
            dfs.append(series)
            
        except Exception as e:
            print(f"    [X] Error processing {code}: {str(e)[:50]}")
            continue
    
    if not dfs:
        print("No data retrieved. Creating empty DataFrame.")
        return pd.DataFrame()
    
    # Merge all indicators on country index
    result = pd.concat(dfs, axis=1, sort=True)
    result.index.name = 'Country'
    
    print(f"[OK] Fetched {result.shape[0]} countries x {result.shape[1]} indicators")
    return result


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans data by removing rows/columns with all missing values.
    
    Args:
        df: Input DataFrame
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Remove columns with all NaN
    df = df.dropna(axis=1, how='all')
    
    # Remove rows with all NaN
    df = df.dropna(axis=0, how='all')
    
    return df


def calculate_missingness(df: pd.DataFrame) -> Dict:
    """
    Calculates missing data statistics.
    
    Args:
        df: Input DataFrame
    
    Returns:
        dict: Missingness statistics
    """
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
    
    # Per column
    col_missing = (df.isna().sum() / len(df) * 100).to_dict()
    
    # Per row
    row_missing = (df.isna().sum(axis=1) / df.shape[1] * 100).to_dict()
    
    return {
        'total_missing_pct': missing_pct,
        'total_cells': total_cells,
        'missing_cells': missing_cells,
        'per_column': col_missing,
        'per_row': row_missing,
        'rows': df.shape[0],
        'columns': df.shape[1]
    }


def impute_missing(
    df: pd.DataFrame,
    n_neighbors: int = 5,
    method: str = 'knn',
    metric: str = 'nan_euclidean'
) -> pd.DataFrame:
    """
    Imputes missing values using KNN or other methods.
    
    Args:
        df: Input DataFrame with missing values
        n_neighbors: Number of neighbors for KNN imputation
        method: Imputation method ('knn' or 'mean')
        metric: Distance metric for KNN (default 'nan_euclidean' which handles NaNs)
    
    Returns:
        pd.DataFrame: DataFrame with imputed values
    """
    df_imputed = df.copy()
    
    if method == 'knn':
        print(f"Applying KNN imputation (n_neighbors={n_neighbors}, metric={metric})...")
        imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance', metric=metric)
        df_imputed.iloc[:, :] = imputer.fit_transform(df)
    
    elif method == 'mean':
        print("Applying mean imputation...")
        df_imputed = df.fillna(df.mean())
    
    else:
        print(f"Unknown imputation method: {method}")
        return df
    
    return df_imputed


def scale_features(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scales features to have mean=0 and std=1.
    
    Args:
        df: Input DataFrame
        scaler: Optional pre-fitted scaler (for test set)
        fit: If True, fit scaler on data; if False, use provided scaler
    
    Returns:
        Tuple of (scaled_DataFrame, scaler_object)
    """
    if fit or scaler is None:
        print("Fitting StandardScaler...")
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df),
            index=df.index,
            columns=df.columns
        )
    else:
        print("Applying pre-fitted StandardScaler...")
        df_scaled = pd.DataFrame(
            scaler.transform(df),
            index=df.index,
            columns=df.columns
        )
    
    return df_scaled, scaler


def process_data_pipeline(
    indicator_codes: Optional[List[str]] = None,
    use_curated: bool = True,
    n_neighbors: int = 5,
    metric: str = 'nan_euclidean',
    apply_imputation: bool = True,
    apply_scaling: bool = True
) -> Tuple[pd.DataFrame, StandardScaler, Dict]:
    """
    Full data processing pipeline: fetch → clean → impute → scale.
    
    Args:
        indicator_codes: List of indicator codes
        use_curated: Use curated indicator list
        n_neighbors: KNN neighbors for imputation
        metric: Distance metric for KNN (default 'nan_euclidean')
        apply_imputation: Whether to apply KNN imputation
        apply_scaling: Whether to scale features
    
    Returns:
        Tuple of (processed_data, scaler, stats_dict)
    """
    print("\n" + "="*60)
    print("DATA PROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Fetch indicators
    print("\nStep 1: Fetching indicators from World Bank API...")
    df = fetch_indicators(
        indicator_codes=indicator_codes,
        use_curated=use_curated,
        most_recent_year_only=True
    )
    
    if df.empty:
        raise ValueError("No data fetched from World Bank API")
    
    stats = {}
    stats['initial'] = calculate_missingness(df)
    print(f"  Initial: {df.shape[0]} countries × {df.shape[1]} indicators")
    print(f"  Missing: {stats['initial']['total_missing_pct']:.1f}%")
    
    # Step 2: Clean data
    print("\nStep 2: Cleaning data...")
    df = clean_data(df)
    stats['after_cleaning'] = calculate_missingness(df)
    print(f"  After cleaning: {df.shape[0]} countries × {df.shape[1]} indicators")
    
    # Step 3: Impute missing values
    if apply_imputation:
        print("\nStep 3: Imputing missing values...")
        df = impute_missing(df, n_neighbors=n_neighbors, method='knn', metric=metric)
        stats['after_imputation'] = calculate_missingness(df)
        print(f"  After imputation: {stats['after_imputation']['total_missing_pct']:.1f}% missing")
    
    # Step 4: Scale features
    scaler = None
    if apply_scaling:
        print("\nStep 4: Scaling features...")
        df, scaler = scale_features(df, fit=True)
        print(f"  Features scaled: mean≈0, std≈1")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60 + "\n")
    
    return df, scaler, stats


if __name__ == "__main__":
    # Example usage
    try:
        df, scaler, stats = process_data_pipeline(
            n_neighbors=5,
            apply_imputation=True,
            apply_scaling=True
        )
        
        print("\nFinal Data Shape:", df.shape)
        print("\nFirst 5 countries:")
        print(df.head())
        
        print("\nData Summary:")
        print(df.describe())
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
