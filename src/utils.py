"""
Utility functions for the Global Cluster Distribution application.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


def get_iso3_codes() -> Dict[str, str]:
    """
    Returns mapping of country names to ISO-3 codes for Plotly choropleth mapping.
    Uses World Bank country codes which are already ISO-3 in most cases.
    
    Returns:
        dict: {country_code: iso3_code}
    """
    # World Bank mostly uses ISO-3 codes already
    # This is a simplified mapping; wbgapi returns ISO-3 codes in most cases
    iso3_mapping = {
        # Common mappings
        'USA': 'USA',
        'CHN': 'CHN',
        'IND': 'IND',
        'BRA': 'BRA',
        'GBR': 'GBR',
        'DEU': 'DEU',
        'FRA': 'FRA',
        'JPN': 'JPN',
        'CAN': 'CAN',
        'AUS': 'AUS',
        # Add more as needed
    }
    return iso3_mapping


def split_train_validation(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: Optional[int] = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data into training and validation sets.
    
    Args:
        df: Input DataFrame
        test_size: Proportion for validation set (0.0-1.0)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, validation_df)
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    n_samples = len(df)
    n_validation = int(n_samples * test_size)
    
    indices = np.arange(n_samples)
    if random_state is not None:
        np.random.seed(random_state)
    
    np.random.shuffle(indices)
    
    val_indices = indices[:n_validation]
    train_indices = indices[n_validation:]
    
    train_df = df.iloc[train_indices].copy()
    val_df = df.iloc[val_indices].copy()
    
    return train_df, val_df


def get_cluster_profiles(
    data: pd.DataFrame,
    labels: np.ndarray,
    cluster_names: Optional[Dict[int, str]] = None
) -> pd.DataFrame:
    """
    Generates cluster profiles showing mean feature values per cluster.
    
    Args:
        data: Original scaled or unscaled data
        labels: Cluster assignments
        cluster_names: Optional mapping of cluster ID to name
    
    Returns:
        pd.DataFrame: Cluster profiles with mean values
    """
    data_with_labels = data.copy()
    data_with_labels['Cluster'] = labels
    
    profiles = data_with_labels.groupby('Cluster').mean()
    
    if cluster_names:
        profiles.index = profiles.index.map(
            lambda x: cluster_names.get(x, f'Cluster {x}')
        )
    else:
        profiles.index = [f'Cluster {i}' for i in profiles.index]
    
    return profiles


def get_feature_importance(
    data: pd.DataFrame,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Calculates feature importance by cluster separation.
    Features with higher between-cluster variance are more important.
    
    Args:
        data: Input data
        labels: Cluster assignments
    
    Returns:
        dict: {feature_name: importance_score}
    """
    between_cluster_variance = {}
    
    for col in data.columns:
        # Total variance
        total_var = data[col].var()
        
        # Within-cluster variance
        within_var = 0
        for cluster_id in np.unique(labels):
            cluster_data = data[labels == cluster_id][col]
            within_var += cluster_data.var() * len(cluster_data) / len(data)
        
        # Between-cluster variance
        between_var = total_var - within_var if total_var > 0 else 0
        importance = between_var / (total_var + 1e-10)  # Normalize
        between_cluster_variance[col] = importance
    
    return between_cluster_variance


def format_metrics(metrics: Dict) -> str:
    """
    Formats model metrics for display.
    
    Args:
        metrics: Dictionary of metrics
    
    Returns:
        str: Formatted metrics string
    """
    output = []
    for key, value in metrics.items():
        if isinstance(value, float):
            output.append(f"{key}: {value:.4f}")
        else:
            output.append(f"{key}: {value}")
    
    return "\n".join(output)


def get_top_features_per_cluster(
    profiles: pd.DataFrame,
    n_features: int = 5
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Identifies top distinguishing features for each cluster.
    
    Args:
        profiles: Cluster profiles DataFrame
        n_features: Number of top features to return
    
    Returns:
        dict: {cluster_name: [(feature_name, value), ...]}
    """
    top_features = {}
    
    for cluster_name in profiles.index:
        cluster_values = profiles.loc[cluster_name]
        # Get features with highest absolute values (most distinctive)
        top_indices = np.argsort(np.abs(cluster_values))[-n_features:][::-1]
        top_feats = [
            (profiles.columns[i], cluster_values.iloc[i])
            for i in top_indices
        ]
        top_features[cluster_name] = top_feats
    
    return top_features


if __name__ == "__main__":
    print("Utility functions module loaded successfully")
