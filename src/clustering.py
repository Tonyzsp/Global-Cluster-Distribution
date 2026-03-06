"""
Machine Learning clustering module for World Bank data.
Implements K-Means and Gaussian Mixture Model clustering with validation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


class ClusteringModel:
    """Base class for clustering models."""
    
    def __init__(self, data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize clustering model.
        
        Args:
            data: Input data (countries x features)
            test_size: Fraction for validation set
            random_state: Random seed for reproducibility
        """
        self.data = data
        self.test_size = test_size
        self.random_state = random_state
        
        # Split data into train and validation
        self.train_data, self.val_data = train_test_split(
            data, 
            test_size=test_size, 
            random_state=random_state
        )
        
        self.model = None
        self.train_labels = None
        self.val_labels = None
        self.metrics = {}
    
    def calculate_silhouette(self, data: pd.DataFrame, labels: np.ndarray) -> float:
        """Calculate silhouette score."""
        if len(np.unique(labels)) < 2:
            return -1.0
        return silhouette_score(data, labels)
    
    def get_stability(self) -> Dict:
        """Calculate stability metrics between train and validation sets."""
        train_silhouette = self.calculate_silhouette(self.train_data, self.train_labels)
        val_silhouette = self.calculate_silhouette(self.val_data, self.val_labels)
        
        return {
            'train_silhouette': train_silhouette,
            'val_silhouette': val_silhouette,
            'silhouette_diff': abs(train_silhouette - val_silhouette)
        }


class KMeansClustering(ClusteringModel):
    """K-Means clustering implementation."""
    
    def fit(self, n_clusters: int) -> Dict:
        """
        Fit K-Means model.
        
        Args:
            n_clusters: Number of clusters
        
        Returns:
            dict: Metrics including inertia and silhouette scores
        """
        print(f"Training K-Means with k={n_clusters}...")
        
        # Fit on training data
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
            verbose=0
        )
        self.train_labels = self.model.fit_predict(self.train_data)
        
        # Predict on validation data
        self.val_labels = self.model.predict(self.val_data)
        
        # Calculate metrics
        train_silhouette = self.calculate_silhouette(self.train_data, self.train_labels)
        val_silhouette = self.calculate_silhouette(self.val_data, self.val_labels)
        inertia = self.model.inertia_
        train_inertia = self.model.inertia_
        
        # Predict on full data for reporting
        full_labels = self.model.predict(self.data)
        full_silhouette = self.calculate_silhouette(self.data, full_labels)
        
        self.metrics = {
            'algorithm': 'K-Means',
            'n_clusters': n_clusters,
            'inertia': inertia,
            'train_silhouette': train_silhouette,
            'val_silhouette': val_silhouette,
            'full_silhouette': full_silhouette,
            'silhouette_diff': abs(train_silhouette - val_silhouette),
            'n_samples': len(self.data),
            'n_features': self.data.shape[1]
        }
        
        return self.metrics
    
    def predict_full(self) -> np.ndarray:
        """Get cluster labels for all data."""
        return self.model.predict(self.data)


class GMMClustering(ClusteringModel):
    """Gaussian Mixture Model clustering implementation."""
    
    def fit(self, n_components: int) -> Dict:
        """
        Fit GMM model.
        
        Args:
            n_components: Number of mixture components
        
        Returns:
            dict: Metrics including BIC and silhouette scores
        """
        print(f"Training GMM with {n_components} components...")

        # Default covariance type
        cov_type = getattr(self, 'covariance_type', 'full')
        uniform_prior = getattr(self, 'uniform_prior', False)

        # Prepare weights_init if uniform prior requested
        weights_init = None
        if uniform_prior:
            weights_init = np.ones(n_components) / float(n_components)

        # Fit on training data
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=cov_type,
            weights_init=weights_init,
            random_state=self.random_state,
            n_init=10,
            verbose=0
        )
        self.model.fit(self.train_data)
        self.train_labels = self.model.predict(self.train_data)
        
        # Predict on validation data
        self.val_labels = self.model.predict(self.val_data)
        
        # Calculate metrics
        train_silhouette = self.calculate_silhouette(self.train_data, self.train_labels)
        val_silhouette = self.calculate_silhouette(self.val_data, self.val_labels)
        bic = self.model.bic(self.data)
        
        # Predict on full data for reporting
        full_labels = self.model.predict(self.data)
        full_silhouette = self.calculate_silhouette(self.data, full_labels)
        
        # Get soft probabilities
        train_proba = self.model.predict_proba(self.train_data)
        val_proba = self.model.predict_proba(self.val_data)
        
        # Calculate average probability (confidence)
        avg_train_confidence = np.mean(np.max(train_proba, axis=1))
        avg_val_confidence = np.mean(np.max(val_proba, axis=1))
        
        self.metrics = {
            'algorithm': 'GMM',
            'n_components': n_components,
            'covariance_type': cov_type,
            'uniform_prior': uniform_prior,
            'bic': bic,
            'train_silhouette': train_silhouette,
            'val_silhouette': val_silhouette,
            'full_silhouette': full_silhouette,
            'silhouette_diff': abs(train_silhouette - val_silhouette),
            'train_confidence': avg_train_confidence,
            'val_confidence': avg_val_confidence,
            'n_samples': len(self.data),
            'n_features': self.data.shape[1]
        }
        
        return self.metrics
    
    def predict_full(self) -> np.ndarray:
        """Get cluster labels for all data."""
        return self.model.predict(self.data)
    
    def predict_proba_full(self) -> np.ndarray:
        """Get soft probabilities for all data."""
        return self.model.predict_proba(self.data)


def find_optimal_clusters(
    data: pd.DataFrame,
    algorithm: str = 'kmeans',
    k_range: range = range(2, 11),
    test_size: float = 0.3
) -> Dict[int, Dict]:
    """
    Find optimal number of clusters using elbow method and silhouette scores.
    
    Args:
        data: Input data
        algorithm: 'kmeans' or 'gmm'
        k_range: Range of cluster numbers to test
        test_size: Validation set fraction
    
    Returns:
        dict: {n_clusters: metrics_dict}
    """
    print(f"\nFinding optimal clusters for {algorithm.upper()}...")
    results = {}
    
    for k in k_range:
        print(f"  Testing k={k}...", end=' ')
        
        try:
            if algorithm.lower() == 'kmeans':
                model = KMeansClustering(data, test_size=test_size)
                metrics = model.fit(k)
            elif algorithm.lower() == 'gmm':
                model = GMMClustering(data, test_size=test_size)
                metrics = model.fit(k)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            results[k] = metrics
            print(f"Silhouette: {metrics['full_silhouette']:.3f}")
        
        except Exception as e:
            print(f"Error: {str(e)[:30]}")
            continue
    
    return results


def create_cluster_profiles(
    data: pd.DataFrame,
    labels: np.ndarray,
    cluster_names: Optional[Dict[int, str]] = None
) -> pd.DataFrame:
    """
    Create cluster profiles showing mean feature values per cluster.
    
    Args:
        data: Original (unscaled) data or scaled for analysis
        labels: Cluster assignments
        cluster_names: Optional mapping of cluster ID to custom name
    
    Returns:
        pd.DataFrame: Cluster profiles
    """
    data_with_labels = data.copy()
    data_with_labels['__cluster__'] = labels
    
    profiles = data_with_labels.groupby('__cluster__').mean()
    
    if cluster_names:
        profiles.index = profiles.index.map(
            lambda x: cluster_names.get(x, f'Cluster {x}')
        )
    else:
        profiles.index = [f'Cluster {int(i)}' for i in profiles.index]
    
    return profiles


def get_feature_importance_per_cluster(
    data: pd.DataFrame,
    labels: np.ndarray
) -> Dict[int, pd.Series]:
    """
    Get feature importance (between-cluster variance) per cluster.
    
    Args:
        data: Input data
        labels: Cluster assignments
    
    Returns:
        dict: {cluster_id: importance_series}
    """
    importance_dict = {}
    
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_data = data[cluster_mask]
        
        # Variance within cluster for each feature
        feature_var = cluster_data.var()
        importance_dict[cluster_id] = feature_var
    
    return importance_dict


def print_metrics_summary(metrics: Dict) -> None:
    """Pretty print clustering metrics."""
    print("\n" + "="*60)
    print(f"CLUSTERING RESULTS: {metrics['algorithm']}")
    print("="*60)
    
    if metrics['algorithm'] == 'K-Means':
        print(f"Clusters:          {metrics['n_clusters']}")
        print(f"Inertia:           {metrics['inertia']:.2f}")
    else:  # GMM
        print(f"Components:        {metrics['n_components']}")
        print(f"BIC:               {metrics['bic']:.2f}")
        if 'train_confidence' in metrics:
            print(f"Train Confidence:  {metrics['train_confidence']:.4f}")
            print(f"Val Confidence:    {metrics['val_confidence']:.4f}")
    
    print(f"\nSilhouette Scores:")
    print(f"  Training:        {metrics['train_silhouette']:.4f}")
    print(f"  Validation:      {metrics['val_silhouette']:.4f}")
    print(f"  Full Dataset:    {metrics['full_silhouette']:.4f}")
    print(f"  Difference:      {metrics['silhouette_diff']:.4f}")
    
    print(f"\nData Info:")
    print(f"  Samples:         {metrics['n_samples']}")
    print(f"  Features:        {metrics['n_features']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    from data_processor import process_data_pipeline
    
    # Process data
    df, scaler, stats = process_data_pipeline(
        n_neighbors=5,
        apply_imputation=True,
        apply_scaling=True
    )
    
    print("\n" + "="*60)
    print("CLUSTERING EXAMPLE")
    print("="*60 + "\n")
    
    # Find optimal K for K-Means
    kmeans_results = find_optimal_clusters(df, algorithm='kmeans', k_range=range(2, 11))
    
    # Find optimal K for GMM
    gmm_results = find_optimal_clusters(df, algorithm='gmm', k_range=range(2, 11))
    
    # Fit best K-Means model (k=3)
    print("\nFitting final K-Means model with k=3...")
    kmeans = KMeansClustering(df, test_size=0.2)
    kmeans_metrics = kmeans.fit(3)
    print_metrics_summary(kmeans_metrics)
    
    # Fit best GMM model (3 components)
    print("\nFitting final GMM model with 3 components...")
    gmm = GMMClustering(df, test_size=0.2)
    gmm_metrics = gmm.fit(3)
    print_metrics_summary(gmm_metrics)
    
    # Get cluster profiles
    kmeans_labels = kmeans.predict_full()
    profiles = create_cluster_profiles(df, kmeans_labels)
    print("\nK-Means Cluster Profiles:")
    print(profiles)
