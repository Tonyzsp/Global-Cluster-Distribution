# Global Cluster Distribution

Interactive Streamlit application for clustering countries based on World Bank economic and demographic indicators using K-Means and Gaussian Mixture Models (GMM). Features interactive world maps, intelligent missing data imputation via KNN, and comprehensive cluster analysis.

## Features

- **Intelligent Data Imputation**: Uses K-Nearest Neighbors (KNN) to handle missing values while preserving data relationships
- **Dual Clustering Methods**: 
  - **K-Means**: Hard clustering with spherical boundaries
  - **GMM**: Soft, probabilistic clustering with component-based approach
- **Interactive Visualizations**:
  - Plotly Choropleth world map colored by cluster assignment
  - Cluster profile tables showing mean feature values per group
  - Elbow/BIC curves for optimal cluster selection
  - Radar charts for cluster archetypes
- **Comprehensive Validation**:
  - Silhouette scores (training & validation sets)
  - Inertia (K-Means) and BIC (GMM) metrics
  - Stability checks on holdout validation data
- **Flexible Feature Selection**: Choose from 8+ World Bank indicators across economic, demographic, and environmental categories
- **Downloadable Results**: Export cluster assignments and profiles as CSV files

## Quick Start

### Prerequisites
- Python 3.9+
- pip (Python package manager)

### Installation

1. **Clone & Navigate**:
   ```bash
   cd Global-Cluster-Distribution
   ```

2. **Create Virtual Environment** (Windows):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Application**:
   ```bash
   streamlit run app.py
   ```

5. **Open Browser**:
   Navigate to `http://localhost:8501`

## Project Structure

```
Global-Cluster-Distribution/
├── src/
│   ├── data_processor.py      # World Bank data extraction & cleaning
│   ├── clustering.py           # K-Means & GMM implementations
│   └── utils.py                # Helper functions
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── PLAN.md                     # Detailed implementation plan
├── README.md                   # This file
└── .gitignore                  # Git ignore rules
```

## How It Works

### 1. Data Pipeline
- **Extraction**: Fetches curated 15-20 world bank indicators (GDP, Life Expectancy, Population Growth, Literacy, Healthcare, CO2 Emissions, etc.) for ALL countries
- **Coverage**: Includes all 195+ countries in World Bank database with ISO-3 codes
- **Time Series**: Uses latest available year per indicator (varies by country/indicator)
- **Imputation**: Uses KNN to estimate missing values based on similar countries
- **Scaling**: Applies StandardScaler to normalize features (mean=0, std=1)

### 2. Clustering Methods

**K-Means Clustering**
- Partitions countries into k spherical clusters
- Minimizes within-cluster variance (inertia)
- Provides hard assignments (each country to exactly one cluster)
- Best for: Clear, distinct groups

**Gaussian Mixture Model (GMM)**
- Models clusters as probabilistic distributions
- Provides soft assignments (probability of belonging to each component)
- Uses Expectation-Maximization algorithm
- Best for: Overlapping or hierarchical structure

### 3. Validation Strategy
- **Train/Validation Split**: 70% training, 30% validation 
- **Silhouette Score**: Measures cluster separation and cohesion (-1 to 1)
- **Stability Check**: Ensures consistent predictions on validation set
- **Inertia/BIC**: Helps identify optimal cluster count via elbow method

## Usage: Sidebar Controls

- **Indicator Selection**: Multi-select from curated World Bank indicators (GDP, Life Expectancy, etc.)
- **KNN Neighbors**: Choose number of neighbors for missing data imputation
- **Algorithm**: Select K-Means or GMM (Gaussian Mixture Model)
- **Clusters/Components**: Set number of clusters/components (2–10)
- **GMM Covariance Type**: (If GMM selected) Choose covariance type
- **[Run Clustering]**: Click to update all results. All outputs remain frozen until you click this button.


## Main Panel Output

1. **Data Overview**: Missing data statistics and summaries
2. **World Map**: Interactive choropleth showing cluster assignments
3. **Metrics Dashboard**: Silhouette scores, inertia/BIC, stability metrics
4. **Cluster Profiles**: Table and visualizations of cluster characteristics
5. **Download Section**: Export results as CSV

## Troubleshooting

### "ModuleNotFoundError" when running app
- Ensure virtual environment is activated: `.venv\Scripts\activate`
- Reinstall dependencies: `pip install -r requirements.txt`

### Missing World Bank data
- Some countries/indicators may lack historical data
- App automatically filters and warns about exclusions
- Check data availability in [World Bank Open Data](https://data.worldbank.org)


## License

MIT License - See LICENSE file for details.


## Resources

- [World Bank Open Data API](https://datahelpdesk.worldbank.org/knowledgebase/articles/889386)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Plotly Choropleth Maps](https://plotly.com/python/choropleth-maps/)
