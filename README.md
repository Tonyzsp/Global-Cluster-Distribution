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
- **Imputation**: Uses KNN (k=5) to estimate missing values based on similar countries
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
- **Train/Validation Split**: 70% training, 30% validation by default (adjustable)
- **Silhouette Score**: Measures cluster separation and cohesion (-1 to 1)
- **Stability Check**: Ensures consistent predictions on validation set
- **Inertia/BIC**: Helps identify optimal cluster count via elbow method

## Sidebar Controls

```
┌─ INDICATOR SELECTION (15-20 Curated)
│  ├─ □ GDP per Capita
│  ├─ □ Life Expectancy
│  ├─ □ Population Growth
│  ├─ □ Literacy Rate
│  ├─ □ Healthcare Spending (% GDP)
│  ├─ □ CO2 Emissions per Capita
│  ├─ ... (+ 9-14 more indicators)
│  ├─ [Select All] [Select Defaults] [Reset]
│  └─ Data Availability: 85-95%
│
├─ COUNTRY FILTERING (Optional)
│  ├─ [ All Countries ] (195+ countries)
│  ├─ Filter by Region/Income Level
│  └─ [Select All] [Deselect All]
│
├─ TIME PERIOD
│  ├─ Year Range: [2010 ─────→ 2024]
│  └─ ☑ Use Latest Available Year
│
├─ ALGORITHM
│  ├─ ○ K-Means
│  └─ ○ GMM
│
├─ HYPERPARAMETERS
│  ├─ Clusters/Components: [2────────10]
│  ├─ KNN n_neighbors: [3───────────10]
│  ├─ Train/Val Split: [60%─────────90%]
│  └─ ☑ Apply KNN Imputation
│
└─ [RUN CLUSTERING]
```

## Main Panel Output

1. **Data Overview**: Missing data statistics and summaries
2. **World Map**: Interactive choropleth showing cluster assignments
3. **Metrics Dashboard**: Silhouette scores, inertia/BIC, stability metrics
4. **Cluster Profiles**: Table and visualizations of cluster characteristics
5. **Download Section**: Export results as CSV

## Curated World Bank Indicators (15-20)

The app uses `wbgapi` to fetch a curated set of important development indicators:

| Category | Indicator | Code | Description |
|----------|-----------|------|-------------|
| **Economic** | GDP per Capita | NY.GDP.PCAP.CD | Annual GDP per person (USD) |
| | GNI per Capita | NY.GNP.PCAP.CD | Annual GNI per person (USD) |
| | Trade (% GDP) | NE.TRD.GNFS.CD | Imports + Exports as % of GDP |
| | Inflation | FP.CPI.TOTL.ZG | Annual inflation rate (%) |
| **Demographic** | Life Expectancy | SP.DYN.LE00.IN | Average years of life expected |
| | Population Growth | SP.POP.GROW | Annual % population change |
| | Fertility Rate | SP.DYN.TFRT.IN | Births per woman |
| | Urban Population | SP.URB.TOTL.IN.ZS | % of population in urban areas |
| **Education** | Literacy Rate | SE.ADT.LITR.ZS | % of population 15+ who can read/write |
| | Primary Enrollment | SE.PRM.ENRR | Gross primary school enrollment ratio |
| | Secondary Enrollment | SE.SEC.ENRR | Gross secondary school enrollment ratio |
| | Tertiary Enrollment | SE.TER.ENRR | Gross tertiary school enrollment ratio |
| **Health** | Infant Mortality | SP.DYN.IMRT.IN | Deaths per 1,000 live births under age 5 |
| | Under-5 Mortality | SP.DYN.CDRT.IN | Deaths per 1,000 children under age 5 |
| | Health Spending | SH.XPD.CHEX.GD.ZP | Healthcare expenditure (% of GDP) |
| **Environment** | CO2 Emissions | EN.ATM.CO2E.PC | Metric tons of CO2 per capita |
| | Forest Area | AG.LND.FRST.ZS | Forest as % of land area |
| | Electricity Access | EG.ELC.ACCS.ZS | % population with access to electricity |

All indicators are fetched for **all available countries** (195+) using the latest available year per indicator.

## Configuration

### Environment Variables (Optional)
None required; all configuration via Streamlit UI.

### Streamlit Config (`.streamlit/config.toml`)
See [PLAN.md](./PLAN.md) for optional theme customization.

## Dependencies

- **streamlit** (^1.28.0) - Web UI framework and deployment
- **wbgapi** (^1.0.12) - World Bank Data API
- **pandas** (^2.1.0) - Data manipulation and analysis
- **numpy** (^1.24.0) - Numerical computing
- **scikit-learn** (^1.3.0) - ML algorithms and preprocessing
- **plotly** (^5.17.0) - Interactive visualizations
- **scipy** (^1.11.0) - Scientific computing and statistics

## Troubleshooting

### "ModuleNotFoundError" when running app
- Ensure virtual environment is activated: `.venv\Scripts\activate`
- Reinstall dependencies: `pip install -r requirements.txt`

### Missing World Bank data
- Some countries/indicators may lack historical data
- App automatically filters and warns about exclusions
- Check data availability in [World Bank Open Data](https://data.worldbank.org)

### Map not rendering
- Ensure countries have valid ISO-3 codes
- Check browser console for errors (F12)
- Try clearing Streamlit cache: `streamlit cache clear`

### Slow data loading
- First run may take 30-60s to fetch World Bank data
- Results are cached locally by default
- Subsequent runs are much faster

## Example Clusters

When running with default settings, you might discover clusters like:

- **Industrialized**: High GDP, Life Expectancy, low population growth
- **Emerging Markets**: Growing GDP, improving health, rising education
- **Developing**: Low GDP, high population growth, lower education
- **Small Island Nations**: Unique geographic and economic profiles
- **OPEC Producers**: Resource-dependent economies

## Development Notes

For detailed implementation plan and enhancement ideas, see [PLAN.md](./PLAN.md).

### Key Design Decisions

1. **KNN Imputation** over other methods: Captures inter-country relationships
2. **Train/Validation Split** over cross-validation: Simpler UI interpretation
3. **Plotly Choropleth** for mapping: Native country boundaries, rich interactivity
4. **StandardScaler after imputation**: Prevents data leakage in distance calculations
5. **Sidebar controls**: Keeps main panel focused on results visualization

## License

MIT License - See LICENSE file for details.

## Contributing

Suggestions for improvements:
- Additional indicators or data sources
- Time-series clustering (year selection)
- PCA visualization
- Regional filtering
- Custom cluster labeling UI

## Resources

- [World Bank Open Data API](https://datahelpdesk.worldbank.org/knowledgebase/articles/889386)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Plotly Choropleth Maps](https://plotly.com/python/choropleth-maps/)

## Known Issues & Troubleshooting

### World Bank API Errors
- Some indicators (e.g., NE.TRD.GNFS.CD, SH.XPD.CHEX.GD.ZP, EN.ATM.CO2E.PC) may return `APIError: JSON decoding error` if no data is available. The app will skip these and continue, but you may see warnings in the logs.

### Indicator Selection & UI Freezing
- If you add/remove indicators or change hyperparameters, the app will not update results until you click **Run Clustering**. This is by design: all outputs (map, profiles, metrics, overview) are frozen to the last successful run.
- If you clear all indicators but have previous output, the Data Overview and other sections will continue to show the last results.

### KeyError or Data Overview Not Working
- If you select indicators that are not available in the loaded data, the app will show a friendly message and not error. Only indicators present in the data can be used.

### Indentation or Streamlit Errors
- If you see `IndentationError` or Streamlit duplicate element errors, check for correct indentation and ensure only one 'Run Clustering' button is present in the sidebar or main panel.

### Download Button
- The download button for CSV export only appears if data is available from the last run. If you clear all indicators or haven't run clustering, it will not show.

---

**For more troubleshooting and design notes, see PLAN.md.**

---

**Last Updated**: March 5, 2026
