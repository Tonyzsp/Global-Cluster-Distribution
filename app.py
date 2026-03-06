"""
Global Cluster Distribution - Interactive World Bank Data Clustering Application
Streamlit-based UI for K-Means and GMM clustering of countries
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import fetch_indicators, CURATED_INDICATORS, clean_data, impute_missing, scale_features
from clustering import KMeansClustering, GMMClustering, create_cluster_profiles
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Global Cluster Distribution",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Country name mapping (World Bank codes to full names)
COUNTRY_NAMES = {
    'ABW': 'Aruba', 'AFE': 'Africa (exclude high income)', 'AFG': 'Afghanistan', 'AFW': 'Africa (West)', 'AGO': 'Angola',
    'ALB': 'Albania', 'AND': 'Andorra', 'ARE': 'United Arab Emirates', 'ARG': 'Argentina', 'ARM': 'Armenia',
    'AUS': 'Australia', 'AUT': 'Austria', 'AZE': 'Azerbaijan', 'BDI': 'Burundi', 'BEL': 'Belgium',
    'BEN': 'Benin', 'BFA': 'Burkina Faso', 'BGD': 'Bangladesh', 'BGR': 'Bulgaria', 'BHR': 'Bahrain', 'BHS': 'Bahamas',
    'BIH': 'Bosnia and Herzegovina', 'BLR': 'Belarus', 'BLZ': 'Belize', 'BMU': 'Bermuda', 'BOL': 'Bolivia', 'BRA': 'Brazil',
    'BRB': 'Barbados', 'BRN': 'Brunei', 'BTN': 'Bhutan', 'BWA': 'Botswana', 'CAF': 'Central African Republic',
    'CAN': 'Canada', 'CHE': 'Switzerland', 'CHL': 'Chile', 'CHN': 'China', 'CIV': "Côte d'Ivoire",
    'CMR': 'Cameroon', 'COD': 'Congo Dem. Rep.', 'COG': 'Congo', 'COL': 'Colombia', 'COM': 'Comoros',
    'CPV': 'Cape Verde', 'CRI': 'Costa Rica', 'CUB': 'Cuba', 'CUW': 'Curaçao', 'CYM': 'Cayman Islands',
    'CYP': 'Cyprus', 'CZE': 'Czech Republic', 'DEU': 'Germany', 'DJI': 'Djibouti', 'DMA': 'Dominica',
    'DNK': 'Denmark', 'DOM': 'Dominican Republic', 'DZA': 'Algeria', 'ECU': 'Ecuador', 'EGY': 'Egypt',
    'ERI': 'Eritrea', 'ESP': 'Spain', 'EST': 'Estonia', 'ETH': 'Ethiopia', 'FIN': 'Finland',
    'FJI': 'Fiji', 'FRA': 'France', 'FRO': 'Faroe Islands', 'FSM': 'Micronesia', 'GAB': 'Gabon',
    'GBR': 'United Kingdom', 'GEO': 'Georgia', 'GHA': 'Ghana', 'GIB': 'Gibraltar', 'GIN': 'Guinea',
    'GMB': 'Gambia', 'GNB': 'Guinea-Bissau', 'GNQ': 'Equatorial Guinea', 'GRC': 'Greece', 'GRD': 'Grenada',
    'GRL': 'Greenland', 'GTM': 'Guatemala', 'GUY': 'Guyana', 'HIC': 'High income', 'HKG': 'Hong Kong',
    'HND': 'Honduras', 'HRV': 'Croatia', 'HTI': 'Haiti', 'HUN': 'Hungary', 'IDN': 'Indonesia',
    'IND': 'India', 'IRL': 'Ireland', 'IRN': 'Iran', 'IRQ': 'Iraq', 'ISL': 'Iceland',
    'ISR': 'Israel', 'ITA': 'Italy', 'JAM': 'Jamaica', 'JOR': 'Jordan', 'JPN': 'Japan',
    'KAZ': 'Kazakhstan', 'KEN': 'Kenya', 'KGZ': 'Kyrgyzstan', 'KHM': 'Cambodia', 'KIR': 'Kiribati',
    'KNA': 'St. Kitts and Nevis', 'KOR': 'Korea', 'KWT': 'Kuwait', 'LAC': 'Latin America & Caribbean', 'LAO': 'Lao',
    'LBN': 'Lebanon', 'LBR': 'Liberia', 'LBY': 'Libya', 'LCA': 'St. Lucia', 'LDC': 'Least developed countries',
    'LIC': 'Low income', 'LIE': 'Liechtenstein', 'LKA': 'Sri Lanka', 'LMC': 'Lower middle income', 'LSO': 'Lesotho',
    'LTU': 'Lithuania', 'LUX': 'Luxembourg', 'LVA': 'Latvia', 'MAC': 'Macao', 'MAR': 'Morocco',
    'MCO': 'Monaco', 'MDA': 'Moldova', 'MDG': 'Madagascar', 'MDV': 'Maldives', 'MEA': 'Middle East & North Africa',
    'MEX': 'Mexico', 'MHL': 'Marshall Islands', 'MKD': 'North Macedonia', 'MLI': 'Mali', 'MLT': 'Malta',
    'MMR': 'Myanmar', 'MNE': 'Montenegro', 'MNG': 'Mongolia', 'MOZ': 'Mozambique', 'MRT': 'Mauritania',
    'MUS': 'Mauritius', 'MWI': 'Malawi', 'MYS': 'Malaysia', 'NAM': 'Namibia', 'NER': 'Niger',
    'NGA': 'Nigeria', 'NIC': 'Nicaragua', 'NLD': 'Netherlands', 'NOR': 'Norway', 'NPL': 'Nepal',
    'NRU': 'Nauru', 'NZL': 'New Zealand', 'OED': 'OECD members', 'OMN': 'Oman', 'PAK': 'Pakistan',
    'PAN': 'Panama', 'PER': 'Peru', 'PHL': 'Philippines', 'PLW': 'Palau', 'PNG': 'Papua New Guinea',
    'POL': 'Poland', 'PRI': 'Puerto Rico', 'PRK': 'North Korea', 'PRT': 'Portugal', 'PRY': 'Paraguay',
    'PSE': 'West Bank and Gaza', 'QAT': 'Qatar', 'ROU': 'Romania', 'RUS': 'Russian Federation', 'RWA': 'Rwanda',
    'SAU': 'Saudi Arabia', 'SDN': 'Sudan', 'SEN': 'Senegal', 'SGP': 'Singapore', 'SHN': 'St. Helena',
    'SLB': 'Solomon Islands', 'SLE': 'Sierra Leone', 'SLV': 'El Salvador', 'SMR': 'San Marino', 'SOM': 'Somalia',
    'SRB': 'Serbia', 'SSD': 'South Sudan', 'STP': 'São Tomé and Príncipe', 'SUR': 'Suriname', 'SVK': 'Slovak Republic',
    'SVN': 'Slovenia', 'SWE': 'Sweden', 'SWZ': 'Eswatini', 'SYC': 'Seychelles', 'SYR': 'Syrian Arab Republic',
    'TCA': 'Turks and Caicos Islands', 'TCD': 'Chad', 'TGO': 'Togo', 'THA': 'Thailand', 'TJK': 'Tajikistan',
    'TKL': 'Tokelau', 'TKM': 'Turkmenistan', 'TLS': 'Timor-Leste', 'TON': 'Tonga', 'TTO': 'Trinidad and Tobago',
    'TUN': 'Tunisia', 'TUR': 'Turkey', 'TUV': 'Tuvalu', 'TWN': 'Taiwan', 'TZA': 'Tanzania',
    'UGA': 'Uganda', 'UKR': 'Ukraine', 'UMC': 'Upper middle income', 'USA': 'United States', 'URY': 'Uruguay',
    'UZB': 'Uzbekistan', 'VCT': 'St. Vincent and the Grenadines', 'VEN': 'Venezuela', 'VGB': 'British Virgin Islands',
    'VIR': 'Virgin Islands (U.S.)', 'VNM': 'Vietnam', 'VUT': 'Vanuatu', 'WLD': 'World', 'WSM': 'Samoa',
    'YEM': 'Yemen', 'ZAF': 'South Africa', 'ZMB': 'Zambia', 'ZWE': 'Zimbabwe'
}


@st.cache_resource
def load_data_cached(knn_neighbors: int = 5):
    """Load and cache processed data using internal defaults (metric=nan_euclidean).

    Args:
        knn_neighbors: number of neighbors for KNN imputation (cached key)
    """
    with st.spinner("⏳ Loading World Bank data..."):
        try:
            # Fetch raw data
            df_raw = fetch_indicators(use_curated=True)

            # Clean: remove rows with >50% missing
            df_raw = df_raw.dropna(how='all')  # Remove all-NaN rows
            df_raw = df_raw[df_raw.isna().sum(axis=1) / df_raw.shape[1] < 0.5]  # Keep rows with <50% missing

            # Impute missing values with internal defaults (nan_euclidean)
            df_imputed = impute_missing(df_raw, n_neighbors=knn_neighbors, method='knn', metric='nan_euclidean')

            # Scale
            df_scaled, scaler = scale_features(df_imputed, fit=True)

            return df_raw, df_scaled, scaler
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return None, None, None


def get_country_name(code: str) -> str:
    """Get full country name from code."""
    return COUNTRY_NAMES.get(code, code)


def main():
    st.markdown("# 🌍 Global Cluster Distribution")
    st.markdown("_Interactive clustering of World Bank indicators across countries_")
    
    # ============= SIDEBAR - Configuration =============
    st.sidebar.markdown("## 🎛️ Configuration")
    
    # Data processing
    st.sidebar.markdown("### Data Processing")
    knn_neighbors = st.sidebar.slider("KNN Neighbors (for imputation)", min_value=2, max_value=15, value=5, step=1, help="Number of neighbors used by KNN imputer")

    # Load data (uses nan_euclidean metric internally)
    df_raw, df_scaled, scaler = load_data_cached(knn_neighbors=knn_neighbors)
    
    if df_scaled is None:
        return
    
    # Data info
        st.markdown(f"**Data:** {len(df_raw)} countries/regions × {df_raw.shape[1]} indicators | **Processing:** KNN Imputation (k={knn_neighbors}, metric=nan_euclidean) + StandardScaler")
    
    # Indicator selection (simple multiselect)
    st.sidebar.markdown("### Indicators")
    available_indicators = list(df_raw.columns)
    selected_indicators = st.sidebar.multiselect(
        "Select indicators to cluster",
        available_indicators,
        default=[],
        help="Start with no indicators selected — choose indicators to enable clustering"
    )
    
    if not selected_indicators:
        st.info("Select at least one indicator to enable previews and clustering.")
        return

    # Only keep indicators that actually exist in the loaded data (guard against API-skipped series)
    valid_selected = [ind for ind in selected_indicators if ind in df_scaled.columns]
    if not valid_selected:
        st.info("The selected indicators are not available in the loaded data. Please choose different indicators.")
        return

    df_selected = df_scaled[valid_selected].copy()
    df_raw_selected = df_raw[valid_selected].copy()
    # overwrite selected_indicators variable to the valid subset for downstream use
    selected_indicators = valid_selected
    # DISPLAY VARIABLES: keep UI frozen to last run results until user clicks Run
    if 'last_labels' in st.session_state:
        display_df_selected = st.session_state.df_selected
        display_df_raw_selected = st.session_state.df_raw_selected
        display_selected_indicators = st.session_state.selected_indicators
    else:
        display_df_selected = df_selected
        display_df_raw_selected = df_raw_selected
        display_selected_indicators = list(selected_indicators)
    
    # Algorithm
    st.sidebar.markdown("### Algorithm")
    algorithm = st.sidebar.radio("Clustering method", ["K-Means", "GMM"], index=0)
    
    # Hyperparameters
    st.sidebar.markdown("### Hyperparameters")
    n_clusters = st.sidebar.slider("Clusters/Components", min_value=2, max_value=10, value=3, step=1)
    
    # GMM options
    cov_type = 'full'
    if algorithm == 'GMM':
        cov_type = st.sidebar.selectbox("Covariance Type", options=['full', 'tied', 'diag', 'spherical'], index=0)
    
    # Run button is rendered in the map tab to ensure results update only when pressed
    
    # ============= MAIN PANEL =============
    
    # Data stats (show frozen/last-run view until Run is clicked)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Countries/Regions", len(display_df_selected))
    with col2:
        st.metric("Indicators Selected", len(display_selected_indicators))
    with col3:
        try:
            missing_pct = (display_df_raw_selected.isna().sum().sum() / (len(display_df_raw_selected) * len(display_selected_indicators))) * 100
        except Exception:
            missing_pct = 0.0
        st.metric("Original Missing Data %", f"{missing_pct:.1f}%")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Clustering Map", "📊 Cluster Profiles", "📈 Metrics", "🔍 Data Overview"])
    
    # ============= TAB 1: Clustering Map =============
    with tab1:
        def _render_results(df_sel, df_raw_sel, labels, algorithm_name, n_clusters_val, selected_inds):
            """Render map and cluster distribution from stored results without using current config."""
            # Build map_data
            map_data = pd.DataFrame({
                'Country_Code': df_sel.index,
                'Full_Name': df_sel.index.map(get_country_name),
                'Cluster': labels
            })
            for col in selected_inds:
                # use raw values for hover
                map_data[col] = df_raw_sel[col].values

            map_data['Cluster_str'] = map_data['Cluster'].astype(str)
            category_order = [str(i) for i in range(n_clusters_val)]

            fig = px.choropleth(
                map_data,
                locations='Country_Code',
                color='Cluster_str',
                hover_name='Full_Name',
                hover_data=selected_inds,
                title=f"{algorithm_name} Clustering ({n_clusters_val} clusters)",
                color_discrete_sequence=px.colors.qualitative.Plotly,
                category_orders={'Cluster_str': category_order}
            )
            fig.update_geos(fitbounds="locations", projection_type="natural earth")
            fig.update_layout(title=f"{algorithm_name} Clustering ({n_clusters_val} clusters)", height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Cluster distribution (bar + size list). Do not show simple table.
            cluster_counts = pd.Series(labels).value_counts().sort_index()
            col1, col2 = st.columns([2, 1])
            with col1:
                fig_dist = px.bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    labels={'x': 'Cluster', 'y': 'Number of Countries/Regions'},
                    title="Cluster Distribution"
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            with col2:
                st.markdown("### Cluster Sizes")
                for i in range(n_clusters_val):
                    count = int(cluster_counts.get(i, 0))
                    st.write(f"**Cluster {i}**: {count}")

        # If user clicked Run, perform clustering and store results
        if st.sidebar.button("🚀 Run Clustering", use_container_width=True):
            st.session_state.run_clustering = True

        if 'run_clustering' in st.session_state and st.session_state.run_clustering:
            try:
                with st.spinner(f"⏳ Training {algorithm}..."):
                    # Fit model with internal train/val split (80/20)
                    if algorithm == "K-Means":
                        model = KMeansClustering(df_selected, test_size=0.2)
                        metrics = model.fit(n_clusters)
                    else:
                        model = GMMClustering(df_selected, test_size=0.2)
                        model.covariance_type = cov_type
                        model.uniform_prior = True
                        metrics = model.fit(n_clusters)

                    labels = model.predict_full()
                    # normalize labels to 0..k-1 ordering
                    unique_labels = np.unique(labels)
                    label_mapping = {old: new for new, old in enumerate(sorted(unique_labels))}
                    labels_mapped = np.array([label_mapping[l] for l in labels])
                    labels = labels_mapped

                    # store copies in session state so UI doesn't change until next Run
                    st.session_state.last_model = model
                    st.session_state.last_metrics = metrics
                    st.session_state.last_labels = labels
                    st.session_state.df_selected = df_selected.copy()
                    st.session_state.df_raw_selected = df_raw_selected.copy()
                    st.session_state.selected_indicators = list(selected_indicators)
                    st.session_state.n_clusters = n_clusters
                    st.session_state.run_clustering = False

                    # render results from stored state
                    _render_results(st.session_state.df_selected, st.session_state.df_raw_selected, st.session_state.last_labels, algorithm, st.session_state.n_clusters, st.session_state.selected_indicators)
            except Exception as e:
                st.error(f"❌ Clustering error: {str(e)}")
        elif 'last_labels' in st.session_state:
            # render previously computed results without reacting to current controls
            _render_results(st.session_state.df_selected, st.session_state.df_raw_selected, st.session_state.last_labels, st.session_state.last_metrics['algorithm'], st.session_state.n_clusters, st.session_state.selected_indicators)
        else:
            st.info("👈 Configure and click '🚀 Run Clustering' to start")
    
    # ============= TAB 2: Cluster Profiles =============
    with tab2:
        if 'last_model' in st.session_state:
            profiles = create_cluster_profiles(
                st.session_state.df_selected,
                st.session_state.last_labels
            )
            
            st.markdown("### Mean Feature Values per Cluster\n(Top identifying indicator highlighted)")

            # Compute top identifying feature per cluster using scaled data
            profiles_scaled = profiles  # profiles is computed from scaled data
            global_mean_scaled = st.session_state.df_selected.mean()
            # difference magnitude per feature
            diff = (profiles_scaled.sub(global_mean_scaled)).abs()
            top_features = diff.idxmax(axis=1)  # Series indexed by cluster name -> feature

            # Build display table using original units (raw data means)
            profiles_raw = create_cluster_profiles(st.session_state.df_raw_selected, st.session_state.last_labels)

            # Ensure indices align (both generated as 'Cluster {i}')
            def highlight_row_top(s):
                # s is a Series for a cluster (raw means)
                cluster_name = s.name
                top = top_features.get(cluster_name, None)
                styles = []
                for col in s.index:
                    if col == top:
                        styles.append('background-color: #ffd54f; font-weight: bold')
                    else:
                        styles.append('')
                return styles

            styled = profiles_raw.style.apply(highlight_row_top, axis=1)
            st.dataframe(styled, width=1200, height=300)

            st.markdown("**How the top identifying indicator is calculated:**")
            st.caption("For each cluster we compute the mean (on scaled features) and compare it to the global mean (scaled). The indicator with the largest absolute difference (cluster mean - global mean) is chosen as the top identifying indicator for that cluster.")

            # Allow user to pick a cluster and view the countries in that cluster
            cluster_options = profiles.index.tolist()
            chosen_cluster = st.selectbox("Select cluster to list member countries", options=cluster_options)
            # extract numeric id from 'Cluster X'
            try:
                chosen_id = int(str(chosen_cluster).split()[-1])
            except Exception:
                chosen_id = None

            if chosen_id is not None:
                # build a table of countries and raw values for this cluster
                labels = st.session_state.last_labels
                countries_idx = st.session_state.df_raw_selected.index
                cluster_map = pd.Series(labels, index=countries_idx, name='Cluster')
                df_members = st.session_state.df_raw_selected.copy()
                df_members['Cluster'] = cluster_map
                df_members = df_members[df_members['Cluster'] == chosen_id]
                if not df_members.empty:
                    # use the frozen indicators from last run to avoid KeyError when user changed selection
                    display = df_members[st.session_state.selected_indicators].copy()
                    display.index = display.index.map(get_country_name)
                    st.markdown(f"### Countries in {chosen_cluster}")
                    st.dataframe(display, width=1200, height=400)
                else:
                    st.info("No countries found for this cluster.")

        else:
            st.info("Run clustering first to see profiles")
    
    # ============= TAB 3: Metrics =============
    with tab3:
        if 'last_metrics' in st.session_state:
            metrics = st.session_state.last_metrics
            
            st.markdown(f"### {metrics['algorithm']} Validation Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Train Silhouette", f"{metrics['train_silhouette']:.4f}")
            with col2:
                st.metric("Validation Silhouette", f"{metrics['val_silhouette']:.4f}")
            with col3:
                st.metric("Full Dataset", f"{metrics['full_silhouette']:.4f}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Configuration")
                st.write(f"**Algorithm**: {metrics['algorithm']}")
                st.write(f"**Clusters**: {metrics.get('n_clusters', metrics.get('n_components'))}")
                st.write(f"**Samples Used**: {metrics['n_samples']}")
                st.write(f"**Features**: {metrics['n_features']}")
                st.write(f"**Train/Val Split**: 70% / 30%")
            
            with col2:
                st.markdown("### Performance Metrics")
                if 'inertia' in metrics:
                    st.write(f"**Inertia**: {metrics['inertia']:.2f}")
                if 'bic' in metrics:
                    st.write(f"**BIC**: {metrics['bic']:.2f}")
                if 'train_confidence' in metrics:
                    st.write(f"**Avg Confidence**: {metrics.get('train_confidence', 0):.4f}")
                
                st.markdown("**Stability**: ", unsafe_allow_html=True)
                if metrics['silhouette_diff'] < 0.05:
                    st.success(f"✓ Stable (diff: {metrics['silhouette_diff']:.4f})")
                else:
                    st.warning(f"⚠ Check overfitting (diff: {metrics['silhouette_diff']:.4f})")
        
        else:
            st.info("Run clustering first to see metrics")
    
    # ============= TAB 4: Data Overview =============
    with tab4:
        st.markdown("### Selected Indicators Summary")
        # Always use frozen display variables for overview, and robustly guard against missing data
        if 'last_labels' not in st.session_state or display_df_raw_selected is None or display_df_raw_selected.empty:
            st.info("Run clustering to see data overview.")
        elif not display_selected_indicators:
            # If user cleared all indicators but we have previous output, keep showing last output
            st.dataframe(display_df_raw_selected.describe(), width=1200, height=300)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Data Availability")
                data_avail = (1 - display_df_raw_selected.isna().sum() / len(display_df_raw_selected)) * 100
                fig = px.bar(
                    y=data_avail.index,
                    x=data_avail.values,
                    orientation='h',
                    title="Available Data %",
                    labels={'x': 'Available %', 'y': 'Indicator'}
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("### Sample Data")
                sample_df = display_df_raw_selected.head(10).copy()
                sample_df.index = sample_df.index.map(get_country_name)
                st.dataframe(sample_df, width=1000, height=300)
            try:
                csv = display_df_raw_selected.to_csv()
            except Exception:
                csv = ""
            if csv:
                st.download_button(
                    label="📥 Download Data (CSV)",
                    data=csv,
                    file_name=f"cluster_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No downloadable data available. Run clustering to generate export.")
        else:
            try:
                st.dataframe(display_df_raw_selected.describe(), width=1200, height=300)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Data Availability")
                    data_avail = (1 - display_df_raw_selected.isna().sum() / len(display_df_raw_selected)) * 100
                    fig = px.bar(
                        y=data_avail.index,
                        x=data_avail.values,
                        orientation='h',
                        title="Available Data %",
                        labels={'x': 'Available %', 'y': 'Indicator'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.markdown("### Sample Data")
                    sample_df = display_df_raw_selected.head(10).copy()
                    sample_df.index = sample_df.index.map(get_country_name)
                    st.dataframe(sample_df, width=1000, height=300)
            except Exception:
                st.info("Overview unavailable. Run clustering to generate data preview.")

            # Download data (export the displayed/frozen raw table)
            try:
                csv = display_df_raw_selected.to_csv()
            except Exception:
                csv = ""
            if csv:
                st.download_button(
                    label="📥 Download Data (CSV)",
                    data=csv,
                    file_name=f"cluster_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No downloadable data available. Run clustering to generate export.")
    
    # ============= FOOTER =============
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("📊 Source: World Bank API (Latest available year)")
    with col2:
        st.caption("🤖 Methods: K-Means & Gaussian Mixture Model")
    with col3:
        st.caption(f"⚙️ Processing: KNN Imputation (k={knn_neighbors}, metric=nan_euclidean) + StandardScaler")


if __name__ == "__main__":
    main()
