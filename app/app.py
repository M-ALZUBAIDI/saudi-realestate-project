import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# PAGE SETUP
# ============================================
st.set_page_config(page_title="Saudi Real Estate Dashboard", layout="wide")
st.title("🏠 Saudi Arabia Real Estate Dashboard")
st.markdown("Comparing Kaggle listing data against the official GASTAT price index (Q1 2022 vs Q1 2023)")

# ============================================
# LOAD DATA
# ============================================
df_2022 = pd.read_csv("data/clean_2022Q1.csv")
df_2023 = pd.read_csv("data/clean_2023Q1.csv")
df_all = pd.read_csv("data/clean_combined.csv")
comparison = pd.read_csv("data/final_comparison.csv")

# ============================================
# SIDEBAR - Dataset description + filters
# ============================================
st.sidebar.header("About this Dataset")
st.sidebar.write(
    "Real estate transactions in Saudi Arabia, Q1 2022 and Q1 2023, "
    "sourced from Kaggle. Compared against the official GASTAT price index."
)

st.sidebar.header("Filters")

region_options = sorted(df_all['region_en'].unique())
selected_regions = st.sidebar.multiselect("Region", region_options, default=region_options)

type_options = sorted(df_all['property_type_grouped'].unique())
selected_types = st.sidebar.multiselect("Property Type", type_options, default=type_options)

quarter_options = sorted(df_all['source_quarter'].unique())
selected_quarter = st.sidebar.selectbox("Quarter", ["Both"] + list(quarter_options))

price_min = int(df_all['price'].min())
price_max = int(df_all['price'].max())
price_range = st.sidebar.slider("Price Range (SAR)", price_min, price_max, (price_min, 5_000_000))

# ============================================
# APPLY FILTERS
# ============================================
filtered = df_all[
    (df_all['region_en'].isin(selected_regions)) &
    (df_all['property_type_grouped'].isin(selected_types)) &
    (df_all['price'] >= price_range[0]) &
    (df_all['price'] <= price_range[1])
]

if selected_quarter != "Both":
    filtered = filtered[filtered['source_quarter'] == selected_quarter]

# ============================================
# MAIN PAGE - Data preview + summary stats
# ============================================
st.subheader("Data Preview")
st.dataframe(filtered.head(20))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{len(filtered):,}")
col2.metric("Average Price (SAR)", f"{filtered['price'].mean():,.0f}")
col3.metric("Median Price (SAR)", f"{filtered['price'].median():,.0f}")
col4.metric("Average Space (m²)", f"{filtered['space'].mean():,.0f}")

# ============================================
# INTERACTIVE VISUALIZATIONS
# ============================================
st.subheader("Price Distribution")
fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.histplot(data=filtered, x='price', bins=50, log_scale=True, ax=ax1)
ax1.set_xlabel("Price (SAR, log scale)")
st.pyplot(fig1)

st.subheader("Price by Region")
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=filtered, x='region_en', y='price', ax=ax2)
ax2.set_yscale('log')
plt.xticks(rotation=45, ha='right')
st.pyplot(fig2)

st.subheader("Price vs Space")
fig3, ax3 = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=filtered, x='space', y='price', hue='classification_en', alpha=0.4, ax=ax3)
ax3.set_xscale('log')
ax3.set_yscale('log')
st.pyplot(fig3)

# ============================================
# INSIGHT SECTION - KAPSARC comparison
# ============================================
st.subheader("📊 Kaggle vs. Official GASTAT Index")
st.markdown(
    "This chart compares the % price change (Q1 2022 → Q1 2023) from raw Kaggle "
    "listings against the official GASTAT index for the same regions."
)

comparison_long = comparison.melt(
    id_vars='region',
    value_vars=['kaggle_pct_change', 'official_pct_change'],
    var_name='source', value_name='pct_change'
)
comparison_long['source'] = comparison_long['source'].map({
    'kaggle_pct_change': 'Kaggle Listings',
    'official_pct_change': 'Official GASTAT Index'
})

fig4, ax4 = plt.subplots(figsize=(12, 5))
sns.barplot(data=comparison_long, x='region', y='pct_change', hue='source', ax=ax4)
ax4.axhline(0, color='black', linewidth=0.8)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig4)

st.markdown(
    "**Key insight:** Official data shows steady, modest growth across most regions. "
    "Kaggle-based averages swing far more sharply and sometimes disagree on direction — "
    "likely due to outliers and property-mix shifts in raw listings data."
)

# ============================================
# WHY DO THEY DIVERGE? - Property mix explanation
# ============================================
st.subheader("🔍 Why Do Kaggle and Official Data Disagree?")
st.markdown(
    "A simple average price is sensitive to **what** was sold, not just price changes. "
    "If the mix of property types shifts between quarters, the average can move even "
    "when typical prices stay the same. Pick a region below to see its property mix shift."
)

# Let the user pick a region to investigate
divergence_region = st.selectbox(
    "Select a region to inspect",
    options=sorted(df_all['region_en'].unique()),
    index=sorted(df_all['region_en'].unique()).index('Najran') if 'Najran' in df_all['region_en'].unique() else 0
)

region_subset = df_all[df_all['region_en'] == divergence_region]

# Calculate property type mix (%) per quarter for this region
mix = region_subset.groupby(['source_quarter', 'property_type_grouped']).size().unstack(fill_value=0)
mix_pct = mix.div(mix.sum(axis=1), axis=0) * 100

col_a, col_b = st.columns([2, 1])

with col_a:
    st.markdown(f"**Property Type Mix in {divergence_region} (% of transactions)**")
    fig5, ax5 = plt.subplots(figsize=(8, 4))
    mix_pct.T.plot(kind='bar', ax=ax5)
    ax5.set_ylabel("% of Transactions")
    ax5.set_xlabel("Property Type")
    ax5.legend(title="Quarter")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig5)

with col_b:
    # Pull this region's numbers from the comparison table
    region_row = comparison[comparison['region'] == divergence_region]
    if not region_row.empty:
        kaggle_change = region_row['kaggle_pct_change'].values[0]
        official_change = region_row['official_pct_change'].values[0]
        st.metric("Kaggle Avg Price Change", f"{kaggle_change:+.1f}%")
        st.metric("Official GASTAT Change", f"{official_change:+.1f}%")
        st.metric("Gap", f"{kaggle_change - official_change:+.1f} pts")

st.markdown(
    "**Reading this:** if the bars for 2022Q1 and 2023Q1 look noticeably different "
    "(e.g. more apartments, fewer land plots, or vice versa), that shift in *what* was "
    "sold — combined with a few extreme-value transactions — helps explain why the "
    "simple average diverges from GASTAT's standardized index."
)
