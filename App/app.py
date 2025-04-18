import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px

st.set_page_config(page_title="Global Climate Change Dashboard", layout="wide")

st.title("🌍 Global Climate Change Dashboard")

st.markdown("""
This dashboard visualizes historical CO₂ emissions and forecasts future trends using an ARIMA model.  
Upload the dataset, select countries and time ranges to explore CO₂ emission patterns over the years.
""")

uploaded_file = st.sidebar.file_uploader("Upload your CO₂ dataset CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)


    st.sidebar.subheader("🔍 Filter Options")

    df = df.dropna(subset=['country', 'year', 'co2'])

    countries = df['country'].unique().tolist()
    countries.sort()

    selected_countries = st.sidebar.multiselect("Select Countries", countries, default=countries[:1])

    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (2000, max_year))

    filtered_df = df[
        (df['country'].isin(selected_countries)) &
        (df['year'] >= year_range[0]) &
        (df['year'] <= year_range[1])
    ]
    
    total_emissions = filtered_df['co2'].sum()
    st.metric(label="🌡️ Total CO₂ Emissions (Selected Range)", value=f"{total_emissions:,.2f} Million Metric Tons")

    st.subheader("📊 Filtered Raw Data")
    st.dataframe(filtered_df.head())

    def arima_forecast(df):
        df_global = df.groupby('year')['co2'].sum().reset_index()
        df_prophet = df_global.rename(columns={"year": "ds", "co2": "y"})
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')

        model = ARIMA(df_prophet['y'], order=(5, 1, 0))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=19)

        last_year = df_prophet['ds'].max().year
        years_future = pd.to_datetime([f"{last_year + i}-01-01" for i in range(1, 20)])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_prophet['ds'], df_prophet['y'], label='Historical CO₂ Emissions')
        ax.plot(years_future, forecast, label='Forecasted CO₂ Emissions (Next 19 Years)', color='red')
        ax.set_title("Forecasted CO₂ Emissions for the Next 19 Years")
        ax.set_xlabel("Year")
        ax.set_ylabel("CO₂ Emissions (Million Metric Tons)")
        ax.legend()
        return fig

    st.subheader("📈 Forecast Plot")
    fig = arima_forecast(filtered_df)
    st.pyplot(fig)

    st.subheader("📊 Emission Trends by Country")
    line_df = filtered_df.groupby(['year', 'country'])['co2'].sum().reset_index()
    pivot_df = line_df.pivot(index='year', columns='country', values='co2')
    st.line_chart(pivot_df)

    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name='filtered_co2_data.csv',
        mime='text/csv'
    )

    st.subheader("🗺️ Global CO₂ Emissions Map")

    map_df = filtered_df.groupby('country', as_index=False)['co2'].mean()

    fig_map = px.choropleth(
        map_df,
        locations="country",
        locationmode="country names",
        color="co2",
        color_continuous_scale="Reds",
        title="Average CO₂ Emissions by Country (Selected Years)"
    )
    st.plotly_chart(fig_map)



else:
    st.warning("Please upload a CSV file to get started.")
