# Streamlit App for Forecasting Future USD/MYR Exchange Rates
# General Libraries
import numpy as np
import pandas as pd
from datetime import date
import warnings
warnings.filterwarnings("ignore")

# Visualisation Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Streamlit and Interactive Visualisation
import streamlit as st
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid, GridOptionsBuilder
from pygwalker.api.streamlit import StreamlitRenderer
import streamlit.components.v1 as components

# Time Series and Econometrics
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
from pmdarima import auto_arima

# Financial Data
import yfinance as yf

# Load Models
import joblib

# https://icons.getbootstrap.com/ for icons

def streamlit_menu(example=1, options=["Home", "Contact"], icons=["coin", "bar-chart"]):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=options,  # required
                icons=icons,  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=options,  # required
            icons=icons,  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 3. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=options,  # required
            icons=icons,  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        return selected

def read_data(displayDate=True):

    complete_df = pd.read_csv("https://raw.githubusercontent.com/ooihiangee/raw_data/refs/heads/main/complete_df.csv")

    # Rename the columns
    complete_df = complete_df.rename(
        columns={'usd_myr': 'ER', 
                'crude_oil': 'CRUDE', 
                'dow_jones': 'DJ',
                'klci': 'KLCI',
                'exports_my': 'EXPMY',
                'imports_my': 'IMPMY',
                'ipi_my': 'IPIMY',
                'cpi_my': 'CPIMY',
                'm1_my': 'M1MY',
                'm2_my': 'M2MY',
                'opr_my': 'OPR',
                'exports_us': 'EXPUS',
                'imports_us': 'IMPUS',
                'ipi_us': 'IPIUS',
                'cpi_us': 'CPIUS',
                'm1_us': 'M1US',
                'm2_us': 'M2US',
                'fedrate_us': 'FFER',     
                })

    if displayDate==False:
        complete_df.set_index('Date', inplace=True)
        complete_df.index = pd.to_datetime(complete_df.index)

    return complete_df

def data_dict():

    # Data dictionary
    data_dict = {
        "No.": list(range(1, 19)),
        "Variables": [
            "USD/MYR Currency Exchange Rates", "Crude Oil Prices", "Dow Jones Industrial Average",
            "Kuala Lumpur Composite Index", "Malaysia Exports", "Malaysia Imports",
            "Malaysia Industrial Production Index", "Malaysia Consumer Price Index",
            "Malaysia Money Supply M1", "Malaysia Money Supply M2", "Malaysia Overnight Policy Rates",
            "U.S. Exports", "U.S. Imports", "U.S. Industrial Production Index",
            "U.S. Consumer Price Index", "U.S. Money Supply M1", "U.S. Money Supply M2", "U.S. Federal Fund Effective Rates"
        ],
        "Abbreviations": [
            "ER", "CRUDE", "DJ", "KLCI", "EXPMY", "IMPMY", "IPIMY", "CPIMY",
            "M1MY", "M2MY", "OPR", "EXPUS", "IMPUS", "IPIUS", "CPIUS",
            "M1US", "M2US", "FFER"
        ],
        "Descriptions": [
            "Price of Ringgit Malaysia for every 1 United States Dollar",
            "Price of every barrel of crude oil from Brent, Dubai and West Texas Intermediate (WTI)",
            "Price-weighted index that tracks 30 large, publicly owned U.S. companies trading on NYSE and NASDAQ",
            "Market-valued-weighted stock market index made up of the thirty largest companies on the Bursa Malaysia",
            "The value of goods and services exported from Malaysia",
            "The value of goods and services imported into Malaysia",
            "Measurement of production of industrial commodities in the mining, manufacturing, and electricity sectors in real terms",
            "Measurement of the cost of purchasing a constant, representative 'basket' of goods and services",
            "Currency in Circulation + Demand Deposits",
            "M1 + Narrow Quasi-Money",
            "BNM’s policy interest rate that influences, among others, banks’ lending and financing rates, as well as deposit rates.",
            "The value of goods and services exported from the United States",
            "The value of goods and services imported into the United States",
            "Measurement of the real output of industrial commodities in the mining, manufacturing, and electricity sectors",
            "Measurement of the cost of purchasing a constant, representative 'basket' of goods and services",
            "Most liquid forms of money, such as cash, checking deposits, and other highly liquid accounts.",
            "M1 plus savings accounts, small time deposits, and retail money market funds",
            "Interest rate at which depository institutions (banks and credit unions) lend reserve balances to each other overnight."
        ],
        "Sources": [
            "Yahoo Finance", "Yahoo Finance", "Yahoo Finance", "Yahoo Finance", "Malaysia Open Data",
            "Malaysia Open Data", "Malaysia Open Data", "Malaysia Open Data", "Malaysia Open Data",
            "Malaysia Open Data", "BNM", "U.S. Census Bureau", "U.S. Census Bureau", "FRED",
            "FRED", "FRED", "FRED", "FRED"
        ],
        "Sources URL": ["https://finance.yahoo.com/quote/usdmyr%3dx/history/", "https://finance.yahoo.com/quote/cl%3df/history/", 
                        "https://finance.yahoo.com/quote/%5EDJI/history/", "https://finance.yahoo.com/quote/%5EKLSE/history/", 
                        "https://open.dosm.gov.my/data-catalogue/trade_sitc_1d?section=overall&visual=table", 
                        "https://open.dosm.gov.my/data-catalogue/trade_sitc_1d?section=overall&visual=table", 
                        "https://data.gov.my/data-catalogue/ipi", "https://data.gov.my/data-catalogue/cpi_headline", 
                        "https://data.gov.my/data-catalogue/monetary_aggregates", "https://data.gov.my/data-catalogue/monetary_aggregates",
                        "https://www.bnm.gov.my/monetary-stability/opr-decisions", 
                        "https://www.census.gov/foreign-trade/balance/c0015.html", "https://www.census.gov/foreign-trade/balance/c0015.html",
                        "https://fred.stlouisfed.org/series/ipb50001n", "https://fred.stlouisfed.org/series/cpaltt01usm657n", "https://fred.stlouisfed.org/series/m1ns", 
                        "https://fred.stlouisfed.org/series/m2ns", "https://fred.stlouisfed.org/series/fedfunds"]
    }

    data_dict_df = pd.DataFrame(data_dict)
    
    return pd.DataFrame(data_dict_df)

def line_plot(selected_variable):

    # Reset the index for Plotly compatibility
    df_reset = complete_df.reset_index().rename(columns={"index": "Date"})

    # Create a Plotly line chart
    fig = px.line(
        df_reset,
        x="Date",
        y=selected_variable,
        title=f"Time Series for {selected_variable}",
        labels={"Date": "Date", selected_variable: selected_variable},  # Axis labels
        template="plotly_white"  # Clean visual theme
    )

    # Customize the layout (optional)
    fig.update_layout(
        xaxis=dict(
            showgrid=True,  # Show gridlines for better readability
            tickformat="%Y",  # Format x-axis to show only the year
        ),
        yaxis=dict(showgrid=True),
        title=dict(x=0.5),  # Center the title
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

def ts_plot():
    
    # Create subplot of three columns
    num_vars = len(complete_df.columns)
    ncols = 3  # Number of columns in the subplot grid
    nrows = (num_vars // ncols) + (num_vars % ncols > 0)  # Calculate number of rows
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=complete_df.columns  # Set titles for each subplot
    )

    # Loop through each time series data and add traces
    for i, col in enumerate(complete_df.columns):
        row = i // ncols + 1  # Determine row index (1-based)
        col_index = i % ncols + 1  # Determine column index (1-based)
        fig.add_trace(
            go.Scatter(x=complete_df.index, y=complete_df[col], mode='lines', name=col),
            row=row, col=col_index
        )
        fig.update_xaxes(title_text="Date", row=row, col=col_index)

    # Update layout for better visuals
    fig.update_layout(
        height=nrows * 250,  # Adjust height based on the number of rows
        width=900,  # Set figure width
        showlegend=False,  # Hide legend for a cleaner look
        template="plotly_white",  # Use a clean template
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def descriptive_stats(selected_variable):

    # Plot histogram for distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(complete_df[selected_variable], bins=30, kde=True, ax=ax, color="skyblue", alpha=0.7)
    ax.set_title(f"Histogram and Density Plot of {selected_variable}")
    ax.set_xlabel(selected_variable)
    ax.set_ylabel("Frequency/Density")
    st.pyplot(fig)

def acf_pacf_plot(selected_variable):

    data = complete_df[selected_variable].dropna()

    # Set up the plot area with two subplots for ACF and PACF
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f'ACF and PACF for {selected_variable}', fontsize=16)

    # Plot ACF
    plot_acf(data, ax=axes[0], lags=40, title=f'ACF of {selected_variable}')

    # Plot PACF
    plot_pacf(data, ax=axes[1], lags=40, title=f'PACF of {selected_variable}', method='ywm')

    plt.tight_layout()
    st.pyplot(fig)

def seasonal_decomposition(selected_variable):

    # Decomposition plot
    decomposition = seasonal_decompose(complete_df[selected_variable], model='additive')

    # Plot the individual components
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(decomposition.trend, label='Trend')
    ax.plot(decomposition.seasonal, label='Seasonal')
    ax.plot(decomposition.resid, label='Residual')
    ax.plot(decomposition.observed, label='Observed')

    ax.set_title(f"Decomposition of {selected_variable}")
    ax.legend()
    st.pyplot(fig)

    # Perform seasonal decomposition
    decomposition = sm.tsa.seasonal_decompose(
        complete_df[selected_variable], model="additive", period=12
    )

    # Plot the decomposed components
    fig = decomposition.plot()
    st.pyplot(fig)

@st.cache_resource

def create_lagged_features(df, num_lags=3):
    for col in df.columns:
        for lag in range(1, num_lags + 1):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Drop rows with NaN values created by lagging
    df.dropna(inplace=True)
    return df

def ets_forecast(df, forecast_period):

    forecast_results = pd.DataFrame()

    # Define a dictionary for the number of decimal places for each column
    decimal_places_dict = {'ER': 16, 'CRUDE': 15, 'DJ': 12, 'KLCI': 13, 'EXPMY': 11, 'IMPMY': 12, 'IPIMY': 3, 'CPIMY': 1, 'M1MY': 11, 'M2MY': 10, 'OPR': 2, 'EXPUS': 5, 'IMPUS': 4, 'IPIUS': 4, 'CPIUS': 3, 'M1US': 1, 'M2US': 1, 'FFER': 2}

    for column in df.columns:
        
        # Extract the time series
        ts = df[column].dropna()
        
        decomposition = seasonal_decompose(ts, model='additive', period=12)
        
        has_trend = decomposition.trend is not None and decomposition.trend.dropna().std() > 0
        has_seasonality = decomposition.seasonal is not None and decomposition.seasonal.dropna().std() > 0
        
        print(f"{column} - Trend: {has_trend}, Seasonality: {has_seasonality}")

        # Fit ETS model
        seasonal_type = 'add' if has_seasonality else None
        model = ExponentialSmoothing(
            ts,
            trend='add' if has_trend else None,
            seasonal=seasonal_type,
            seasonal_periods=12 if has_seasonality else None
        )
        
        fitted_model = model.fit()
        
        # Forecast for the next n months (1-12)
        forecast = fitted_model.forecast(steps=forecast_period)

        # Apply different rounding rules based on column
        if column == 'OPR':  # to ensure 0.25 step requirement
            forecast = round(forecast * 4) / 4

        # Round the forecast based on the decimal places for each column
        if column in decimal_places_dict:
            decimal_places = decimal_places_dict[column]
            forecast = forecast.round(decimal_places)

        forecast_results[column] = forecast

    return forecast_results

def generate_centered_table_html(df):
    html = f"""
    <style>
    .centered-table {{
        margin: auto;
        text-align: center;
    }}
    .centered-table th {{
        text-align: center;
        padding: 8px;
        border: 1px solid #ddd;
        background-color: #f2f2f2;
        color: black; /* Set header font color to black */
        font-weight: bold;
    }}
    .centered-table td {{
        text-align: center;
        padding: 8px;
        border: 1px solid #ddd;
    }}
    </style>
    <table class="centered-table">
        <thead>
            <tr>
                {''.join(f'<th>{col}</th>' for col in df.columns)}
            </tr>
        </thead>
        <tbody>
            {''.join(
                '<tr>' + ''.join(f'<td>{value}</td>' for value in row) + '</tr>'
                for row in df.values
            )}
        </tbody>
    </table>
    """
    return html

###############################################################################################################################################

# Streamlit UI - Set page configuration
st.set_page_config(layout="wide")

# 1 = sidebar menu, 2 = horizontal menu, 3 = horizontal menu w/ custom menu
selected = streamlit_menu(example = 2, 
                          options=["About", "Dashboard", "Forecasting Model", "Source Codes", "Contact Me"],
                          icons=["house", "bar-chart-fill", "bar-chart-steps", "code", "person"])

complete_df = read_data(displayDate=False)

###############################################################################################################################################

# About Page
if selected == "About":

    # Title of the page
    st.markdown(
    "<h1 style='text-align: center;'>Overview - Slideshow</h1>",
    unsafe_allow_html=True
    )

    # Slide
    html_code = """
    <div style="display: flex; justify-content: center; width: 100%;">
        <div style="position: relative; width: 80%; height: 0; padding-top: 46.2500%; padding-bottom: 0; box-shadow: 0 2px 8px 0 rgba(63,69,81,0.16); margin-top: 1.6em; margin-bottom: 0; overflow: hidden; border-radius: 8px; will-change: transform;">
            <iframe loading="lazy" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; border: none; padding: 0;margin: 0;" src="https://www.canva.com/design/DAGarHnrIs0/wg17PXW_zePkuLSVITwDPg/view?embed" allowfullscreen="allowfullscreen" allow="fullscreen">
            </iframe>
        </div>
    </div>
    """

    # Display the HTML content
    st.components.v1.html(html_code, height=650)

    # Background
    st.markdown("<h3>Background</h3>", unsafe_allow_html=True)
    st.markdown("""
    * The currency exchange rate facilitates the international trade of goods and services as well as  the transfer of capital. It indicates the external competitiveness of a country’s economy.
    * Currency strength is directly dependent on the changes in the key macroeconomic indicators such as gross domestic product, interest rate,  inflation, foreign exchange rate, unemployment, etc. 
    * Understanding their sensitivity to changes assists policymakers in predicting future directions and assessing potential economic impacts.
    """)

    # Limitations of Existing Research
    st.markdown("<h3>Limitations of Existing Research</h3>", unsafe_allow_html=True)
    st.markdown("""
    * **Focus on Authors' Own Countries:**
        * Even though many of the researchers come up with different approaches such as VAR model (Antwi et al., 2020), ARDL model (Munir & Iftikhar, 2023; Thevakumar & Jayathilaka, 2022) and deep learning models (Biswas et al., 2023) to examine the effect of macroeconomic factors on the currency exchange rates., they are focusing the studies on their own countries.
    * **Use of Annual Data:**
        * Several studies had been conducted in Malaysia (Mohamed et al., 2021; Shukri et al., 2021) to investigate the impact of economic factors on Malaysia's exchange rate volatility. However, these studies utilised annual data. 
        * Studies utilising annual data may not effectively capture the fast-paced fluctuations in currency exchange rates.     
    * **Data Obsolescence:** 
        * (Biswas et al., 2023) used data until 2019 and (Ohaegbulem & Iheaka, 2024) used data until 2021 only in their studies.
        * The outdated data may limit the relevance of their findings to current economic conditions. 
    * **Lack of Bilateral Analysis:**
        * There is a lack of studies that analyse the relationship between the macroeconomic factors of both the home country and the foreign country.
            
    Thus, this study will focus the context mainly to Malaysia instead of other countries. In addition, this study aims to also utilise the most contemporary dataset from a set of both Malaysia and US macroeconomic factors to discern the determinants of both long-run and short-run dynamics of the Malaysia currency exchange rates over the time.
    """)

    # Data
    st.markdown("<h3>Data Used</h3>", unsafe_allow_html=True)
    st.markdown("There are a total of 18 variables utilised in this study to map the relationship between the macroeconomic factors and USD/MYR currency exchange rates. All the data is secondary data obtained from a range of reliable source, namely Yahoo Finance at https://finance.yahoo.com/, Malaysia’s official open data portal at https://data.gov.my/, Central Bank of Malaysia (BNM) at https://www.bnm.gov.my/, United States Census Bureau at https://www.census.gov/ and Federal Reserve Economic Data (FRED) at https://fred.stlouisfed.org/. The data obtained is of monthly time series data which spans the period from January 2015 until July 2024.")
    
    # Data Dictionary
    st.caption('**Table 1.** Data Dictionary.')
    st.dataframe(data_dict(), height=360, use_container_width=True, hide_index=True)
    
    # Time Series Data
    st.caption("**Table 2. Time Series Data of Macroeconomic Variables and USD/MYR Currency Exchange Rates.**")
    st.dataframe(read_data(), height=360, use_container_width=True, hide_index=True)

    # Model Performance
    st.markdown("<h3>Model Performance</h3>", unsafe_allow_html=True)
    st.markdown("The model performance is evaluated based on the Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE) and R-squared (R2). The model with the lowest error metrics is considered the best model for forecasting the USD/MYR currency exchange rates.")
    
    # Define the data as a dictionary
    performance_data = {
        'No.': [1, 2, 3, 4, 5, 6],
        'Model': ['ARDL', 'SVM', 'Random Forest', 'XGBoost', 'LightGBM', 'LSTM'],
        'RMSE': [0.059773664, 0.059151017, 0.057159107, 0.056987775, 0.056604, 0.058797031],
        'NRMSE': [0.099518518, 0.098481859, 0.095165483, 0.094880229, 0.094241273, 0.097892499],
        'MAE': [0.044245332, 0.04290451, 0.045547599, 0.042900981, 0.045134084, 0.044116224],
        'MAPE (%)': [0.98252715, 0.950490555, 1.012725491, 0.946950675, 0.998747381, 0.974567329],
        'R2': [0.908097826, 0.910002496, 0.915961769, 0.916464816, 0.917586136, 0.911076444],
        'Training Time (s)': [30872.7, 1.588016987, 175.2644548, 49.95268512, 17.94534421, 92.97024298]
    }

    # Create the DataFrame
    performance_data_df = pd.DataFrame(performance_data)

    # Display the DataFrame
    st.dataframe(performance_data_df, hide_index=True)

    # References
    st.markdown("<h3>References</h3>", unsafe_allow_html=True)
    st.markdown('''
    Antwi, S., Issah, M., Patience, A., & Antwi, S. (2020). The effect of macroeconomic variables on exchange rate: Evidence from Ghana. Cogent Economics & Finance, 8(1), 1821483. https://doi.org/10.1080/23322039.2020.1821483\n
    Biswas, A., Uday, I. A., Rahat, K. M., Akter, M. S., & Mahdy, M. R. C. (2023). Forecasting the United State Dollar(USD)/Bangladeshi Taka (BDT) exchange rate with deep learning models: Inclusion of macroeconomic factors influencing the currency exchange rates. PLOS ONE, 18(2), e0279602. https://doi.org/10.1371/journal.pone.0279602\n
    Mohamed, S., Abdullah, M., Noh, M. K. A., Isa, M. A. M., Hassan, S. S., Ibrahim, W. M. F. W., & Nasrul, F. (2021). Impact of Economic Factors Towards Exchange Rate in Malaysia. International Journal of Academic Research in Economics and Managment and Sciences, 10(1).\n
    Munir, K., & Iftikhar, M. (2023). Macroeconomic determinants of the real exchange rate in Pakistan. International Journal of ADVANCED AND APPLIED SCIENCES, 10(8), 12-18. https://doi.org/10.21833/ijaas.2023.08.002\n 
    Ohaegbulem, E. U., & Iheaka, V. C. (2024). The Impact of Macroeconomic Factors on Nigerian-Naira Exchange Rate Fluctuations (1981-2021). Asian Journal of Probability and Statistics, 26(2), 18-36. https://doi.org/10.9734/ajpas/2024/v26i2589\n 
    Shukri, J., Habibullah, M. S., Ghani, R. A., & Suhaily, M. (2021). The macroeconomic fundamentals of the real exchange rate in Malaysia: Some empirical evidence'. Jurnal Ekonomi Malaysia, 55(2), 81-89.\n 
    Thevakumar, P., & Jayathilaka, R. (2022). Exchange rate sensitivity influencing the economy: The case of Sri Lanka. PLOS ONE, 17(6), e0269538. https://doi.org/10.1371/journal.pone.0269538\n 
    ''')

###############################################################################################################################################

# Dashboard Page
if selected == "Dashboard":

    # Title of the page
    st.title('Interactive Dashboard')

    with st.expander(r"$\textsf{\Large Playground}$", expanded=True):
        st.info("This section provides users flexibility to mingle with the data and visualise it.")

        # Reset the index to make it a column
        complete_df_reset = complete_df.reset_index()

        # Configure Pyg Renderer with performance options
        pyg_app = StreamlitRenderer(
            complete_df_reset, 
            spec="./config.json",  # Optional configuration file
            scrolling=True,        # Enable scrolling for large datasets
            dark_theme=False,      # Optional theme
            use_kernel_calc=True   # Use kernel calculation for performance
        )
        pyg_app.explorer()
   
    with st.expander(r"$\textsf{\Large Time Series Plots}$", expanded=True):
        # Time Series Plots
        st.markdown("<h2 style='text-align: center;'>Time Series Plots</h2>", unsafe_allow_html=True)
        ts_plot()

    with st.expander(r"$\textsf{\Large Detailed Analysis of Each Time Series Variable}$", expanded=True):
        # Select Variable
        selected_variable = st.selectbox("Select a time series variable: ", complete_df.columns)

        # Create a container with a scrollable area
        with st.container():

            # Descriptive Plots
            st.markdown("<h2 style='text-align: center;'>Histogram</h2>", unsafe_allow_html=True)
            descriptive_stats(selected_variable)

            # Seasonal Decomposition Plots
            st.markdown("<h2 style='text-align: center;'>Seasonal Decomposition Plots</h2>", unsafe_allow_html=True)
            seasonal_decomposition(selected_variable)

            # ACF & PACF Plots 
            st.markdown("<h2 style='text-align: center;'>ACF & PACF Plots</h2>", unsafe_allow_html=True)
            acf_pacf_plot(selected_variable)

###############################################################################################################################################

# Model Page
if selected == "Forecasting Model":

    # Sidebar Navigation
    st.title("Time Series Forecasting")
    st.write("This section forecasts USD/MYR currency exchange rates using pre-trained models on historical monthly data of macroeconomic factors with lagged features.")  
    
    # Steps to Follow
    with st.expander(r"$\textsf{\Large Steps to Follow}$", expanded=True):
        st.markdown("""
        1. **Choose the Forecast Horizon**: Select the number of months you want to forecast.
        2. **Upload Data**: Upload your CSV file with at least 24 months observation points of the macroeconomic factors.
        """)

    # What the App Will Do Next
    with st.expander(r"$\textsf{\Large What the App Will Do Next}$", expanded=False):
        st.markdown("""
        3. **Data Preprocessing**: Log-transform the data, first difference the data and generate lagged features.
        4. **Modeling**: Fit the models using the historical data.
        5. **Forecasting**: Forecast the future monthly average USD/MYR exchange rates for the chosen number of months.
        """)

    # Sample Input Data
    with st.expander(r"$\textsf{\Large Sample Input Data}$", expanded=True):
        st.dataframe(read_data(), height=210, hide_index=True)

    # Forecast Settings
    st.subheader("Choose Your Forecast Horizon")
    forecast_period = st.slider("Forecast Period (in months)", min_value=1, max_value=12, value=6)

    # User Upload Data
    st.subheader("Upload Your Time Series Data")
    uploaded_file = st.file_uploader("Upload CSV File (Date, Value)", type=["csv"])

    if uploaded_file:

        # Load and display data
        data_load_state = st.text('Loading data...') 

        try:
            data = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
        except ValueError as ve:
            st.error(f"Error: {ve}. Please ensure the 'Date' column exists and is in a valid date format.")
        except pd.errors.ParserError:
            st.error("Error: Failed to parse the CSV file. Please ensure it is a valid CSV file.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

        data_load_state.text('Loading data... done!')

        st.write("Uploaded Data:")
        st.dataframe(data, height=210, use_container_width=True)

        # Ensure every required column exists
        required_columns = ['ER', 'CRUDE', 'DJ', 'KLCI', 'EXPMY', 'IMPMY', 'IPIMY', 'CPIMY', 'M1MY', 'M2MY', 'OPR', 'EXPUS', 'IMPUS', 'IPIUS', 'CPIUS', 'M1US', 'M2US', 'FFER']
        
        # Find missing columns
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            missing_columns_str = ", ".join(missing_columns)
            st.error(f"The following required columns are missing from the CSV file: {missing_columns_str}")
        else:
            st.success("All required columns are present in the CSV file.")

            # Parallel processing to fit ARIMA models and forecast
            data_load_state = st.text('Loading ETS model for the forecasting...it may take up to a few seconds...')

            forecast_df = ets_forecast(complete_df, forecast_period=forecast_period)

            data = pd.concat([data, forecast_df], axis=0, ignore_index=False) 

            st.dataframe(data, height=210, use_container_width=True)

            # Log transform the data
            st.write(f"Log-transforming the data...")
            data_log = np.log(data)
            st.dataframe(data_log, height=210, use_container_width=True)

            # Generate lagged features
            st.write(f"First differencing the data...")
            data_log_diff = data_log.diff().dropna()
            st.dataframe(data_log_diff, height=210, use_container_width=True)

            # Generate lagged features
            st.write(f"Generating lagged features (up to 3 months)...")
            data_with_lags = create_lagged_features(data_log_diff, num_lags=3)
            st.dataframe(data_with_lags, height=210, use_container_width=True)

            # Choose the segment of the dataframe to forecast
            combined_df_to_forecast = data_with_lags.iloc[-forecast_period:]

            # -----Load ARDL Model----
            # data_load_state = st.text('Loading ARDL model...')
            # time.sleep(3)

            # -----Load SVM Model-----
            data_load_state = st.text('Loading SVM model...')
            loaded_model = joblib.load('pkl/best_svm_model.pkl') 
            loaded_features = joblib.load('pkl/best_svm_features.pkl') 

            selected_feature_indices = loaded_features.get_support(indices=True)  # Retrieve the names of the selected features
            selected_feature_names = combined_df_to_forecast.columns[selected_feature_indices] 

            X_forecast_svm = combined_df_to_forecast[selected_feature_names] # Select the relevant features from the combined dataframe
            y_forecast_svm = loaded_model.predict(X_forecast_svm) # Make predictions

            log_ER_previous = np.log(complete_df['ER']).iloc[-len(y_forecast_svm)-1:-1].values  # Get the previous log values
            log_ER_reverted_forecast = y_forecast_svm+ log_ER_previous
            ER_forecast_svm = np.exp(log_ER_reverted_forecast)

            # -----Load Random Forest Model-----
            data_load_state = st.text('Loading Random Forest model...')
            loaded_model = joblib.load('pkl/best_rf_model.pkl') 
            loaded_features = joblib.load('pkl/best_rf_features.pkl') 

            selected_feature_indices = loaded_features.get_support(indices=True)  # Retrieve the names of the selected features
            selected_feature_names = combined_df_to_forecast.columns[selected_feature_indices] 

            X_forecast_rf = combined_df_to_forecast[selected_feature_names] # Select the relevant features from the combined dataframe
            y_forecast_rf = loaded_model.predict(X_forecast_rf) # Make predictions

            log_ER_previous = np.log(complete_df['ER']).iloc[-len(y_forecast_rf)-1:-1].values  # Get the previous log values
            log_ER_reverted_forecast = y_forecast_rf+ log_ER_previous
            ER_forecast_rf = np.exp(log_ER_reverted_forecast) 

            # -----Load XGBoost Model-----
            data_load_state = st.text('Loading XGBoost model...')
            loaded_model = joblib.load('pkl/best_xgb_model.pkl') 
            loaded_features = joblib.load('pkl/best_xgb_features.pkl') 

            selected_feature_indices = loaded_features.get_support(indices=True)  # Retrieve the names of the selected features
            selected_feature_names = combined_df_to_forecast.columns[selected_feature_indices] 

            X_forecast_xgb = combined_df_to_forecast[selected_feature_names] # Select the relevant features from the combined dataframe
            y_forecast_xgb = loaded_model.predict(X_forecast_xgb) # Make predictions

            log_ER_previous = np.log(complete_df['ER']).iloc[-len(y_forecast_xgb)-1:-1].values  # Get the previous log values
            log_ER_reverted_forecast = y_forecast_xgb+ log_ER_previous
            ER_forecast_xgb = np.exp(log_ER_reverted_forecast)

            # -----Load LightGBM Model-----
            data_load_state = st.text('Loading LightGBM model...')
            loaded_model = joblib.load('pkl/best_lgb_model.pkl') 
            loaded_features = joblib.load('pkl/best_lgb_features.pkl') 

            selected_feature_indices = loaded_features.get_support(indices=True)  # Retrieve the names of the selected features
            selected_feature_names = combined_df_to_forecast.columns[selected_feature_indices] 

            X_forecast_lgb = combined_df_to_forecast[selected_feature_names] # Select the relevant features from the combined dataframe
            y_forecast_lgb = loaded_model.predict(X_forecast_lgb) # Make predictions

            log_ER_previous = np.log(complete_df['ER']).iloc[-len(y_forecast_lgb)-1:-1].values  # Get the previous log values
            log_ER_reverted_forecast = y_forecast_lgb+ log_ER_previous
            ER_forecast_lgb = np.exp(log_ER_reverted_forecast)

            # -----Load LSTM Model-----
            data_load_state = st.text('Loading LSTM model...')
            loaded_model = joblib.load('pkl/best_lstm_model.pkl') 
            loaded_features = joblib.load('pkl/best_lstm_features.pkl') 

            selected_feature_indices = loaded_features.get_support(indices=True) # Retrieve the names of the selected features
            selected_feature_names = combined_df_to_forecast.columns[selected_feature_indices]

            X_forecast_lstm = combined_df_to_forecast[selected_feature_names] # Select the relevant features from the combined dataframe
            X_forecast_lstm = X_forecast_lstm.values.reshape((X_forecast_lstm.shape[0], 1, X_forecast_lstm.shape[1]))

            y_forecast_lstm = loaded_model.predict(X_forecast_lstm)
            y_forecast_lstm = y_forecast_lstm.flatten() # Make predictions

            log_ER_previous = np.log(complete_df['ER']).iloc[-len(y_forecast_lstm)-1:-1].values  # Get the previous log values
            log_ER_reverted_forecast = y_forecast_lstm + log_ER_previous
            ER_forecast_lstm = np.exp(log_ER_reverted_forecast) 

            # Define the forecasted data
            data = {
                "SVM": ER_forecast_svm,
                "RF": ER_forecast_rf,
                "XGB": ER_forecast_xgb,
                "LightGBM": ER_forecast_lgb,
                "LSTM": ER_forecast_lstm
            }

            # Define the dates as the index
            dates = pd.date_range(start=forecast_df.index[0], end=forecast_df.index[-1], freq="MS")

            # Create the DataFrame
            forecast_df = pd.DataFrame(data, index=dates)
            forecast_df.index.name = "Date"

            # Plot the forecast data
            forecast_df.index.name = "Date"
            forecast_df.reset_index(inplace=True)

            # Ensure Date is in datetime format
            forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])

            # Streamlit App
            st.title("Forecasted USD/MYR Exchange Rates by Models")

            # Line Chart using Altair
            chart = alt.Chart(forecast_df).transform_fold(
                fold=["SVM", "RF", "XGB", "LightGBM", "LSTM"],
                as_=["Model", "Exchange Rate"]
            ).mark_line(point=True).encode(
                x=alt.X("Date:T", title="Date", timeUnit="yearmonth", axis=alt.Axis(format="%b %y")),
                y=alt.Y("Exchange Rate:Q", title="Exchange Rate (USD/MYR)", scale=alt.Scale(zero=False)),
                color=alt.Color("Model:N", title="Model"),
                tooltip=["Date:T", "Model:N", "Exchange Rate:Q"]
            ).properties(
                width=700,
                height=400,
                title="Forecasted Exchange Rates"
            )

            st.altair_chart(chart, use_container_width=True)

            # Forecast Future Values
            st.markdown(generate_centered_table_html(forecast_df), unsafe_allow_html=True)
    
    else:
        st.info("Please upload a CSV file to start forecasting.")

###############################################################################################################################################

# Source Codes Page
if selected == "Source Codes":

    st.title("Source Codes")
 
    st.markdown("""
    <a href="https://github.com/ooihiangee/Econometric-Modelling-of-USD-MYR" target="_blank" style="background-color: white; color: black; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; border-radius: 5px;">Visit my GitHub Repository</a>
    """, unsafe_allow_html=True)

    st.markdown("""
    This GitHub repository includes:
    - Pre-trained model files (`.pkl`) for efficient deployment.
    - A Jupyter Notebook detailing the entire workflow from data aggregation to evaluation.
    - A Streamlit script for deploying the web application.
                
    It contains the comprehensive source code and resources for analysing the relationship between USD/MYR currency exchange rates and key macroeconomic factors. It is organised into the following components:

    1. **Data Collection**  
    Scripts and methodologies for gathering macroeconomic and exchange rate data.

    2. **Data Preprocessing**  
    Aggregate and merge raw data to ensure consistency.

    3. **Data Cleaning**  
    Steps to refine and clean the dataset for more accurate analysis.

    4. **Data Transformation**  
    Log transformations, differencing and introduction of lag features to increase the predictive power of the models.

    5. **Exploratory Data Analysis (EDA)**  
    Insights into the time series data through statistical summaries and visualisations.

    6. **Modeling and Evaluation**  
    Implementation of econometric and machine learning models (e.g., ARDL, Random Forest, XGBoost, LSTM) to analyse the relationships, along with model evaluation.

    8. **Streamlit Application Script**  
    Python script to built a user-friendly web application built with Streamlit for interactive analysis and forecasting.  
    """)

##############################################################################################################################################

# Source Codes Page
if selected == "Contact Me":

    info_icon = ":information_source:"

    st.info(
        f"{info_icon} You can always reach me at hiangee@yahoo.com should you encounter any technical issues or have any feedback to make improvements to this app.",
    )

    import os

    # Function to save feedback to a CSV file
    def save_feedback(feedback_data):

        # Check if the feedback file exists, if not create it
        if not os.path.isfile('feedback.csv'):
            # Create a new DataFrame and save it to CSV
            df = pd.DataFrame(columns=['Name', 'Feedback'])
            df.to_csv('feedback.csv', index=False)

        # Append the new feedback to the existing CSV file
        with open('feedback.csv', 'a') as f:
            feedback_data.to_csv(f, header=False, index=False)

    # Streamlit app layout
    st.title("User Feedback Form")

    # Create a form for user feedback
    with st.form(key='feedback_form'):
        name = st.text_input("Your Name")
        feedback = st.text_area("Your Feedback", height=150)
        
        submit_button = st.form_submit_button("Submit Feedback")

        if submit_button:
            # Create a DataFrame from the input data
            feedback_data = pd.DataFrame({
                'Name': [name],
                'Feedback': [feedback]
            })

            # Save the feedback
            save_feedback(feedback_data)

            # Display a success message
            st.success("Thank you for your feedback!")

    # Optional: Display existing feedback (for admin view)
    if st.checkbox("Show previous feedback"):
        if os.path.isfile('feedback.csv'):
            previous_feedback = pd.read_csv('feedback.csv')
            previous_feedback.reset_index(drop=True, inplace=True)
            previous_feedback.index = range(1, len(previous_feedback) + 1)
            st.table(previous_feedback)
            # st.write(previous_feedback)
        else:
            st.write("No feedback received yet.")
