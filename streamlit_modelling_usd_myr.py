# Streamlit App for Forecasting Future USD/MYR Exchange Rates
# General Libraries
import numpy as np
import pandas as pd
from datetime import date
import time
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
import statsmodels.api as sm
# from pmdarima import auto_arima

# Financial Data
import yfinance as yf

# Parallel Processing
from joblib import Parallel, delayed
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

def load_rf_model():
    return joblib.load("random_forest_model.pkl")

def create_lagged_features(df, num_lags=3):
    for col in df.columns:
        for lag in range(1, num_lags + 1):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Drop rows with NaN values created by lagging
    df.dropna(inplace=True)
    return df

def fit_and_forecast(series, column_name, n_periods):
    print(f"Processing {column_name}")
    model = auto_arima(series, seasonal=True, m=12, stepwise=True, suppress_warnings=True, error_action='ignore')
    forecast = model.predict(n_periods=n_periods)  
    return column_name, forecast

# Generate HTML for centered table
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

# FontAwesome icon for info
info_icon = ":information_source:"

st.info(
    f"{info_icon} You can always reach me at hiangee@yahoo.com should you encounter any technical issues or have any feedback to make improvements to this app.",
)

# 1 = sidebar menu, 2 = horizontal menu, 3 = horizontal menu w/ custom menu
selected = streamlit_menu(example = 2, 
                          options=["About", "Dashboard", "Forecasting Model", "Source Codes", "UAT"],
                          icons=["house", "bar-chart-fill", "bar-chart-steps", "file-earmark-medical-fill", "file-earmark-medical-fill"])

complete_df = read_data(displayDate=False)

###############################################################################################################################################

# About Page
if selected == "About":

    # Title of the page
    st.title('Econometric Modelling of USD/MYR Exchange Rate Dynamics and Key Macroeconomic Factors')

    # Author
    st.markdown('''
    **Ooi Hian Gee (17203457)** - *Master of Data Science*
    ''')

    # Abstract
    st.markdown("<h3>Abstract</h3>", unsafe_allow_html=True)
    st.info('''
    Stability in a country’s currency exchange rates is extremely crucial as fluctuations in the exchange rates significantly impact the international trade, investment, debt, and a country’s overall economic health. The exchanges rates are always linked to a variety of internal and external factors, among them are what we call macroeconomic factors which include money supply, inflation rates, Industrial Production Index (IPI) and others. This research seeks to explore the underlying relationship between these macroeconomic variables and the currency exchange rate of Malaysian Ringgits (MYR) by leveraging latest available data up to the year of 2024 using different models. The motivation behind this research lies in the recognition that a thorough understanding of the relationships between macroeconomic variables and currency exchange rates can be beneficial for informed policymaking and budgeting. In view of Malaysia as a developing country where its economy is evolving by leap and bounds at the present, a comprehensive analysis becomes necessary.     ''')

    st.markdown('**Keywords:** *monetary economics; bilateral analysis; predictive model; machine learning*')
    
    # Background
    st.markdown("<h3>Background</h3>", unsafe_allow_html=True)
    st.markdown("""
    The currency exchange rate, facilitates the international trade of goods and services as well as  the transfer of capital. It indicates the external competitiveness of a country’s economy.
    Currency strength is directly dependent on the changes in the minimum of the key macroeconomic indicators such as gross domestic product, main interest rate,  inflation, foreign exchange rate, unemployment, etc. 
    Gauging the sensitivity of currency exchange rates reacting to changes assists policymakers to predict their future direction and further understand the potential impacts brought to the country’s economy.
                """)

    # Problem Statement
    st.markdown("<h3>Problem Statement</h3>", unsafe_allow_html=True)
    st.markdown("""
    The exchange rate plays a crucial role in a country's international trade and economic position. Fluctuations in exchange rates can have profound consequences for policymakers, investors, businesses and consumers in making their decisions. Even though many of the researchers come up with different approaches such as VAR model (Antwi et al., 2020), ARDL model (Munir & Iftikhar, 2023; Thevakumar & Jayathilaka, 2022) and deep learning models (Biswas et al., 2023) to examine the effect of macroeconomic factors on the currency exchange rates., they are focusing the studies on their own countries. Several studies had been conducted in Malaysia (Mohamed et al., 2021; Shukri et al., 2021) to investigate the impact of economic factors on Malaysia's exchange rate volatility. However, these studies utilised annual data which may not effectively capture the fast-paced fluctuations in currency exchange rates.\n
    Many of the recent research undertook a variety of novel methods and models to draw relationships between macroeconomic features and the currency pairs. Nevertheless, the data they used is mostly not up to date. For instance, (Biswas et al., 2023) used data until 2019 and (Ohaegbulem & Iheaka, 2024) used data until 2021 only in their studies. This limits the relevance of their findings in the context of current economic conditions. Aside from that, most studies focus solely on the macroeconomic factors of only one country. Since currency pairs represent the relative value between two countries' currencies, we may overlook the potential mutual influence from the other country on the strength of currency pairs.\n
    In conclusion, the existing literature furnishes substantial relationship established between the macroeconomic variables and the overall currency exchange rate of a country. Notably, this study will fill the research gap by focusing the context mainly to Malaysia instead of other countries. In addition, this study aims to also utilise the most contemporary dataset from a set of both Malaysia and US macroeconomic factors to discern the determinants of both long-run and short-run dynamics of the Malaysia currency exchange rates over the time.
    """)

    # Data
    st.markdown("<h3>Data Used</h3>", unsafe_allow_html=True)
    st.markdown("There are a total of 18 variables utilised in this study to map the relationship between the macroeconomic factors and USD/MYR currency exchange rates. All the data is secondary data obtained from a range of reliable source, namely Yahoo Finance at https://finance.yahoo.com/, Malaysia’s official open data portal at https://data.gov.my/, Central Bank of Malaysia (BNM) at https://www.bnm.gov.my/, United States Census Bureau at https://www.census.gov/ and Federal Reserve Economic Data (FRED) at https://fred.stlouisfed.org/. The data obtained is of monthly time series data which spans the period from January 2015 until July 2024.")
   
    # Data Dictionary
    st.caption('**Table 1.** Data Dictionary.')
    AgGrid(data_dict(), height=560)

    # Data Points   
    # st.caption('**Table 2.** Time Series Data of Macroeconomic Variables and USD/MYR Currency Exchange Rates.')
    # AgGrid(read_data(displayDate=True), height=350)
    st.caption("**Table 2. Time Series Data**")
    gb = GridOptionsBuilder.from_dataframe(read_data())
    gb.configure_default_column(autoHeight=True)
    gb.configure_grid_options(domLayout='autoHeight', autoSizeColumns=True)
    grid_options = gb.build()
    AgGrid(read_data(displayDate=True), gridOptions=grid_options, height=300)

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

    # Display the DataFrame
    # st.dataframe(complete_df)
    # Additional expandable section for further details

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
                
        # # Dropdown for selecting a variable
        # selected_variable = st.selectbox("", complete_df.columns)
        
        # # Time Series Plots
        # st.markdown("<h2>Time Series Plots</h2>", unsafe_allow_html=True)
        # line_plot(selected_variable)

        # # Descriptive Statistics
        # st.markdown("<h2>Time Series Plots</h2>", unsafe_allow_html=True)    
        # descriptive_stats(selected_variable)

        # # ACF and PACF Plots
        # st.markdown("<h2>ACF and PACF Plots</h2>", unsafe_allow_html=True)
        # acf_pacf_plot(selected_variable)

###############################################################################################################################################

# Model Page
if selected == "Forecasting Model":

    # Sidebar Navigation
    st.title("Time Series Forecasting")
    st.write("This section forecasts USD/MYR currency exchange rates using pre-trained models on historical monthly data of macroeconomic factors with lagged features.")  
    
    # st.info(f"""1. Please make sure you have at least 5 months observation points.           
    #         2. To follow the naming conventions of each column correcly.  
    #          """)
    
    # Steps to Follow
    with st.expander(r"$\textsf{\Large Steps to Follow}$", expanded=True):
        st.markdown("""
        1. **Choose the Forecast Horizon**: Select the number of months you want to forecast.
        2. **Upload Data**: Upload your CSV file with at least 5 months observation points of the macroeconomic factors.
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
        st.dataframe(read_data(), height=210)

    # Forecast Settings
    st.subheader("Choose Your Forecast Horizon")
    forecast_period = st.slider("Forecast Period (in months)", min_value=1, max_value=12, value=6)

    # User Upload Data
    st.subheader("Upload Your Time Series Data")
    uploaded_file = st.file_uploader("Upload CSV File (Date, Value)", type=["csv"])

    # if uploaded_file:

    #     # Load and display data
    #     data_load_state = st.text('Loading data...')    
    #     data = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
    #     data_load_state.text('Loading data... done!')
    #     st.write("Uploaded Data:")
    #     st.dataframe(data, height=210, use_container_width=True)

    #     # Ensure every required column exists
    #     required_columns = ['ER', 'CRUDE', 'DJ', 'KLCI', 'EXPMY', 'IMPMY', 'IPIMY', 'CPIMY', 'M1MY', 'M2MY', 'OPR', 'EXPUS', 'IMPUS', 'IPIUS', 'CPIUS', 'M1US', 'M2US', 'FFER']
        
    #     # Find missing columns
    #     missing_columns = [col for col in required_columns if col not in data.columns]

    #     if missing_columns:
    #         missing_columns_str = ", ".join(missing_columns)
    #         st.error(f"The following required columns are missing from the CSV file: {missing_columns_str}")
    #     else:
    #         st.success("All required columns are present in the CSV file.")

    #         # Parallel processing to fit ARIMA models and forecast
    #         data_load_state = st.text('Loading ARIMA model for the forecasting...it may take up to 1 minute...')

    #         results = Parallel(n_jobs=-1)(
    #             delayed(fit_and_forecast)(data[col], col, forecast_period) for col in data.columns
    #         )

    #         # Convert the results into a DataFrame
    #         forecast_dict = {col: forecast for col, forecast in results}
    #         forecast_df = pd.DataFrame(forecast_dict, index=pd.date_range(
    #             start=data.index[-1] + pd.offsets.MonthBegin(1), periods=forecast_period, freq='MS'
    #         ))

    #         data = pd.concat([complete_df, forecast_df], axis=0, ignore_index=False) 

    #         st.dataframe(data, height=210, use_container_width=True)

    #         # Log transform the data
    #         st.write(f"Log-transforming the data...")
    #         data_log = np.log(data)
    #         st.dataframe(data_log, height=210, use_container_width=True)

    #         # Generate lagged features
    #         st.write(f"First differencing the data...")
    #         data_log_diff = data_log.diff().dropna()
    #         st.dataframe(data_log_diff, height=210, use_container_width=True)

    #         # Generate lagged features
    #         st.write(f"Generating lagged features (up to 3 months)...")
    #         data_with_lags = create_lagged_features(data_log_diff, num_lags=3)
    #         st.dataframe(data_with_lags, height=210, use_container_width=True)

    #         # Choose the segment of the dataframe to forecast
    #         combined_df_to_forecast = data_with_lags.iloc[-forecast_period:]

    #         # -----Load ARDL Model----
    #         # data_load_state = st.text('Loading ARDL model...')
    #         # time.sleep(3)

    #         # -----Load SVM Model-----
    #         data_load_state = st.text('Loading SVM model...')
    #         loaded_model = joblib.load('best_svm_model.pkl') 
    #         loaded_features = joblib.load('best_svm_features.pkl') 

    #         selected_feature_indices = loaded_features.get_support(indices=True)  # Retrieve the names of the selected features
    #         selected_feature_names = combined_df_to_forecast.columns[selected_feature_indices] 

    #         X_forecast_svm = combined_df_to_forecast[selected_feature_names] # Select the relevant features from the combined dataframe
    #         y_forecast_svm = loaded_model.predict(X_forecast_svm) # Make predictions

    #         log_ER_previous = np.log(complete_df['ER']).iloc[-len(y_forecast_svm)-1:-1].values  # Get the previous log values
    #         log_ER_reverted_forecast = y_forecast_svm+ log_ER_previous
    #         ER_forecast_svm = np.exp(log_ER_reverted_forecast)

    #         # -----Load Random Forest Model-----
    #         data_load_state = st.text('Loading Random Forest model...')
    #         loaded_model = joblib.load('best_rf_model.pkl') 
    #         loaded_features = joblib.load('best_rf_features.pkl') 

    #         selected_feature_indices = loaded_features.get_support(indices=True)  # Retrieve the names of the selected features
    #         selected_feature_names = combined_df_to_forecast.columns[selected_feature_indices] 

    #         X_forecast_rf = combined_df_to_forecast[selected_feature_names] # Select the relevant features from the combined dataframe
    #         y_forecast_rf = loaded_model.predict(X_forecast_rf) # Make predictions

    #         log_ER_previous = np.log(complete_df['ER']).iloc[-len(y_forecast_rf)-1:-1].values  # Get the previous log values
    #         log_ER_reverted_forecast = y_forecast_rf+ log_ER_previous
    #         ER_forecast_rf = np.exp(log_ER_reverted_forecast) 

    #         # -----Load XGBoost Model-----
    #         data_load_state = st.text('Loading XGBoost model...')
    #         loaded_model = joblib.load('best_xgb_model.pkl') 
    #         loaded_features = joblib.load('best_xgb_features.pkl') 

    #         selected_feature_indices = loaded_features.get_support(indices=True)  # Retrieve the names of the selected features
    #         selected_feature_names = combined_df_to_forecast.columns[selected_feature_indices] 

    #         X_forecast_xgb = combined_df_to_forecast[selected_feature_names] # Select the relevant features from the combined dataframe
    #         y_forecast_xgb = loaded_model.predict(X_forecast_xgb) # Make predictions

    #         log_ER_previous = np.log(complete_df['ER']).iloc[-len(y_forecast_xgb)-1:-1].values  # Get the previous log values
    #         log_ER_reverted_forecast = y_forecast_xgb+ log_ER_previous
    #         ER_forecast_xgb = np.exp(log_ER_reverted_forecast)  

    #         # Define the forecasted data
    #         data = {
    #             "SVM": ER_forecast_svm,
    #             "RF": ER_forecast_rf,
    #             "XGB": ER_forecast_xgb
    #         }

    #         # Define the dates as the index
    #         dates = pd.date_range(start=forecast_df.index[0], end=forecast_df.index[-1], freq="MS")

    #         # Create the DataFrame
    #         forecast_df = pd.DataFrame(data, index=dates)
    #         forecast_df.index.name = "Date"

    #         # Plot the forecast data
    #         forecast_df.index.name = "Date"
    #         forecast_df.reset_index(inplace=True)

    #         # Ensure Date is in datetime format
    #         forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])

    #         # Streamlit App
    #         st.title("Forecasted USD/MYR Exchange Rates by Models")

    #         # Line Chart using Altair
    #         chart = alt.Chart(forecast_df).transform_fold(
    #             fold=["SVM", "RF", "XGB"],
    #             as_=["Model", "Exchange Rate"]
    #         ).mark_line(point=True).encode(
    #             x=alt.X("Date:T", title="Date", timeUnit="yearmonth", axis=alt.Axis(format="%b %y")),
    #             y=alt.Y("Exchange Rate:Q", title="Exchange Rate (USD/MYR)", scale=alt.Scale(zero=False)),
    #             color=alt.Color("Model:N", title="Model"),
    #             tooltip=["Date:T", "Model:N", "Exchange Rate:Q"]
    #         ).properties(
    #             width=700,
    #             height=400,
    #             title="Forecasted Exchange Rates"
    #         )

    #         st.altair_chart(chart, use_container_width=True)

    #         # Forecast Future Values
    #         st.markdown(generate_centered_table_html(forecast_df), unsafe_allow_html=True)

    #         # # Prepare forecast dates
    #         # last_date = data["Date"].iloc[-1]
    #         # forecast_dates = [last_date + timedelta(days=30 * i) for i in range(1, forecast_period + 1)]

    #         # # Display Results
    #         # forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecasted Value": predictions})
    #         # st.write("Forecasted Values:")
    #         # st.dataframe(forecast_df)

    #         # # Plot Forecast
    #         # st.subheader("Forecast Visualization")
    #         # fig = go.Figure()
    #         # fig.add_trace(go.Scatter(x=data["Date"], y=data["value"], mode='lines+markers', name="Historical Data"))
    #         # fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecasted Value"],
    #         #                         mode='lines+markers', name="Forecasted Data"))
    #         # st.plotly_chart(fig, use_container_width=True)

    # else:
    #     st.info("Please upload a CSV file to start forecasting.")

###############################################################################################################################################

# Source Codes Page
if selected == "Source Codes":

    st.title("Github Codes")

    # st.title("Jupyterlite in Streamlit")
    # st.sidebar.header("Configuration")
    # components.iframe(
    #     "https://jupyterlite.github.io/demo/repl/index.html?kernel=python&toolbar=1",
    #     height=500
    # )

    st.write("https://github.com/ooihiangee/Econometric-Modelling-of-USD-MYR")
    st.markdown("""
        The github repostory contains the source codes for the following:
        a. Data Collection
        b. Data Preprocessing
        c. Data Cleaning
        d. Data Transformation
        e. Data Exploration
        f. Data Visualization
        g. Data Modelling
        h. Streamlit App Development
        """)

    # nb = read_ipynb('C:/Users/ooihi/Downloads/machine learning for time series data in Python.ipynb')
    # nb.display()

###############################################################################################################################################

# UAT
if selected == "UAT":

    # Function to fit ARIMA and forecast for a single series
    # def fit_and_forecast(series, column_name):
    #     print(f"Processing {column_name}")
    #     model = auto_arima(series, seasonal=True, m=12, stepwise=True, suppress_warnings=True, error_action='ignore')
    #     forecast = model.predict(n_periods=12)  # Generate 12-month forecast
    #     return column_name, forecast

    # # Parallel processing to fit ARIMA models and forecast
    # results = Parallel(n_jobs=-1)(
    #     delayed(fit_and_forecast)(complete_df[col], col) for col in complete_df.columns
    # )

    # # Convert the results into a DataFrame
    # forecast_dict = {col: forecast for col, forecast in results}
    # forecast_df = pd.DataFrame(forecast_dict, index=pd.date_range(
    #     start=complete_df.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq='MS'
    # ))

    # # Load the saved Random Forest model and feature selector
    # model = joblib.load("best_lgb_model.pkl")
    # selector = joblib.load("best_lgb_features.pkl")

    # # Streamlit app title
    # st.title("Exchange Rate Forecasting App")

    # # User input form
    # st.header("Enter Macro-Economic Variables")
    # num_features = selector.get_support(indices=True).shape[0]

    # # Example: Assuming 5 macroeconomic features
    # feature_inputs = []
    # for i in range(num_features):
    #     feature_value = st.number_input(f"Feature {i+1}", min_value=0.0, max_value=10000.0, step=0.1)
    #     feature_inputs.append(feature_value)

    # # Predict button
    # if st.button("Forecast Exchange Rate"):
    #     # Convert user inputs to a NumPy array and reshape for prediction
    #     user_input_array = np.array(feature_inputs).reshape(1, -1)
        
    #     # Transform the input using the selector
    #     transformed_input = selector.transform(user_input_array)
        
    #     # Predict the exchange rate
    #     predicted_er = model.predict(transformed_input)
        
    #     # Display the result
    #     st.subheader("Forecasted Exchange Rate")
    #     st.write(f"The predicted exchange rate is: {predicted_er[0]:.2f}")

    # st.sidebar.title("Navigation")
    # options = st.sidebar.radio("Select a section:", ["Upload Data", "Data Preprocessing", "Modeling", "Evaluation", "Forecasting"])

    # if options == "Upload Data":
    #     st.title("Upload Data")
    #     uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])
    #     if uploaded_file:
    #         data = pd.read_csv(uploaded_file)
    #         st.write("Dataset Preview:")
    #         st.write(data)

    # elif options == "Data Preprocessing":
    #     st.title("Data Preprocessing")
    #     if 'data' in locals():
    #         st.write("Dataset Preview:")
    #         st.write(data)
    #         # Stationarity tests, scaling, and lagging options
    #         st.write("Perform stationarity tests or preprocess data.")
    #     else:
    #         st.error("Upload a dataset first.")

    # elif options == "Modeling":
    #     st.title("Modeling")
    #     if 'data' in locals():
    #         st.write("Select features and target variable.")
    #         X = st.multiselect("Select independent variables:", data.columns)
    #         y = st.selectbox("Select dependent variable:", data.columns)
    #         if X and y:
    #             X_train, X_test, y_train, y_test = train_test_split(data[X], data[y], test_size=0.2, random_state=42)
                
    #             model_choice = st.selectbox("Choose a model:", ["Random Forest", "XGBoost"])
    #             if model_choice == "Random Forest":
    #                 n_estimators = st.slider("Number of trees:", 10, 500, 100)
    #                 model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    #             elif model_choice == "XGBoost":
    #                 learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1)
    #                 model = XGBRegressor(learning_rate=learning_rate, random_state=42)

    #             if st.button("Train Model"):
    #                 model.fit(X_train, y_train)
    #                 st.success("Model trained successfully!")

    # elif options == "Evaluation":
    #     st.title("Evaluation")
    #     if 'model' in locals() and 'X_test' in locals():
    #         predictions = model.predict(X_test)
    #         st.write("R-squared:", r2_score(y_test, predictions))
    #         st.write("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))

    #         # Plot actual vs predicted
    #         fig, ax = plt.subplots()
    #         ax.scatter(y_test, predictions)
    #         ax.set_xlabel("Actual")
    #         ax.set_ylabel("Predicted")
    #         st.pyplot(fig)

    # elif options == "Forecasting":
    #     st.title("Forecasting")
    #     # Include options for new data input and prediction
    #     st.write("Coming Soon!")

    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    st.title('Stock Forecast App')

    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
    selected_stock = st.selectbox('Select dataset for prediction', stocks)

    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    @st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')

    st.subheader('Raw data')
    st.write(data.tail())

    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
        
    plot_raw_data()

    # Predict forecast with Prophet.
    # df_train = data[['Date','Close']]
    # df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    # m = Prophet()
    # m.fit(df_train)
    # future = m.make_future_dataframe(periods=period)
    # forecast = m.predict(future)

    # # Show and plot forecast
    # st.subheader('Forecast data')
    # st.write(forecast.tail())
        
    # st.write(f'Forecast plot for {n_years} years')
    # fig1 = plot_plotly(m, forecast)
    # st.plotly_chart(fig1)

    # st.write("Forecast components")
    # fig2 = m.plot_components(forecast)
    # st.write(fig2)