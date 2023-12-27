import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
from datetime import timedelta
import plotly.express as px
from urllib.request import urlopen
import json
sns.set(style='dark')

# Define function to create specific dataframe
def create_monthly_orders_df(df):
    latest_month = pd.Timestamp(df['order_purchase_timestamp'].max())
    twelve_months_before = latest_month - pd.DateOffset(months=12)

    df_last_12_month = df[(df['order_purchase_timestamp'] <= latest_month) & (df['order_purchase_timestamp'] >= twelve_months_before)]

    # Resample dataframe based on month
    monthly_orders_df = df_last_12_month.resample(rule='M', on='order_purchase_timestamp').agg({
        "order_id": "nunique",
        "price": "sum"
    })
    monthly_orders_df.index = monthly_orders_df.index.strftime('%Y-%m')
    monthly_orders_df = monthly_orders_df.reset_index()
    monthly_orders_df.rename(columns={
        "order_id": "num_of_order",
        "price": "revenue"
    }, inplace=True)

    return monthly_orders_df

def map_preparation(df):
    df = df.dropna(subset=['geolocation_lat_x', 'geolocation_lng_x'])
    df['customer_zip_code_prefix'] = df['customer_zip_code_prefix'].astype(str)

    # Load brazil geojson
    with urlopen('https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson') as response:
        Brazil = json.load(response)

    state_codes = ['SP', 'BA', 'GO', 'RN', 'PR', 'RS', 'RJ', 'MG', 'SC', 'RR', 'PE',
               'TO', 'CE', 'DF', 'SE', 'MT', 'PB', 'PA', 'RO', 'ES', 'AP', 'MS',
               'MA', 'PI', 'AL', 'AC', 'AM']

    state_names = ['São Paulo', 'Bahia', 'Goiás', 'Rio Grande do Norte', 'Paraná', 'Rio Grande do Sul',
               'Rio de Janeiro', 'Minas Gerais', 'Santa Catarina', 'Roraima', 'Pernambuco',
               'Tocantins', 'Ceará', 'Distrito Federal', 'Sergipe', 'Mato Grosso', 'Paraíba',
               'Pará', 'Rondônia', 'Espírito Santo', 'Amapá', 'Mato Grosso do Sul',
               'Maranhão', 'Piauí', 'Alagoas', 'Acre', 'Amazonas']

    # Create a dictionary mapping state codes to state names
    state_dict = dict(zip(state_codes, state_names))

    # Map the state codes to their corresponding full names
    mapped_state_names = [state_dict[code] for code in state_codes]

    state_dict = dict(zip(state_codes, state_names))

    df['customer_state_name'] = df['customer_state'].map(state_dict)
    df['seller_state_name'] = df['seller_state'].map(state_dict)

    state_id_map = {}
    for feature in Brazil ['features']:
        feature['id'] = feature['properties']['name']
        state_id_map[feature['properties']['sigla']] = feature['id']

    return df, Brazil

    

def create_df_state_price_cust_means(df):
    state_price_cust_means = df.groupby(by='customer_state_name')['price'].mean().reset_index()
    
    return state_price_cust_means

def create_df_payment_percentage(df):
    unique_payment_types = df["payment_type"].unique()

    payment_type_mapping = {payment_type: index + 1 for index, payment_type in enumerate(unique_payment_types)}

    # Map the payment types to numerical values
    df["payment_type_numeric"] = df["payment_type"].map(payment_type_mapping)

    # Create dataframe which calculate number of orders for each state
    num_of_orders = df.groupby(by=['customer_state_name'])['order_id'].nunique().reset_index()

    payment_method = df.groupby(by=['customer_state_name', 'payment_type'])['order_id'].nunique().reset_index()

    payment_method_max = payment_method.loc[payment_method.groupby('customer_state_name')['order_id'].idxmax()]

    # Join table num_of_orders and payment_method to get percentage of credit card usage overall all payment methods

    df_payment_percentage = pd.merge(
        left=payment_method_max,
        right=num_of_orders,
        how="left",
        left_on="customer_state_name",
        right_on="customer_state_name"
    )

    # Calculate percentage value
    df_payment_percentage["credit_card_percentage"] = df_payment_percentage["order_id_x"]*100/df_payment_percentage["order_id_y"]

    return df_payment_percentage

def create_df_seller_performance(df):
    df_seller_performance = df.groupby(by="seller_id_short").agg({
    "product_id":"nunique",
    "review_score":"mean",
    "approved_time_m": "mean",
    "price":"sum"
    })
    return df_seller_performance
    
def create_df_revenue_cat_filtered(df, num_of_product):
    # Create dataframe group by product category and distance group
    distance_group_product_category = df.groupby(by=["product_category_name_english","distance_group"]).agg({
        "price": "sum",
        "order_id":"nunique"}).sort_values(by="order_id", ascending=False).reset_index()

    # Create dataframe group only by product category
    product_category_group = df.groupby(by="product_category_name_english").agg({
        "price": "sum",
        "order_id":"nunique"}).sort_values(by="order_id", ascending=False).reset_index()

    # Merge both grouped category to sort based on top sold products
    proportion_distance_group = pd.merge(
        left=distance_group_product_category,
        right=product_category_group,
        how="left",
        left_on="product_category_name_english",
        right_on="product_category_name_english"
    )

    proportion_distance_group =  proportion_distance_group.nlargest(num_of_product*3, 'order_id_y')

    # Pivot the DataFrame and sort based on number of solds 
    pivot_num_orders = proportion_distance_group.pivot_table(index='product_category_name_english', columns='distance_group', values='order_id_x', aggfunc='sum', fill_value=0).reset_index()
    pivot_num_orders["Sum"] = pivot_num_orders["Long"] + pivot_num_orders["Medium"] + pivot_num_orders["Short"]
    pivot_num_orders = pivot_num_orders.sort_values(by="Sum", ascending=False)
    category_sequence = pivot_num_orders['product_category_name_english'] # Take the sequence order in purpose to sorting index on next graph
    pivot_num_orders.drop(columns="Sum", inplace=True)

    # Creating dataframe consisting revenue generated each product category and filtered to have same categories as pivot_num_orders dataframe
    df_revenue_category = df_all.groupby(by='product_category_name_english')['price'].sum().reset_index()
    df_revenue_cat_filtered = df_revenue_category[df_revenue_category.product_category_name_english.isin(pivot_num_orders.product_category_name_english)]

    return pivot_num_orders, df_revenue_cat_filtered


def create_review_corr(df):
    columns_of_interest_1 = [ "product_photos_qty", "product_description_lenght", "delivery_time", "approved_time_m","review_answer_time_h", "review_score" ]
    df_review_corr = df[columns_of_interest_1]

    # Calculate the correlation matrix
    review_corr = df_review_corr.corr()

    return review_corr

def create_freight_value_corr(df):
    columns_of_interest_2 = ['product_volume_cm3', 'product_weight_g', 'distance_seller_customer_km', 'price', 'freight_value', 'delivery_time']
    df_freight_value_corr = df[columns_of_interest_2]

    # Calculate the correlation matrix
    freight_value_corr = df_freight_value_corr .corr()

    return freight_value_corr

def create_df_rfm(df, max_date):
    df_rfm = df.groupby(by="customer_unique_id_short", as_index=False).agg({
    "order_purchase_timestamp": "max", 
    "order_id": "nunique",
    "price": "sum" 
    })
    df_rfm.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]
    
    # menghitung kapan terakhir pelanggan melakukan transaksi (hari)
    df_rfm["max_order_timestamp"] = df_rfm["max_order_timestamp"].dt.date
    recent_date = df["order_purchase_timestamp"].dt.date.max()
    df_rfm["recency"] = df_rfm["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    
    df_rfm.drop("max_order_timestamp", axis=1, inplace=True)

    return df_rfm

# Prepare dataframe
df_all = pd.read_csv("all_data_e_commerce.csv") 
 #Trim the unique id to be shorten
df_all['customer_unique_id_short'] = df_all['customer_unique_id'].str[-8:]

# Change datatype of certain columns to datetime
datetime_columns = ['order_purchase_timestamp','order_approved_at', 'order_delivered_carrier_date',
                     'order_delivered_customer_date', 'order_estimated_delivery_date', 'shipping_limit_date',
                     'review_creation_date','review_answer_timestamp']

df_all.sort_values(by="order_purchase_timestamp", inplace=True)
df_all.reset_index(inplace=True)
 
for column in datetime_columns:
    df_all[column] = pd.to_datetime(df_all[column], format='mixed')

# Create date filter
min_date = df_all["order_purchase_timestamp"].min()
max_date = df_all["order_purchase_timestamp"].max()
# Calculate 1 year before max_date
one_year_before_max_date = max_date - timedelta(days=365)
 
with st.sidebar:
    # Set default value 1 year from latest date:
    start_date, end_date = st.date_input(
        label='Pick time range: ',
        min_value=min_date,
        max_value=max_date,
        value=[one_year_before_max_date, max_date])
    
    num_of_seller = st.number_input("Show how many seller(s): ", min_value=1, max_value=10, value=5)

    num_of_product = st.number_input("Show how many top-selling product(s): ", min_value=1, max_value=20, value=10)

    num_of_cust = st.number_input("Show how many top RFM customer(s): ", min_value=1, max_value=10, value=5)

# Because the date format containing time value, the start date will be reduce by 1 day, while the end date will be increased by 1 day
start_date = start_date - timedelta(days=1)
end_date = end_date + timedelta(days=1)
main_df = df_all[(df_all["order_purchase_timestamp"] >= str(start_date)) & 
                (df_all["order_purchase_timestamp"] <= str(end_date))]

# Create all necessary dataframe
monthly_orders_df = create_monthly_orders_df(main_df)
df_state_price_cust_means = create_df_state_price_cust_means(main_df)
df_map, Brazil = map_preparation(main_df)
state_price_cust_means = create_df_state_price_cust_means(df_map)
df_payment_percentage = create_df_payment_percentage(df_map)
df_seller_performance = create_df_seller_performance(main_df)
df_by_price = df_seller_performance.sort_values(by="price", ascending=False).head(num_of_seller).reset_index()
df_by_approved_time = df_seller_performance.sort_values(by="approved_time_m", ascending=True).head(num_of_seller).reset_index()
df_by_review = df_seller_performance.sort_values(by="review_score", ascending=False).head(num_of_seller).reset_index()
pivot_num_orders, df_revenue_cat_filtered = create_df_revenue_cat_filtered(main_df, num_of_product)
df_rfm = create_df_rfm(main_df, max_date)
df_top_recency = df_rfm.sort_values(by="recency", ascending=True).head(num_of_cust)
df_top_frequency = df_top_frequency = df_rfm.sort_values(by="frequency", ascending=False).head(num_of_cust)
df_top_monetary = df_rfm.sort_values(by="monetary", ascending=False).head(num_of_cust)
review_corr = create_review_corr(main_df)
freight_value_corr = create_freight_value_corr(main_df)


tab1, tab2 = st.tabs(["General Tab", "Correlation Tab"])
 
with tab1:
    st.header('Brazilian E-Commerce Dashboard 🛍️')
    st.markdown("Dataset source: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)")
    st.write('')
    st.write('Latest date can be retrieved: ', main_df["order_purchase_timestamp"].max())
    
    st.markdown("---") 

    # Revenue generated graph
    st.subheader('Monthly Orders and Incoming Money')

    # Cards
    col1, col2 = st.columns(2)
    
    with col1:
        total_orders = monthly_orders_df.num_of_order.sum()
        st.metric("Total orders", value=total_orders)
    
    with col2:
        total_revenue = format_currency(monthly_orders_df.revenue.sum(), 'BRL', locale='pt_BR') 
        st.metric("Total Revenue", value=total_revenue)

    # Line chart
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Plot the first dataset on the first y-axis
    ax1.plot(monthly_orders_df["order_purchase_timestamp"], monthly_orders_df["num_of_order"], marker='o', linewidth=2, color='b', label='Num of orders')
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Num of orders', color='b')
    ax1.tick_params('y', colors='b')

    # Create a second y-axis
    ax2 = ax1.twinx()

    # Plot the second dataset on the second y-axis
    ax2.plot(monthly_orders_df["order_purchase_timestamp"], monthly_orders_df["revenue"], marker='o', linewidth=2, color='r', label='Incoming Money')
    ax2.set_ylabel('Incoming Money', color='r')
    ax2.tick_params('y', colors='r')

    plt.title("Number of Orders and Incoming Money per Month", loc="center", fontsize=14) 
    plt.grid()
    plt.xticks(fontsize=10) 
    plt.yticks(fontsize=10) 

    st.pyplot(fig)

    st.markdown("---") 

    # Plot Spending Map
    st.subheader("Spending Demography")
    range_cust_min = state_price_cust_means["price"].min()
    range_cust_max = state_price_cust_means["price"].max()

    fig = px.choropleth(state_price_cust_means, 
                    geojson=Brazil, 
                    locations='customer_state_name', 
                    color='price',
                    color_continuous_scale="Viridis",
                    range_color=(range_cust_min, range_cust_max),
                    scope="south america",
                    labels={'Spending':'price'}
                    )
    fig.update_layout(margin={"r":0, "t":0, "l":0, "b":0})

    st.plotly_chart(fig)

    st.markdown("---") 

    # Plot Most Popular Payment Type Demography
    st.subheader("Most Popular Payment Type Demography")
    range_pp_min = df_payment_percentage["credit_card_percentage"].min()
    range_pp_max = df_payment_percentage["credit_card_percentage"].max()

    fig = px.choropleth(df_payment_percentage, 
                    geojson=Brazil, 
                    locations='customer_state_name', 
                    color='credit_card_percentage',
                    color_continuous_scale="Viridis",
                    range_color=(range_pp_min, range_pp_max),
                    scope="south america",
                    labels={'credit_card_percentage':'credit_card_percentage'}
                    )
    fig.update_layout(margin={"r":0, "t":0, "l":0, "b":0})
    st.plotly_chart(fig)

    st.markdown("---") 
    # Plot seller performance
    st.subheader("Seller Performance")

    # Create long seller id legend to be print out - by price
    seller_by_price_legends = []

    for seller in df_by_price['seller_id_short']:
        seller_legend = seller + ' : ' + main_df[main_df['seller_id_short'] == seller]['seller_id'].astype(str).iloc[0]
        seller_by_price_legends.append(seller_legend)

    # Create long seller id legend to be print out - by review
    seller_by_review_legends = []

    for seller in df_by_review['seller_id_short']:
        seller_legend = seller + ' : ' + main_df[main_df['seller_id_short'] == seller]['seller_id'].astype(str).iloc[0]
        seller_by_review_legends.append(seller_legend)

    # Create long seller id legend to be print out - by approved time
    seller_by_app_legends = []

    for seller in df_by_approved_time['seller_id_short']:
        seller_legend = seller + ' : ' + main_df[main_df['seller_id_short'] == seller]['seller_id'].astype(str).iloc[0]
        seller_by_app_legends.append(seller_legend)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(40, 8))
    
    colors = ["#72BCD4", "#72BCD4", "#72BCD4", "#72BCD4", "#72BCD4"]
    
    sns.barplot(y="price", x="seller_id_short", data=df_by_price, hue="seller_id_short", palette=colors, ax=ax[0])
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[0].set_title("By Revenue Generated", loc="center", fontsize=18)
    ax[0].tick_params(axis ='x', labelsize=15)
    ax[0].legend(seller_by_price_legends, loc='lower left')

    sns.barplot(y="review_score", x="seller_id_short", data=df_by_review,hue="seller_id_short", palette=colors, ax=ax[1])
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].set_title("By Review Score", loc="center", fontsize=18)
    ax[1].tick_params(axis='x', labelsize=15)
    ax[1].legend(seller_by_review_legends, loc='lower left')
    
    sns.barplot(y="approved_time_m", x="seller_id_short", data=df_by_approved_time, hue="seller_id_short", palette=colors, ax=ax[2])
    ax[2].set_ylabel("Approved Time (Mins)")
    ax[2].set_xlabel(None)
    ax[2].set_title("By Approved Time", loc="center", fontsize=18)
    ax[2].tick_params(axis='x', labelsize=15)
    ax[2].legend(seller_by_app_legends, loc='lower left')

    for ax_ in ax:
        ax_.tick_params(axis='x', labelrotation=45)
        
    plt.suptitle("Best Seller Based on Several Parameters", fontsize=20)
    st.pyplot(fig)

    st.markdown("---") 

    st.subheader('Top-Selling Product Categories Based on Number of Orders')


    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot the first bar chart using Pandas DataFrame plot function on the first y-axis
    pivot_num_orders.plot(x="product_category_name_english", kind='bar', stacked=True, ax=ax1)
    ax1.set_xlabel('Product Category')
    ax1.set_ylabel('Number of Orders', color='b')
    ax1.set_xticklabels(pivot_num_orders["product_category_name_english"], rotation=45, ha='right')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # Create second y-axis
    ax2 = ax1.twinx()

    ax2.plot(df_revenue_cat_filtered["product_category_name_english"], df_revenue_cat_filtered["price"], marker='o', linewidth=2, color='r', label='Revenue')
    ax2.set_ylabel('Revenue', color='r')

    # Show the plot
    plt.title('Top-Selling Product Categories Based on Number of Orders')
    st.pyplot(fig)

    st.markdown("---") 
    # Plot best customer based on RFM parameters
    st.subheader('Best Customer Based on RFM Parameters')

    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_recency = round(df_rfm.recency.mean(), 1)
        st.metric("Average Recency (days)", value=avg_recency)
    
    with col2:
        avg_frequency = round(df_rfm.frequency.mean(), 2)
        st.metric("Average Frequency", value=avg_frequency)
    
    with col3:
        avg_frequency = format_currency(df_rfm.monetary.mean(), 'BRL', locale='pt_BR') 
        st.metric("Average Monetary", value=avg_frequency)

    by_recency_legends = []
    for cust in df_top_recency['customer_id']:
        by_recency_legend = cust + ' : ' + main_df[main_df['customer_unique_id_short'] == cust]['customer_id'].astype(str).iloc[0]
        by_recency_legends.append(by_recency_legend)

    by_frequency_legends = []
    for cust in df_top_frequency['customer_id']:
        by_frequency_legend = cust + ' : ' + main_df[main_df['customer_unique_id_short'] == cust]['customer_id'].astype(str).iloc[0]
        by_frequency_legends.append(by_frequency_legend)

    by_monetary_legends = []
    for cust in df_top_monetary['customer_id']:
        by_monetary_legend = cust + ' : ' + main_df[main_df['customer_unique_id_short'] == cust]['customer_id'].astype(str).iloc[0]
        by_monetary_legends.append(by_monetary_legend)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 6))

    colors = ["#72BCD4", "#72BCD4", "#72BCD4", "#72BCD4", "#72BCD4"]

    sns.barplot(y="recency", x="customer_id", data=df_top_recency,hue="customer_id", palette=colors, ax=ax[0])
    ax[0].set_ylabel(None)
    ax[0].set_xlabel('Customer ID Short')
    ax[0].set_title("By Recency (days)", loc="center", fontsize=18)
    ax[0].tick_params(axis ='x', labelsize=15)
    ax[0].legend(by_recency_legends)

    sns.barplot(y="frequency", x="customer_id", data=df_top_frequency,hue="customer_id", palette=colors, ax=ax[1])
    ax[1].set_ylabel(None)
    ax[1].set_xlabel('Customer ID Short')
    ax[1].set_title("By Frequency", loc="center", fontsize=18)
    ax[1].tick_params(axis='x', labelsize=15)
    ax[1].legend(by_frequency_legends)

    sns.barplot(y="monetary", x="customer_id", data=df_top_monetary,hue="customer_id", palette=colors, ax=ax[2])
    ax[2].set_ylabel(None)
    ax[2].set_xlabel('Customer ID Short')
    ax[2].set_title("By Monetary", loc="center", fontsize=18)
    ax[2].tick_params(axis='x', labelsize=15)
    ax[2].legend(by_monetary_legends)
    
    plt.suptitle("Best Customer Based on RFM Parameters", fontsize=20)
    st.pyplot(fig)

 
with tab2:
    st.header('Brazilian E-Commerce Dashboard 🛍️')
    st.markdown("Dataset source: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)")
    st.write('')
    st.write('Latest date can be retrieved: ', main_df["order_purchase_timestamp"].max())
    
    st.markdown("---") 
    

    st.subheader('Product Specifications Correlation Heatmap')
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(freight_value_corr, cmap='viridis', annot=True, fmt='.2f', linewidths=.5, ax=ax)
    plt.title('Product Specifications Correlation Heatmap')
    st.pyplot(fig)

    st.markdown("---") 

    st.subheader('Product Review Correlation Heatmap')
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(review_corr, cmap='viridis', annot=True, fmt='.2f', linewidths=.5)
    plt.title('Product Review Correlation Heatmap')
    st.pyplot(fig)







