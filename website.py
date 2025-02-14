import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st
import re
st.set_page_config(layout='wide', page_title='Startup Analysis')

df = pd.read_csv('startup_cleaned12.csv')

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

city_replacements = {
    'Bengaluru': 'Bangalore',
    'Gurugram': 'Gurgaon',
    'Ahmedabad': 'Ahemdabad',
    'Ahemadabad': 'Ahemdabad',
    'Bhubneswar': 'Bhubaneswar'
}

df['city'] = df['city'].replace(city_replacements)

df = df.dropna(subset=['investor'])

# Function to standardize variations of Ecommerce
def standardize_vertical(value):
    if re.search(r"e-?commerce", value, re.IGNORECASE):
        return 'Ecommerce'
    return value

# Apply the function to the 'vertical' column
df['vertical'] = df['vertical'].apply(standardize_vertical)


def clean_investor_names(name):
    # Remove unwanted characters like \xe2\x80\x99s and any others that don't fit common name patterns
    name = re.sub(r'\\x[a-fA-F0-9]{2}', '', name)
    name = re.sub(r"’s", "", name)
    name = name.replace("’", "")
    return name.strip()

# Apply the function to the investor column
df['investor'] = df['investor'].apply(clean_investor_names)

from sklearn.neighbors import NearestNeighbors

def recommend_investors(startup_name, n_recommendations=5):
    # Prepare the investor data matrix for collaborative filtering
    investor_df = df.assign(investor=df['investor'].str.split(',')).explode('investor')
    investor_df['investor'] = investor_df['investor'].str.strip()

#pivot table
    startup_investor_matrix = investor_df.pivot_table(index='startup', columns='investor', values='amount', fill_value=0)


    if startup_name not in startup_investor_matrix.index:
        return ["Startup not found in the dataset."]

    # Use K-Nearest Neighbors for recommendation
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(startup_investor_matrix)


    startup_index = startup_investor_matrix.index.get_loc(startup_name)

    # Reshape the input startup row to fit the model's input requirements
    startup_data = startup_investor_matrix.iloc[startup_index, :].values.reshape(1, -1)

    # nearest neighbors for the given startup
    distances, indices = model_knn.kneighbors(startup_data, n_neighbors=n_recommendations + 1)


    recommended_indices = indices.flatten()[1:]

    # recommended investors
    recommended_startups = startup_investor_matrix.index[recommended_indices]
    recommended_investors = set(investor_df[investor_df['startup'].isin(recommended_startups)]['investor'].unique())

    # Exclude investors who already invested in the given startup
    known_investors = set(investor_df[investor_df['startup'] == startup_name]['investor'].unique())
    final_recommendations = list(recommended_investors - known_investors)

    return final_recommendations[:n_recommendations]

def display_recommendations(recommendations):
    st.markdown("<h1 style='text-align: center;text-decoration: underline;'>Recommended Investors</h1>", unsafe_allow_html=True)
    if recommendations:
        # Display each recommended investor in a list format
        st.markdown("<h2 style='font-size: 24px;'>Here are the top recommended investors:</h2>", unsafe_allow_html=True)
        st.markdown(
            "<ul style='list-style-type: disc; padding-left: 20px;'>"
            + "".join([f"<li style='font-size: 1.7em; color: white;'>{investor}</li>" for investor in recommendations])
            + "</ul>",
            unsafe_allow_html=True
        )
    else:
        st.write("No recommendations found for this startup.")

def display_recommendation_paragraph():
    st.markdown("""
        <div style="padding: 17px; border-radius: 10px; margin-top: 20px;">
            <p style="font-size: 1.4em; color: white;">
                These investor recommendations are tailored based on an in-depth analysis of historical funding patterns 
                and sectoral preferences. By leveraging machine learning and collaborative filtering, we have identified 
                investors who align closely with the strategic needs and growth potential of your startup. Each suggested 
                investor has demonstrated a commitment to ventures in similar industries, making them well-suited to 
                support your vision. Connecting with the right investors is a critical step toward transforming ideas into 
                impact, and we hope this curated list brings you closer to partnerships that drive innovation and success.
            </p>
        </div>
    """, unsafe_allow_html=True)

@st.cache_data
def train_model():
    # Load data
    df = pd.read_csv('startup_cleaned12.csv')

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df = df.dropna(subset=['amount', 'city', 'vertical', 'round'])
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['amount'] = np.log1p(df['amount'])  # log(1 + amount)

    # Features and target
    X = df[['vertical', 'city', 'round', 'year']]
    y = df['amount']

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['vertical', 'city', 'round']),
            ('num', StandardScaler(), ['year'])
        ]
    )


    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(objective='reg:squarederror',
    max_depth=4,
    learning_rate=0.05,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    alpha=0.5,  # L1 regularization term
    lambda_=0.5,  # L2 regularization term
    random_state=42))
    ])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)


    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }

    return pipeline, metrics


def load_overall_analysis():
    st.title("Overall Analysis")
    col1, col2, col3, col4 = st.columns(4)

    # Total amount invested
    with col1:
        total = round(df['amount'].sum())
        st.metric('Total Amount Invested till date', str(total) + ' Cr')
    with col2:
        max_funding = df.groupby('startup')['amount'].max().sort_values(ascending=False).head(1).values[0]
        st.metric('Maximum amount invested', str(max_funding) + ' Cr')
    # Average amount invested
    with col3:
        avg_funding = df.groupby('startup')['amount'].sum().mean()
        st.metric('Average amount invested', str(round(avg_funding)) + ' Cr')
    with col4:
        total_funded = df['startup'].nunique()
        st.metric('Approximate Funded Startups', total_funded)


    st.markdown("---")

    st.header('Month On Month Graph')
    selected_option = st.selectbox("Select Type", ["Total Amount Invested", "Total Startups Funded"])
    if selected_option == "Total Amount Invested":
        temp_df = df.groupby(['year', 'month'])['amount'].sum().reset_index()
    else:
        temp_df = df.groupby(['year', 'month'])['amount'].count().reset_index()
    temp_df['x_axis'] = temp_df['month'].astype('str') + '-' + temp_df['year'].astype('str')
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    ax5.plot(temp_df['x_axis'], temp_df['amount'], marker='o', color='skyblue', linewidth=2)
    ax5.set_xticks(range(len(temp_df['x_axis'])))
    ax5.set_xticklabels(temp_df['x_axis'], rotation=90, ha='right', fontsize=7)
    ax5.set_xlabel('Month-Year', fontsize=10)
    ax5.set_ylabel(' Amount/ Startup', fontsize=10)
    st.pyplot(fig5)
    ax5.set_title('Amount vs Month-Year', fontsize=12)

    st.markdown("---")

    st.header('Section-Wise Analysis')

    sector_count = df['vertical'].value_counts().head(5)  # Top sectors by count
    sector_sum = df.groupby('vertical')['amount'].sum().nlargest(5)  # Top sectors by sum

    def plot_pie_chart(selected_option):
        if selected_option == 'Highest number of sectors invested':
            data = sector_count
            title = 'Top 5 Sectors by Count'
        else:
            data = sector_sum
            title = 'Top 5 Sectors by Amount'


        fig6, ax6 = plt.subplots(figsize=(7, 7))
        ax6.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90)
        ax6.set_title(title)


        ax6.axis('equal')


        st.pyplot(fig6)


    selected_option = st.selectbox('Select Option', ['Highest number of sectors invested', 'Highest amount invested'])


    plot_pie_chart(selected_option)

    st.markdown("---")

    st.header('City-Wise Analysis')

    city_count = df['city'].value_counts().head(10)  # Top 10 cities by count
    city_sum = df.groupby('city')['amount'].sum().nlargest(10)  # Top 10 cities by sum


    def plot_city_wise_funding(selected_option):
        if selected_option == 'Highest number of investment(city-wise)':
            data = city_count
            title = 'Top Cities by Number of Investments'
            ylabel = 'Number of Investments'
        else:
            data = city_sum
            title = 'Top Cities by Total Investment Amount'
            ylabel = 'Total Investment Amount'


        fig7, ax7 = plt.subplots(figsize=(10, 6))
        ax7.bar(data.index, data.values, color='skyblue')
        ax7.set_title(title)
        ax7.set_xlabel('City')
        ax7.set_ylabel(ylabel)
        plt.xticks(rotation=45, ha='right')


        st.pyplot(fig7)


    selected_option = st.selectbox('Select Option for City-wise Funding', ['Highest number of investment(city-wise)',
                                                                           'Highest amount of money invested(city-wise)'])


    plot_city_wise_funding(selected_option)

    st.markdown("---")

    st.header('Top Investors')


    top_investors = df.groupby('investor')['amount'].sum().nlargest(10)  # Top 10 investors by amount
    def plot_top_investors():
        # Create the bar chart
        fig8, ax8 = plt.subplots(figsize=(10, 6))
        ax8.bar(top_investors.index, top_investors.values, color='orange')
        ax8.set_title('Top Investors by Total Investment Amount')
        ax8.set_xlabel('Investor')
        ax8.set_ylabel('Total Investment Amount (Cr)')
        plt.xticks(rotation=45, ha='right')


        st.pyplot(fig8)

    plot_top_investors()

    st.markdown("---")

    st.header('Funding Types Chart')

    def load_funding_type_chart():

        # Aggregate the data by funding type
        funding_type_data = df.groupby('round')['amount'].sum().reset_index().sort_values(by='amount',
                                                                                          ascending=False).head(15)


        fig9, ax9 = plt.subplots(figsize=(12, 8))
        ax9.bar(funding_type_data['round'], funding_type_data['amount'], color='skyblue')


        ax9.set_xlabel('Funding Type', fontsize=12)
        ax9.set_ylabel('Total Amount Invested (Cr)', fontsize=12)
        ax9.set_title('Total Amount Invested by Funding Type', fontsize=16)


        plt.xticks(rotation=45, ha='right')

        st.pyplot(fig9)

    load_funding_type_chart()


def load_investor_detail(investor):
    st.title(investor)
    last5_df = df[df['investor'].str.contains(investor)].head()[
        ['date', 'startup', 'vertical', 'city', 'round', 'amount']]
    st.subheader('Most Recent Investments of the Company')
    st.dataframe(last5_df)
    col1, col2 = st.columns(2)
    with col1:
        big_df = df[df['investor'].str.contains(investor)].groupby('startup')['amount'].sum().sort_values(
            ascending=False).head()
        st.subheader('Top 5 Investments of the Company')
        fig, ax = plt.subplots()
        ax.bar(big_df.index, big_df.values,color="red")
        st.pyplot(fig)
    with col2:
        ver_series = df[df['investor'].str.contains(investor)].groupby('vertical')['amount'].sum().head(5)
        st.subheader('Sector Wise Investment Chart (Top 5)')
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.pie(ver_series, labels=ver_series.index, autopct="%0.01f%%")
        st.pyplot(fig1)
    col3, col4 = st.columns(2)
    with col3:
        round_series = df[df['investor'].str.contains(investor)].groupby('round')['amount'].sum()
        st.subheader('Round Wise Investment Chart')
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.pie(round_series, labels=round_series.index, autopct="%0.01f%%")
        st.pyplot(fig2)
    with col4:
        city_series = df[df['investor'].str.contains(investor)].groupby('city')['amount'].sum().head(5)
        st.subheader('City Wise Investment Chart (Top 5)')
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        ax3.pie(city_series, labels=city_series.index, autopct="%0.01f%%")
        st.pyplot(fig3)

    df['year'] = df['date'].dt.year


    investor_filtered_df = df[df['investor'].str.contains(investor)]
    year_series = investor_filtered_df.groupby('year')['amount'].sum()


    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.plot(year_series.index, year_series.values, marker='o', color='skyblue', linewidth=2, linestyle='-')


    ax4.set_xlabel("Year")
    ax4.set_ylabel("Amount in CR")
    ax4.set_title("Year-wise Investment")


    ax4.set_xticks(year_series.index)
    ax4.set_xticklabels(year_series.index, rotation=45, ha='right')
    ax4.grid(True, linestyle='-', alpha=0.7)
    st.subheader('Year Wise Investment Chart')
    st.pyplot(fig4)


st.sidebar.title("Startup Funding Analysis")
option = st.sidebar.selectbox("Enter your choice", ["Overall Analysis", "Startup", "Investor", "Recommend Investors","Funding Amount Prediction"])

if option == "Overall Analysis":
    load_overall_analysis()

elif option == "Startup":
    st.sidebar.selectbox("Select Startup", sorted(df['startup'].unique().tolist()))
    btn1 = st.sidebar.button("Find Startup Details")

elif option == "Investor":
    selected_investor = st.sidebar.selectbox("Select Investor", sorted(set(df['investor'].str.split(',').sum())))
    btn2 = st.sidebar.button("Find Investor Details")
    if btn2:
        load_investor_detail(selected_investor)

elif option == "Recommend Investors":
    st.sidebar.header("Investor Recommendations")
    startup_name = st.sidebar.selectbox("Select Startup", sorted(df['startup'].unique().tolist()))
    if st.sidebar.button("Recommend Investors"):
        recommendations = recommend_investors(startup_name)
        display_recommendations(recommendations)
        display_recommendation_paragraph()


elif option == "Funding Amount Prediction":
    st.title("Funding Amount Prediction")

    model, metrics = train_model()
    st.subheader("Model Performance")
    st.write(f"Mean Absolute Error (MAE): {metrics['MAE']:.2f}")
    st.write(f"R² Score: {metrics['R2']:.2f}")

    st.subheader("Enter Details to Predict Funding Amount")
    vertical = st.selectbox("Select Sector", sorted(df['vertical'].unique()))
    city = st.selectbox("Select City", sorted(df['city'].unique()))
    funding_round = st.selectbox("Select Funding Round", sorted(df['round'].unique()))
    year = st.number_input("Enter Year of Funding", min_value=int(df['year'].min()), max_value=int(df['year'].max()))


    if st.button("Predict Funding Amount"):

        user_input = pd.DataFrame({
            'vertical': [vertical],
            'city': [city],
            'round': [funding_round],
            'year': [year]
        })
        predicted_amount = model.predict(user_input)[0]
        st.success(f"Predicted Funding Amount: ₹{predicted_amount:.2f} Cr")
