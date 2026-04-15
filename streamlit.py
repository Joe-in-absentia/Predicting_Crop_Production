import pandas as pd 
import streamlit as st
import plotly.express as px
import joblib
from sqlalchemy import create_engine

# Page title and layout width.

st.set_page_config(page_title="Crop Production Analysis", layout="wide")

st.title("🌾 Crop Production Analysis")     # Main title.
st.markdown("---")

st.sidebar.title("🧭 Dashboard Controls")           # Sidebar title.
st.sidebar.markdown("---")

# Database connection.

engine = create_engine("postgresql://postgres:123456@localhost:5432/crop_data")

# Import the data from the database.

df = pd.read_sql("SELECT * FROM crop_data_cleaned", engine)

# Load the Trained model.

model = joblib.load("random_forest.pkl")
encoder = joblib.load("encoded_data.pkl")

# Sidebar filters.

years = st.sidebar.multiselect("Select Years",sorted(df["Year"].unique()),
         default=sorted(df["Year"].unique()))


# Filter dataframe based on selected years.
filtered_df = df[df["Year"].isin(years)]


st.sidebar.title("📊 Analysis Sections")

selectors = st.sidebar.radio('Choose an Data',[":red[Crop Production Prediction]","Region-wise Production",
                              "Crop Distribution","Outliers & Anomalies"]) 


def production():
    st.subheader("🔮 Predict Crop Production")           # For model prediction.

    # Inputs.
    region = st.selectbox("Select Region", df['Area'].unique())
    year = st.selectbox("Select Year", sorted(df['Year'].unique()))
    crop = st.selectbox("Select Item", df['Item'].unique())

    # Sliders.
    area_harvest = st.slider("Area Harvested (ha)",float(df["Area harvested"].min()),float(df["Area harvested"].max()))

    yield_value = st.slider("Yield (kg/ha)",float(df["Yield"].min()),float(df["Yield"].max()))

    # Prediction.
    if st.button("Predict Production"):

        # Encode the user inputs.
        area_encoded = encoder["Area"].transform([region])[0]
        item_encoded = encoder["Item"].transform([crop])[0]
     
        # Create dataframe.
        input_df = pd.DataFrame([{
            "Area": area_encoded,
            "Item": item_encoded,
            "Year": year,
            "Area harvested": area_harvest,
            "Yield": yield_value}])

        # Predict.
        prediction = model.predict(input_df)[0]

        # Output.
        st.success(
            f"📈 Predicted Production for {crop} in {region} ({year}): "
            f"**{prediction:,.2f} tons**")

def compare_region_production():
    st.subheader("🌍 Compare Region-wise Production")
    
    region_prod = filtered_df.groupby('Area')['Production'].sum().reset_index().sort_values(by='Production', ascending=False)
    
    fig = px.bar(
        region_prod.head(30), 
        x='Area', 
        y='Production',                                 # For region production.
        title="Top 30 Productive Regions",
        color='Production',
        color_continuous_scale='Blues')
    
    st.plotly_chart(fig, width='stretch')

def analyze_crop_distribution():
    st.subheader("🌿 Crop Distribution Analysis")
    
    crop_counts = filtered_df['Item'].value_counts().reset_index()
    crop_counts.columns = ['Item', 'count']
    
    fig = px.bar(
        crop_counts.head(30), 
        x='Item',                                        # For Crop distribution.
        y='count', 
        title="Top 30 Most Cultivated Crops",
        color='count',
        color_continuous_scale='Greens')
    
    st.plotly_chart(fig, width='stretch')


def analyze_outliers():
    st.subheader("⚠️ Outliers & Anomaly Detection")

    col1, col2 = st.columns(2)                          # For outlier detection.

    # Box Plot - Yield Outliers
    with col1:
        fig = px.box(filtered_df, y='Yield', title="Yield Outlier Detection",
                     color_discrete_sequence=['#FF6B35'])
        st.plotly_chart(fig, width='stretch')

    # Box Plot - Production Outliers
    with col2:
        fig = px.box(filtered_df, y='Production', title="Production Outlier Detection",
                     color_discrete_sequence=['#1E90FF'])
        st.plotly_chart(fig, width='stretch')


if selectors == ":red[Crop Production Prediction]":
    production() 
elif selectors == "Region-wise Production":
     compare_region_production()
elif selectors == "Crop Distribution":
    analyze_crop_distribution()
else:
    analyze_outliers()