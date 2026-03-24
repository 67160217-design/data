import streamlit as st
import pandas as pd
import joblib
import numpy as np
# สำคัญ: ต้อง import สองตัวนี้เพื่อให้ระบบรู้จักโครงสร้างในไฟล์ .pkl
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Retail Sales Forecast", layout="centered")

# --- โหลดโมเดล ---
@st.cache_resource # ใช้ cache เพื่อให้โหลดโมเดลครั้งเดียว ช่วยให้แอปเร็วขึ้น
def load_my_model():
    return joblib.load('retail_sales_model.pkl')

model = load_my_model()

st.title("📊 SmartRetail Sales Forecast")
st.write("ระบุข้อมูลด้านล่างเพื่อพยากรณ์ยอดขาย (Units Sold)")

# --- ส่วนรับข้อมูลจากผู้ใช้ (Input) ---
# หมายเหตุ: ชื่อคอลัมน์ต้องตรงกับที่ใช้เทรนใน Colab
col1, col2 = st.columns(2)

with col1:
    st.subheader("Numeric Features")
    inventory = st.number_input("Inventory Level", value=100)
    ordered = st.number_input("Units Ordered", value=50)
    demand = st.number_input("Demand Forecast", value=60)
    price = st.number_input("Price", value=199.0)
    discount = st.slider("Discount", 0.0, 1.0, 0.1)
    comp_price = st.number_input("Competitor Pricing", value=195.0)
    holiday = st.selectbox("Holiday Promotion", [0, 1])

with col2:
    st.subheader("Categorical Features")
    category = st.selectbox("Category", ['Electronics', 'Clothing', 'Food', 'Health'])
    region = st.selectbox("Region", ['North', 'South', 'East', 'West', 'Central'])
    weather = st.selectbox("Weather Condition", ['Sunny', 'Rainy', 'Cloudy'])
    season = st.selectbox("Seasonality", ['Spring', 'Summer', 'Autumn', 'Winter'])
    # เพิ่ม Store_ID และ Product_ID (ถ้าจำเป็นต้องใช้ตามที่เทรนไว้)
    store_id = st.text_input("Store ID", "ST001")
    product_id = st.text_input("Product ID", "PR001")

# --- ส่วนประมวลผลการทำนาย ---
if st.button("🔮 ทำนายยอดขาย"):
    # สร้าง DataFrame จาก Input (ชื่อคอลัมน์ต้องตรงกับ X_train เป๊ะๆ)
    input_data = pd.DataFrame([{
        'Inventory_Level': inventory,
        'Units_Ordered': ordered,
        'Demand_Forecast': demand,
        'Price': price,
        'Discount': discount,
        'Competitor_Pricing': comp_price,
        'Holiday_Promotion': holiday,
        'Store_ID': store_id,
        'Product_ID': product_id,
        'Category': category,
        'Region': region,
        'Weather_Condition': weather,
        'Seasonality': season
    }])
    
    # ทำนายผล (โมเดลที่เป็น Pipeline จะจัดการ Encode/Scale ให้เองอัตโนมัติ)
    prediction = model.predict(input_data)
    
    st.success(f"### ผลการพยากรณ์ยอดขาย: {prediction[0]:.2f} Units")