import streamlit as st
import pandas as pd
import joblib
import os
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

# 1. ตั้งค่าหน้าเว็บให้ดูเป็นทางการ
st.set_page_config(
    page_title="SmartRetail Sales Forecast",
    page_icon="📊",
    layout="wide"
)

# --- ส่วนเสริม: ฟังก์ชันโหลดโมเดลพร้อมตรวจสอบไฟล์ ---
@st.cache_resource
def load_my_model():
    model_path = 'retail_sales_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"❌ ไม่พบไฟล์โมเดล '{model_path}' กรุณาตรวจสอบการอัปโหลดไฟล์")
        return None

model = load_my_model()

# --- ส่วนหัวของแอป ---
st.title("📊 SmartRetail Sales Forecast")
st.markdown("""
แอปพลิเคชันพยากรณ์ยอดขายสินค้าโดยใช้ AI (XGBoost) 
ช่วยให้คุณวางแผนสต็อกสินค้าและตั้งราคาได้อย่างแม่นยำ
---
""")

# --- ส่วนรับข้อมูลจากผู้ใช้ (Input) ---
if model:
    # แบ่งหน้าจอเป็น 2 ฝั่งหลัก
    with st.expander("📝 ระบุข้อมูลรายละเอียดสินค้าและสาขา", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.info("🔢 ข้อมูลตัวเลข (Numerical)")
            inventory = st.number_input("ระดับสต็อกปัจจุบัน (Inventory Level)", min_value=0, value=100)
            ordered = st.number_input("จำนวนที่สั่งซื้อเพิ่ม (Units Ordered)", min_value=0, value=50)
            demand = st.number_input("การพยากรณ์ความต้องการ (Demand Forecast)", min_value=0, value=60)
            price = st.number_input("ราคาสินค้า (Price)", min_value=0.0, value=199.0)
            discount = st.slider("ส่วนลด (Discount %)", 0.0, 1.0, 0.1, help="0.1 คือส่วนลด 10%")
            comp_price = st.number_input("ราคาคู่แข่ง (Competitor Pricing)", min_value=0.0, value=195.0)
            holiday = st.selectbox("โปรโมชันวันหยุด (Holiday Promotion)", options=[0, 1], format_func=lambda x: "มีโปรโมชัน" if x == 1 else "ไม่มีโปรโมชัน")

        with col2:
            st.info("🏷️ ข้อมูลหมวดหมู่ (Categorical)")
            category = st.selectbox("หมวดหมู่สินค้า (Category)", ['Electronics', 'Clothing', 'Food', 'Health'])
            region = st.selectbox("ภูมิภาค (Region)", ['North', 'South', 'East', 'West', 'Central'])
            weather = st.selectbox("สภาพอากาศ (Weather)", ['Sunny', 'Rainy', 'Cloudy'])
            season = st.selectbox("ฤดูกาล (Seasonality)", ['Spring', 'Summer', 'Autumn', 'Winter'])
            store_id = st.text_input("รหัสสาขา (Store ID)", "ST001")
            product_id = st.text_input("รหัสสินค้า (Product ID)", "PR001")

    # --- ส่วนประมวลผล ---
    st.markdown("---")
    center_col = st.columns([1, 2, 1])[1] # ทำให้ปุ่มอยู่ตรงกลาง
    
    with center_col:
        predict_btn = st.button("🔮 คลิกเพื่อพยากรณ์ยอดขาย", use_container_width=True)

    if predict_btn:
        # เตรียมข้อมูล
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
        
        # ทำนายผล
        with st.spinner('กำลังประมวลผล...'):
            prediction = model.predict(input_data)[0]
        
        # แสดงผลลัพธ์แบบ Metric ให้ดูสวยงาม
        st.balloons()
        st.markdown("<h3 style='text-align: center;'>ผลการวิเคราะห์</h3>", unsafe_allow_html=True)
        
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col2.metric(label="ยอดขายที่คาดการณ์ (Units Sold)", value=f"{prediction:.2f} ชิ้น")
        
        # ให้คำแนะนำเบื้องต้น
        if prediction > inventory:
            st.warning(f"⚠️ คำเตือน: ยอดขายที่พยากรณ์ได้ ({prediction:.2f}) สูงกว่าสต็อกที่มีอยู่ ({inventory}) ควรพิจารณาสั่งของเพิ่ม!")
        else:
            st.success("✅ สต็อกสินค้าเพียงพอต่อความต้องการที่พยากรณ์ไว้")

# --- ส่วนท้ายหน้าเว็บ ---
st.sidebar.markdown("### เกี่ยวกับโมเดล")
st.sidebar.write("โมเดล: XGBoost Regressor (Tuned)")
st.sidebar.write("ความแม่นยำ (MAE): ~7.17 Units")
