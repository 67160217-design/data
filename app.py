import streamlit as st
import pandas as pd
import joblib
import os
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

# 1. ตั้งค่าหน้าเว็บให้ดูเป็นทางการและกว้างขึ้น
st.set_page_config(
    page_title="SmartRetail Sales Forecast AI",
    page_icon="📈",
    layout="wide"
)

# --- 2. แต่งสวยด้วย Custom CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .result-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .stMetric {
        background-color: #f1f3f6;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. ฟังก์ชันโหลดโมเดล ---
@st.cache_resource
def load_my_model():
    model_path = 'retail_sales_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_my_model()

# --- 4. ส่วนหัวของแอป (Hero Section) ---
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/3222/3222672.png", width=80)
with col_title:
    st.title("SmartRetail Sales Forecast AI")
    st.caption("ระบบพยากรณ์ยอดขายอัจฉริยะ แม่นยำระดับรายวัน เพื่อการวางแผนสต็อกที่มีประสิทธิภาพ")

st.markdown("---")

if model:
    # --- 5. การจัดเลย์เอาต์เมนู Input ---
    tab1, tab2 = st.tabs(["📋 ข้อมูลหลัก", "⚙️ ข้อมูลขั้นสูง"])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 🏬 สาขาและภูมิภาค")
            region = st.selectbox("ภูมิภาค (Region)", ['North', 'South', 'East', 'West', 'Central'])
            store_id = st.text_input("รหัสสาขา (Store ID)", "ST001")
            
        with c2:
            st.markdown("### 📦 ข้อมูลสต็อก")
            inventory = st.number_input("สต็อกปัจจุบัน", min_value=0, value=100)
            ordered = st.number_input("จำนวนที่สั่งเพิ่ม", min_value=0, value=50)
            
        with c3:
            st.markdown("### 🏷️ ข้อมูลสินค้า")
            category = st.selectbox("หมวดหมู่สินค้า", ['Electronics', 'Clothing', 'Food', 'Health'])
            price = st.number_input("ราคาขาย (฿)", min_value=0.0, value=199.0)

    with tab2:
        c4, c5, c6 = st.columns(3)
        with c4:
            st.markdown("### 🌦️ ปัจจัยภายนอก")
            weather = st.selectbox("สภาพอากาศ", ['Sunny', 'Rainy', 'Cloudy'])
            season = st.selectbox("ฤดูกาล", ['Spring', 'Summer', 'Autumn', 'Winter'])
            
        with c5:
            st.markdown("### 📢 โปรโมชัน")
            discount = st.slider("ส่วนลด (Discount %)", 0.0, 1.0, 0.1)
            holiday = st.selectbox("วันหยุด", options=[0, 1], format_func=lambda x: "วันหยุด/มีโปร" if x == 1 else "วันปกติ")
            
        with c6:
            st.markdown("### 📉 คู่แข่งและเป้าหมาย")
            comp_price = st.number_input("ราคาคู่แข่ง", value=195.0)
            demand = st.number_input("พยากรณ์ Demand", value=60)
            product_id = st.text_input("รหัสสินค้า", "PR001")

    # --- 6. ส่วนประมวลผลการทำนาย ---
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 เริ่มการพยากรณ์ยอดขาย"):
        input_data = pd.DataFrame([{
            'Inventory_Level': inventory, 'Units_Ordered': ordered, 'Demand_Forecast': demand,
            'Price': price, 'Discount': discount, 'Competitor_Pricing': comp_price,
            'Holiday_Promotion': holiday, 'Store_ID': store_id, 'Product_ID': product_id,
            'Category': category, 'Region': region, 'Weather_Condition': weather, 'Seasonality': season
        }])
        
        with st.spinner('🔮 AI กำลังคำนวณยอดขายที่เหมาะสม...'):
            prediction = model.predict(input_data)[0]
        
        # --- 7. แสดงผลลัพธ์แบบ "ว้าว" ---
        st.markdown(f"""
            <div class="result-card">
                <h2 style='color: #ff4b4b;'>ยอดขายพยากรณ์ (Predicted Sales)</h2>
                <h1 style='font-size: 60px;'>{prediction:.2f} <small style='font-size: 20px;'>Units</small></h1>
            </div>
            """, unsafe_allow_html=True)
        
        # คำแนะนำทางธุรกิจ (Business Recommendation)
        st.markdown("<br>", unsafe_allow_html=True)
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            if prediction > inventory:
                st.error(f"🚩 **ความเสี่ยง:** สินค้าอาจขาดสต็อก! (ขาดประมาณ {prediction-inventory:.0f} ชิ้น)")
            else:
                st.success("✅ **สถานะ:** สต็อกเพียงพอสำหรับรอบการขายนี้")
        
        with rec_col2:
            price_gap = price - comp_price
            if price_gap > 0:
                st.warning(f"💡 **แนะนำ:** ราคาคุณสูงกว่าคู่แข่ง {price_gap:.2f}฿ ลองพิจารณาเพิ่มส่วนลด")
            else:
                st.info("💡 **แนะนำ:** ราคาสินค้าของคุณมีความสามารถในการแข่งขันสูง")

else:
    st.warning("⚠️ กรุณาอัปโหลดไฟล์ 'retail_sales_model.pkl' ไปยังโฟลเดอร์เดียวกับแอป")

# --- 8. Sidebar สำหรับ Admin/Developer ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("System Health")
    st.metric("Model Status", "Active", delta="Ready")
    st.write("---")
    st.write("**Model Info:**")
    st.code("Type: XGBoost\nVersion: 3.2.0\nMAE: 7.17")
    st.caption("Developed by SmartRetail AI Team")
