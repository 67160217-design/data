import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="SmartRetail Sales Forecast AI",
    page_icon="📈",
    layout="wide"
)

# --- 2. แต่งสวยด้วย Custom CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%; border-radius: 10px; height: 3.5em;
        background-color: #ff4b4b; color: white; font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #ff3333; border: 1px solid white; }
    .result-card {
        background-color: #ffffff; padding: 30px; border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05); text-align: center;
        border-top: 5px solid #ff4b4b;
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

# --- 4. ส่วนหัวของแอป ---
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/3222/3222672.png", width=80)
with col_title:
    st.title("SmartRetail Sales Forecast AI")
    st.caption("Intelligence System: วิเคราะห์แนวโน้มยอดขายและบริหารจัดการสต็อกด้วยเทคโนโลยี XGBoost")

st.markdown("---")

# --- 5. ระบบ Dynamic Pricing (ค่าเริ่มต้นตามหมวดหมู่) ---
price_defaults = {
    'Electronics': 1500.0,
    'Clothing': 450.0,
    'Food': 99.0,
    'Health': 350.0
}

if model:
    # --- 6. เลย์เอาต์เมนู Input ---
    tab1, tab2 = st.tabs(["📋 ข้อมูลหลัก", "⚙️ ข้อมูลขั้นสูง"])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 🏬 สาขาและภูมิภาค")
            region = st.selectbox("ภูมิภาค (Region)", ['North', 'South', 'East', 'West', 'Central'])
            store_id = st.text_input("รหัสสาขา (Store ID)", "ST001")
            
        with c2:
            st.markdown("### 📦 ข้อมูลสต็อก")
            inventory = st.number_input("สต็อกปัจจุบัน (In-stock)", min_value=0, value=100)
            ordered = st.number_input("จำนวนที่สั่งเพิ่ม (Re-order)", min_value=0, value=50)
            
        with c3:
            st.markdown("### 🏷️ ข้อมูลสินค้า")
            category = st.selectbox("หมวดหมู่สินค้า", list(price_defaults.keys()))
            # ดึงราคาตั้งต้นจาก Dictionary
            price = st.number_input("ราคาขาย (฿)", min_value=0.0, value=price_defaults[category])

    with tab2:
        c4, c5, c6 = st.columns(3)
        with c4:
            st.markdown("### 🌦️ ปัจจัยภายนอก")
            weather = st.selectbox("สภาพอากาศ", ['Sunny', 'Rainy', 'Cloudy'])
            season = st.selectbox("ฤดูกาล", ['Spring', 'Summer', 'Autumn', 'Winter'])
            
        with c5:
            st.markdown("### 📢 โปรโมชัน")
            discount = st.slider("ส่วนลด (Discount Ratio)", 0.0, 1.0, 0.1, help="0.1 หมายถึงส่วนลด 10%")
            holiday = st.selectbox("สถานะวันหยุด", options=[0, 1], format_func=lambda x: "มีโปรโมชันพิเศษ" if x == 1 else "วันปกติ")
            
        with c6:
            st.markdown("### 📉 คู่แข่งและเป้าหมาย")
            comp_price = st.number_input("ราคาคู่แข่ง (฿)", value=price - 10.0)
            demand = st.number_input("ประมาณการ Demand", value=int(inventory * 0.8))
            product_id = st.text_input("รหัสสินค้า", "PR001")

    # --- 7. ส่วนประมวลผลการทำนาย ---
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 วิเคราะห์และทำนายผลยอดขาย"):
        input_data = pd.DataFrame([{
            'Inventory_Level': inventory, 'Units_Ordered': ordered, 'Demand_Forecast': demand,
            'Price': price, 'Discount': discount, 'Competitor_Pricing': comp_price,
            'Holiday_Promotion': holiday, 'Store_ID': store_id, 'Product_ID': product_id,
            'Category': category, 'Region': region, 'Weather_Condition': weather, 'Seasonality': season
        }])
        
        with st.spinner('🔮 AI กำลังวิเคราะห์ความสัมพันธ์ของข้อมูล...'):
            prediction = model.predict(input_data)[0]
            if prediction < 0: prediction = 0 # ป้องกันยอดขายติดลบ
        
        # --- 8. แสดงผลลัพธ์แบบ Insight ---
        st.markdown(f"""
            <div class="result-card">
                <h2 style='color: #555;'>ยอดขายที่คาดการณ์ (Next 24 Hours)</h2>
                <h1 style='font-size: 70px; color: #ff4b4b;'>{prediction:.2f} <small style='font-size: 20px; color: #888;'>ชิ้น</small></h1>
                <p style='color: #888;'>ความแม่นยำของโมเดล (MAE): 7.17</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.subheader("📦 Stock Analysis")
            if prediction > inventory:
                st.error(f"🚩 **สินค้าไม่พอขาย:** คาดว่าจะขาดสต็อกประมาณ {prediction-inventory:.0f} ชิ้น")
            elif (inventory - prediction) < 15:
                st.warning(f"⚠️ **เฝ้าระวัง:** สต็อกปริ่มน้ำ เหลือสำรองประมาณ {inventory-prediction:.0f} ชิ้น")
            else:
                st.success(f"✅ **ปลอดภัย:** มีสต็อกเพียงพอ (เหลือสำรอง {inventory-prediction:.0f} ชิ้น)")
        
        with rec_col2:
            st.subheader("💰 Pricing Strategy")
            price_gap = price - comp_price
            if price_gap > 0:
                st.info(f"💡 **Insight:** ราคาคุณสูงกว่าคู่แข่ง {price_gap:.2f}฿ (ส่วนต่าง { (price_gap/comp_price)*100 :.1f}%)")
            else:
                st.success(f"💎 **Insight:** ราคาถูกกว่าคู่แข่ง {abs(price_gap):.2f}฿ เป็นจุดแข็งในการขาย")

else:
    st.warning("⚠️ ระบบไม่พบไฟล์โมเดล 'retail_sales_model.pkl' กรุณาตรวจสอบใน GitHub Repository")

# --- 9. Sidebar ข้อมูลระบบ ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.title("System Health")
    st.metric("Model Status", "Operational", delta="Ready")
    st.markdown("---")
    st.write("**Model Parameters:**")
    st.json({"Learning Rate": 0.05, "Max Depth": 4, "Estimators": 200}) #
    st.caption("Powered by XGBoost Machine Learning")
