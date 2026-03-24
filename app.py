import streamlit as st
import pandas as pd
import joblib
import os
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

# 1. ตั้งค่าหน้าเว็บให้ดูทันสมัยและกว้างขึ้น
st.set_page_config(
    page_title="SmartRetail AI | Sales Forecast",
    page_icon="📈",
    layout="wide"
)

# --- 2. แต่งสวยด้วย Custom CSS (Modern Enterprise UI) ---
st.markdown("""
    <style>
    /* พื้นหลังหลัก */
    .main { background-color: #f8fafc; font-family: 'Inter', sans-serif; }
    
    /* ปุ่มกด Gradient และ Hover Effect */
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.8em;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white; font-weight: 600; border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); transition: all 0.2s;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); color: #fff; }

    /* การ์ดแสดงผลลัพธ์ Predicted Sales */
    .result-card {
        background: white; padding: 30px; border-radius: 24px;
        border: 1px solid #e2e8f0; text-align: center;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05); margin-bottom: 20px;
    }
    
    /* หัวข้อ Section */
    .section-header {
        color: #1e293b; font-size: 1.1rem; font-weight: 700;
        margin-bottom: 12px; display: flex; align-items: center; gap: 8px;
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

# --- 4. Header Section ---
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/3222/3222672.png", width=70)
with col_title:
    st.markdown("<h1 style='color: #1e293b; margin-bottom: 0;'>SmartRetail <span style='color: #6366f1;'>AI Engine</span></h1>", unsafe_allow_html=True)
    st.caption("ระบบพยากรณ์ยอดขายรายวันด้วย XGBoost Regressor | ความแม่นยำสูง (MAE: 7.17)")

st.write("") 

if model:
    # --- 5. จัดกลุ่ม Input ด้วย Expander หรือ Card ---
    with st.container():
        tab1, tab2 = st.tabs(["📋 ข้อมูลหลัก", "⚙️ ปัจจัยแวดล้อม"])
        
        with tab1:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown('<div class="section-header">🏬 สาขาและภูมิภาค</div>', unsafe_allow_html=True)
                region = st.selectbox("ภูมิภาค (Region)", ['North', 'South', 'East', 'West', 'Central'])
                store_id = st.text_input("รหัสสาขา", "ST001")
            with c2:
                st.markdown('<div class="section-header">📦 สต็อกและการเติมสินค้า</div>', unsafe_allow_html=True)
                inventory = st.number_input("สต็อกปัจจุบัน", min_value=0, value=100)
                ordered = st.number_input("จำนวนที่สั่งเพิ่ม", min_value=0, value=50)
            with c3:
                st.markdown('<div class="section-header">🏷️ รายละเอียดสินค้า</div>', unsafe_allow_html=True)
                category = st.selectbox("หมวดหมู่สินค้า", ['Electronics', 'Clothing', 'Food', 'Health'])
                price = st.number_input("ราคาขาย (฿)", min_value=0.0, value=199.0)

        with tab2:
            c4, c5, c6 = st.columns(3)
            with c4:
                st.markdown('<div class="section-header">🌦️ ปัจจัยแวดล้อม</div>', unsafe_allow_html=True)
                weather = st.selectbox("สภาพอากาศ", ['Sunny', 'Rainy', 'Cloudy'])
                season = st.selectbox("ฤดูกาล", ['Spring', 'Summer', 'Autumn', 'Winter'])
            with c5:
                st.markdown('<div class="section-header">📢 โปรโมชัน</div>', unsafe_allow_html=True)
                discount = st.slider("ส่วนลด (0.0 - 1.0)", 0.0, 1.0, 0.1)
                holiday = st.selectbox("วันหยุด/แคมเปญ", [0, 1], format_func=lambda x: "มีโปรโมชัน" if x == 1 else "วันปกติ")
            with c6:
                st.markdown('<div class="section-header">📊 เป้าหมายและคู่แข่ง</div>', unsafe_allow_html=True)
                comp_price = st.number_input("ราคาคู่แข่ง", value=195.0)
                demand = st.number_input("พยากรณ์ Demand", value=60)
                product_id = st.text_input("รหัสสินค้า", "PR001")

    # --- 6. Prediction Logic ---
    st.write("---")
    if st.button("🚀 คำนวณยอดขายอัจฉริยะ"):
        input_data = pd.DataFrame([{
            'Inventory_Level': inventory, 'Units_Ordered': ordered, 'Demand_Forecast': demand,
            'Price': price, 'Discount': discount, 'Competitor_Pricing': comp_price,
            'Holiday_Promotion': holiday, 'Store_ID': store_id, 'Product_ID': product_id,
            'Category': category, 'Region': region, 'Weather_Condition': weather, 'Seasonality': season
        }])
        
        with st.spinner('🔮 AI กำลังประมวลผลข้อมูล...'):
            prediction = model.predict(input_data)[0]
            if prediction < 0: prediction = 0
        
        # --- 7. แสดงผลลัพธ์ (Clean & Powerful) ---
        st.markdown(f"""
            <div class="result-card">
                <p style='color: #64748b; font-size: 1.1rem; font-weight: 500;'>ยอดขายพยากรณ์ครั้งถัดไป</p>
                <h1 style='font-size: 80px; color: #1e293b; margin: 0;'>{prediction:.2f} <small style='font-size: 24px; color: #94a3b8;'>Units</small></h1>
                <p style='color: #6366f1; font-weight: bold; margin-top: 10px;'>Model Algorithm: XGBoost Optimized</p>
            </div>
            """, unsafe_allow_html=True)
        
        # รายละเอียดคำแนะนำ
        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            st.subheader("📦 Inventory Analysis")
            if prediction > inventory:
                st.error(f"⚠️ **สต็อกไม่เพียงพอ:** คาดว่าสินค้าจะขาดตลาดประมาณ {prediction - inventory:.0f} ชิ้น")
            else:
                st.success("✅ **สต็อกเพียงพอ:** มีสินค้าสำรองสำหรับการขายรอบนี้")
                # แก้ไข Bug Progress Bar: จำกัดค่าให้อยู่ระหว่าง 0.0 - 1.0
                ratio = prediction / inventory if inventory > 0 else 0
                st.progress(float(min(max(ratio, 0.0), 1.0)))

        with rec_col2:
            st.subheader("💡 Business Insight")
            if price > comp_price:
                st.warning(f"ราคาสูงกว่าคู่แข่ง {price-comp_price:.2f}฿ | แนะนำให้เพิ่มสิทธิประโยชน์พิเศษ")
            else:
                st.info("ราคาของคุณมีความสามารถในการแข่งขันสูง (Competitive Price)")

else:
    st.error("⚠️ ระบบไม่พบโมเดล 'retail_sales_model.pkl'")

# --- 8. Sidebar สำหรับ System Health ---
with st.sidebar:
    st.markdown("### 🖥️ System Status")
    st.metric("Engine Status", "Online", delta="Stable")
    st.write("---")
    st.write("**Model Metadata:**")
    st.info(f"MAE: 7.17\nRMSE: 8.40\nXGBoost: 3.2.0")
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)

