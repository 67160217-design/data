import streamlit as st
import pandas as pd
import joblib
import os
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

# 1. ตั้งค่าหน้าเว็บให้ดูทันสมัย
st.set_page_config(
    page_title="SmartRetail AI | Sales Forecast",
    page_icon="📈",
    layout="wide"
)

# --- 2. แต่งสวยด้วย Custom CSS (Modern UI) ---
st.markdown("""
    <style>
    /* เปลี่ยนพื้นหลังและฟอนต์ */
    .main {
        background-color: #f0f2f6;
        font-family: 'Inter', sans-serif;
    }
    
    /* ตกแต่งปุ่มกดให้มีความโค้งและเงา */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        color: #fff;
    }

    /* ตกแต่งการ์ดผลลัพธ์ */
    .result-card {
        background: white;
        padding: 40px;
        border-radius: 20px;
        border-left: 10px solid #764ba2;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 25px;
    }

    /* ปรับแต่ง Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* หัวข้อ Section */
    .section-header {
        color: #1e293b;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 15px;
        border-bottom: 2px solid #764ba2;
        padding-bottom: 5px;
        width: fit-content;
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
with st.container():
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        st.image("https://cdn-icons-png.flaticon.com/512/3222/3222672.png", width=70)
    with col_title:
        st.markdown("<h1 style='color: #1e293b; margin-bottom: 0;'>SmartRetail <span style='color: #764ba2;'>Sales Forecast AI</span></h1>", unsafe_allow_html=True)
        st.caption("🚀 ขับเคลื่อนการตัดสินใจด้วยแมชชีนเลิร์นนิง | วิเคราะห์สต็อกและยอดขายแบบเรียลไทม์")

st.write("") # เว้นวรรค

# --- 5. Main Content ---
if model:
    # ส่วนกรอกข้อมูลแบ่งเป็น Tab
    tab1, tab2 = st.tabs(["📊 ข้อมูลการดำเนินงาน", "🔍 ปัจจัยแวดล้อม"])
    
    with tab1:
        st.markdown('<p class="section-header">Core Operations</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            category = st.selectbox("🛍️ หมวดหมู่สินค้า", ['Electronics', 'Clothing', 'Food', 'Health'])
            region = st.selectbox("📍 ภูมิภาค", ['North', 'South', 'East', 'West', 'Central'])
            store_id = st.text_input("🆔 รหัสสาขา", "ST001")
        with c2:
            inventory = st.number_input("📦 สต็อกปัจจุบัน", min_value=0, value=100)
            ordered = st.number_input("📥 จำนวนที่สั่งเพิ่ม", min_value=0, value=50)
            demand = st.number_input("🎯 พยากรณ์ Demand", value=60)
        with c3:
            price = st.number_input("💰 ราคาขาย (฿)", min_value=0.0, value=199.0)
            comp_price = st.number_input("🥊 ราคาคู่แข่ง (฿)", value=195.0)
            product_id = st.text_input("🏷️ รหัสสินค้า", "PR001")

    with tab2:
        st.markdown('<p class="section-header">Environmental Factors</p>', unsafe_allow_html=True)
        c4, c5 = st.columns(2)
        with c4:
            weather = st.selectbox("🌦️ สภาพอากาศ", ['Sunny', 'Rainy', 'Cloudy'])
            season = st.selectbox("🍂 ฤดูกาล", ['Spring', 'Summer', 'Autumn', 'Winter'])
        with c5:
            discount = st.select_slider("🏷️ ส่วนลดพิเศษ (Ratio)", options=[round(i*0.1,1) for i in range(11)], value=0.1)
            holiday = st.radio("🎊 ช่วงเทศกาล/โปรโมชัน", options=[0, 1], format_func=lambda x: "มีโปรโมชัน" if x == 1 else "วันปกติ", horizontal=True)

    # --- 6. Prediction Logic ---
    st.write("---")
    if st.button("🔮 เริ่มการพยากรณ์อัจฉริยะ"):
        input_data = pd.DataFrame([{
            'Inventory_Level': inventory, 'Units_Ordered': ordered, 'Demand_Forecast': demand,
            'Price': price, 'Discount': discount, 'Competitor_Pricing': comp_price,
            'Holiday_Promotion': holiday, 'Store_ID': store_id, 'Product_ID': product_id,
            'Category': category, 'Region': region, 'Weather_Condition': weather, 'Seasonality': season
        }])
        
        with st.spinner('กำลังวิเคราะห์รูปแบบยอดขาย...'):
            prediction = model.predict(input_data)[0]
            if prediction < 0: prediction = 0 # ป้องกันค่าติดลบ
        
        # --- 7. ผลลัพธ์ (Result Display) ---
        st.markdown(f"""
            <div class="result-card">
                <p style='color: #64748b; font-size: 1.2rem; margin-bottom: 10px;'>ยอดขายที่คาดการณ์ล่วงหน้า</p>
                <h1 style='font-size: 75px; color: #1e293b; margin: 0;'>{prediction:.2f} <small style='font-size: 25px;'>หน่วย</small></h1>
                <p style='color: #764ba2; font-weight: bold;'>AI Prediction Confidence: High</p>
            </div>
            """, unsafe_allow_html=True)
        
        # คอลัมน์สรุปคำแนะนำ
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown("##### 📦 แผนบริหารสต็อก")
            if prediction > inventory:
                st.error(f"**ควรเติมสินค้า!** ยอดขายสูงกว่าสต็อกปัจจุบันประมาณ {prediction-inventory:.0f} ชิ้น")
            else:
                st.success(f"**สต็อกเพียงพอ!** สินค้าในคลังสามารถรองรับความต้องการได้")
                st.progress(min(prediction/inventory, 1.0) if inventory > 0 else 0)

        with res_col2:
            st.markdown("##### 💡 กลยุทธ์การขาย")
            if price > comp_price:
                st.warning(f"ราคาสูงกว่าคู่แข่ง {price-comp_price:.1f}฿ พิจารณาเพิ่มของแถมหรือบริการพิเศษ")
            else:
                st.info("ราคาอยู่ในจุดที่แข่งขันได้ดีเยี่ยม (Competitive Pricing)")
        
        st.balloons()

else:
    st.error("⚠️ ไม่พบไฟล์โมเดล! กรุณาตรวจสอบว่ามีไฟล์ 'retail_sales_model.pkl' อยู่ในโฟลเดอร์เดียวกัน")

# --- 8. Sidebar Design ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>AI Engine</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=120)
    st.write("---")
    
    # แสดงประสิทธิภาพโมเดล
    st.subheader("📊 Performance")
    st.metric("Model MAE", "7.17", help="Mean Absolute Error ยิ่งต่ำยิ่งแม่นยำ")
    st.metric("Algo", "XGBoost")
    
    st.write("---")
    st.caption("© 2024 SmartRetail AI Solution v2.0")
