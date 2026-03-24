import streamlit as st
import pandas as pd
import joblib
import os

# 1. ตั้งค่าหน้าเว็บสไตล์ Adidas (Minimal & Bold)
st.set_page_config(
    page_title="ADIDAS | Sales Performance AI",
    page_icon="👟",
    layout="wide"
)

# --- 2. Custom CSS (Adidas Aesthetic) ---
st.markdown("""
    <style>
    /* พื้นหลังขาวสะอาดตา */
    .main { background-color: #ffffff; font-family: 'Inter', sans-serif; }
    
    /* ปุ่มกดสไตล์ Adidas (เหลี่ยม, ดำ, ตัวพิมพ์ใหญ่) */
    .stButton>button {
        width: 100%; border-radius: 0px; height: 3.5em;
        background-color: #000000; color: #ffffff;
        font-weight: 800; text-transform: uppercase;
        letter-spacing: 2px; border: none; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #333333; color: #ffffff; border: none; }

    /* การ์ดผลลัพธ์เน้นความเข้ม */
    .result-card {
        background: #000000; color: #ffffff; padding: 40px;
        border-radius: 0px; text-align: center;
        border-top: 10px solid #555555; margin-bottom: 25px;
    }
    
    /* แถบ 3 เส้น (The Three Stripes) */
    .adidas-stripes {
        height: 4px; width: 50px; background: #000;
        margin-bottom: 3px; display: inline-block;
    }

    /* หัวข้อ Section ตัวหนา */
    .section-header {
        color: #000000; font-size: 1.2rem; font-weight: 900;
        text-transform: uppercase; letter-spacing: 1px;
        border-bottom: 3px solid #000; padding-bottom: 5px; margin-bottom: 20px;
    }
    
    /* ตกแต่ง Tab */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f3f4f6; border-radius: 0px;
        padding: 10px 25px; font-weight: 700;
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
    # Adidas Logo Style Icon
    st.image("https://cdn-icons-png.flaticon.com/512/732/732160.png", width=75)
with col_title:
    st.markdown("<h1 style='color: #000000; margin-bottom: 0; font-weight: 900; letter-spacing: -1px;'>SMARTRETAIL <span style='font-weight: 400;'>PERFORMANCE AI</span></h1>", unsafe_allow_html=True)
    st.caption("PRECISION DATA FOR UNSTOPPABLE GROWTH | MAE: 7.17")

st.write("") 

# ระบบราคา Default ตามหมวดหมู่
price_map = {'Electronics': 2500.0, 'Clothing': 890.0, 'Food': 150.0, 'Health': 450.0}

if model:
    # --- 5. Input Data Tabs ---
    tab1, tab2 = st.tabs(["⚡ OPERATIONAL DATA", "🌍 EXTERNAL FACTORS"])
    
    with tab1:
        st.markdown('<div class="section-header">Core Metrics</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            category = st.selectbox("PRODUCT CATEGORY", list(price_map.keys()))
            region = st.selectbox("SALES REGION", ['North', 'South', 'East', 'West', 'Central'])
            store_id = st.text_input("STORE ID", "AD_TH_01")
        with c2:
            inventory = st.number_input("CURRENT STOCK", min_value=0, value=100)
            ordered = st.number_input("RE-ORDER UNITS", min_value=0, value=50)
        with c3:
            price = st.number_input("UNIT PRICE (฿)", min_value=0.0, value=price_map[category])
            comp_price = st.number_input("COMPETITOR PRICE (฿)", value=price - 50.0)

    with tab2:
        st.markdown('<div class="section-header">Contextual Factors</div>', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            weather = st.selectbox("WEATHER", ['Sunny', 'Rainy', 'Cloudy'])
            season = st.selectbox("SEASON", ['Spring', 'Summer', 'Autumn', 'Winter'])
        with c5:
            discount = st.slider("DISCOUNT RATE (0.0 - 1.0)", 0.0, 1.0, 0.1)
            holiday = st.selectbox("CAMPAIGN STATUS", [0, 1], format_func=lambda x: "ACTIVE PROMOTION" if x == 1 else "NORMAL")
        with c6:
            demand = st.number_input("MARKET DEMAND", value=60)
            product_id = st.text_input("PRODUCT SKU", "SKU-AD-001")

    # --- 6. Prediction Logic ---
    st.write("---")
    if st.button("RUN FORECAST ENGINE"):
        input_data = pd.DataFrame([{
            'Inventory_Level': inventory, 'Units_Ordered': ordered, 'Demand_Forecast': demand,
            'Price': price, 'Discount': discount, 'Competitor_Pricing': comp_price,
            'Holiday_Promotion': holiday, 'Store_ID': store_id, 'Product_ID': product_id,
            'Category': category, 'Region': region, 'Weather_Condition': weather, 'Seasonality': season
        }])
        
        with st.spinner('CALCULATING PERFORMANCE...'):
            prediction = model.predict(input_data)[0]
            if prediction < 0: prediction = 0
        
        # --- 7. Result Display (Adidas Card) ---
        st.markdown(f"""
            <div class="result-card">
                <div style='margin-bottom: 10px;'>
                    <div class="adidas-stripes"></div> <div class="adidas-stripes"></div> <div class="adidas-stripes" style="background:#555"></div>
                </div>
                <p style='text-transform: uppercase; letter-spacing: 3px; font-size: 0.9rem; opacity: 0.8;'>Predicted Sales Volume</p>
                <h1 style='font-size: 90px; margin: 0; font-weight: 900;'>{prediction:.2f}</h1>
                <p style='font-size: 1.2rem; font-weight: 300; letter-spacing: 2px;'>UNITS PER CYCLE</p>
            </div>
            """, unsafe_allow_html=True)
        
        # คอลัมน์สรุป
        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            st.markdown("### 📦 INVENTORY")
            if prediction > inventory:
                st.error(f"**URGENT:** Stock shortfall of {prediction - inventory:.0f} units.")
            else:
                st.success("**OPTIMAL:** Stock levels are sufficient for target.")
                # Progress bar fix
                ratio = prediction / inventory if inventory > 0 else 0
                st.progress(float(min(max(ratio, 0.0), 1.0)))

        with rec_col2:
            st.markdown("### 💡 STRATEGY")
            if price > comp_price:
                st.warning(f"Premium pricing active (+{price-comp_price:.0f}฿). Monitor conversion rates.")
            else:
                st.info("Competitive advantage: Pricing is below market average.")

else:
    st.error("⚠️ CRITICAL ERROR: Model file 'retail_sales_model.pkl' not found.")

# --- 8. Sidebar ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: black; font-weight: 900;'>ENGINE STATUS</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/732/732160.png", width=100)
    st.write("---")
    st.metric("STABILITY", "ACTIVE", delta="STABLE")
    st.write("**Model Details:**")
    st.code("XGBOOST V3.2\nMAE: 7.17\nRMSE: 8.40")
    st.write("---")
    st.caption("ADIDAS RETAIL ANALYTICS © 2026")
