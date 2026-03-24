import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. ตั้งค่าหน้าเพจ
st.set_page_config(page_title="SmartRetail Forecast", page_icon="🛒", layout="wide")

# 2. โหลดโมเดลพร้อมระบบจัดการ Error
@st.cache_resource
def load_model():
    try:
        return joblib.load('retail_sales_model.pkl')
    except FileNotFoundError:
        return None

model = load_model()

# 3. ส่วนหัวของเว็บ
st.title("🛒 SmartRetail: ระบบคาดการณ์ยอดขายอัจฉริยะ")
st.markdown("แอปพลิเคชันนี้ใช้ Machine Learning เพื่อช่วยประเมินยอดขายสินค้าและบริหารจัดการสต๊อก")

if model is None:
    st.error("❌ ไม่พบไฟล์ 'retail_sales_model.pkl' กรุณาตรวจสอบว่าไฟล์โมเดลอยู่ในโฟลเดอร์เดียวกันกับ app.py")
    st.stop()

# 4. ส่วนรับข้อมูล (Sidebar)
st.sidebar.header("📝 ข้อมูลสถานการณ์")

price = st.sidebar.number_input("ราคาสินค้า (Price)", min_value=0.0, value=50.0)
discount = st.sidebar.number_input("ส่วนลด (%)", min_value=0.0, max_value=100.0, value=0.0)
comp_price = st.sidebar.number_input("ราคาคู่แข่ง (Competitor Price)", min_value=0.0, value=50.0)
inventory = st.sidebar.number_input("สินค้าคงคลังปัจจุบัน (Inventory Level)", min_value=0, value=100)

category = st.sidebar.selectbox("หมวดหมู่สินค้า (Category)", 
                                ['Groceries', 'Toys', 'Electronics', 'Clothing', 'Furniture'])
weather = st.sidebar.selectbox("สภาพอากาศ (Weather)", 
                               ['Sunny', 'Cloudy', 'Rainy', 'Snowy'])
holiday = st.sidebar.radio("โปรโมชัน/วันหยุด", [0, 1], format_func=lambda x: "มี" if x == 1 else "ไม่มี")

# 5. ฟังก์ชันแปลงข้อมูล (Preprocessing)
def preprocess_input(cat, weath):
    # ตัวอย่าง Label Encoding (ต้องตรงกับตอนที่ใช้เทรนโมเดล)
    cat_map = {'Clothing': 0, 'Electronics': 1, 'Furniture': 2, 'Groceries': 3, 'Toys': 4}
    weath_map = {'Cloudy': 0, 'Rainy': 1, 'Snowy': 2, 'Sunny': 3}
    
    return cat_map.get(cat, 0), weath_map.get(weath, 0)

# 6. ประมวลผลเมื่อกดปุ่ม
if st.sidebar.button("🚀 ทำนายยอดขาย", type="primary"):
    
    cat_encoded, weath_encoded = preprocess_input(category, weather)
    
    # สร้าง DataFrame ให้โครงสร้างเหมือนตอน Train
    input_df = pd.DataFrame([[
        price, discount, comp_price, inventory, 
        cat_encoded, weath_encoded, holiday
    ]], columns=['Price', 'Discount', 'Competitor_Pricing', 'Inventory_Level', 
                 'Category', 'Weather_Condition', 'Holiday_Promotion'])
    
    try:
        prediction = model.predict(input_df)[0]
        predicted_sales = max(0, int(np.round(prediction))) # ป้องกันยอดขายติดลบ

        # แสดงผลลัพธ์
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("ยอดขายที่คาดการณ์", f"{predicted_sales} ชิ้น")
        
        with col2:
            stock_status = "⚠️ ไม่เพียงพอ" if predicted_sales > inventory else "✅ เพียงพอ"
            st.metric("สถานะสต๊อก", stock_status)

        # Business Logic Alert
        if predicted_sales > inventory:
            st.error(f"🚨 **คำเตือน:** สต๊อกขาด! ต้องการสินค้าเพิ่มอย่างน้อย {predicted_sales - inventory} ชิ้น")
        elif predicted_sales < (inventory * 0.2):
            st.warning("📦 **ข้อเสนอแนะ:** สต๊อกเหลือมากเกินไป แนะนำให้จัดโปรโมชันเพื่อระบายสินค้า")
        else:
            st.success("👌 **สถานะปกติ:** ระดับสต๊อกเหมาะสมกับยอดขายที่คาดการณ์")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
