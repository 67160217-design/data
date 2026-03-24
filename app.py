import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. ตั้งค่าหน้าเพจ
st.set_page_config(page_title="SmartRetail Forecast", page_icon="🛒", layout="wide")

# 2. โหลดโมเดล (ใช้ @st.cache_resource เพื่อไม่ให้โหลดใหม่ทุกครั้งที่ขยับหน้าเว็บ)
@st.cache_resource
def load_model():
    return joblib.load('retail_sales_model.pkl')

model = load_model()

# 3. ส่วนหัวของเว็บ
st.title("🛒 SmartRetail: ระบบคาดการณ์ยอดขายอัจฉริยะ")
st.markdown("แอปพลิเคชันนี้ใช้ Machine Learning (XGBoost) เพื่อช่วยผู้จัดการร้านประเมินยอดขายสินค้าในแต่ละวัน")

# 4. สร้าง Sidebar สำหรับรับค่าจากผู้ใช้งาน
st.sidebar.header("📝 กรอกข้อมูลสถานการณ์วันนี้")

# รับค่าตัวเลข
price = st.sidebar.number_input("ราคาสินค้า (Price)", min_value=0.0, value=50.0)
discount = st.sidebar.number_input("ส่วนลด (%)", min_value=0.0, max_value=100.0, value=0.0)
comp_price = st.sidebar.number_input("ราคาคู่แข่ง (Competitor Price)", min_value=0.0, value=50.0)
inventory = st.sidebar.number_input("สินค้าคงคลังปัจจุบัน (Inventory Level)", min_value=0, value=100)

# รับค่าหมวดหมู่
category = st.sidebar.selectbox("หมวดหมู่สินค้า (Category)", ['Groceries', 'Toys', 'Electronics', 'Clothing', 'Furniture'])
weather = st.sidebar.selectbox("สภาพอากาศ (Weather)", ['Sunny', 'Cloudy', 'Rainy', 'Snowy'])
holiday = st.sidebar.radio("มีโปรโมชัน/วันหยุดหรือไม่?", [0, 1], format_func=lambda x: "มี (1)" if x == 1 else "ไม่มี (0)")

# 5. เมื่อกดปุ่ม "ทำนายยอดขาย"
if st.sidebar.button("🚀 ทำนายยอดขาย", type="primary"):
    
    # แปลงข้อมูลที่ผู้ใช้กรอก เป็น DataFrame
    input_data = pd.DataFrame({
        'Price': [price],
        'Discount': [discount],
        'Competitor_Pricing': [comp_price],
        'Inventory_Level': [inventory],
        'Category': [category],
        'Weather_Condition': [weather],
        'Holiday_Promotion': [holiday]
    })
    
    # ทำนายผล
    prediction = model.predict(input_data)[0]
    predicted_sales = int(np.round(prediction))
    
    # แสดงผลลัพธ์
    st.subheader("📊 ผลการคาดการณ์")
    st.metric("ยอดขายที่คาดหวัง (ชิ้น)", f"{predicted_sales} ชิ้น")
    
    # แจ้งเตือนสถานะสต๊อก (ใส่ลูกเล่น Business Logic)
    if predicted_sales > inventory:
        st.error(f"⚠️ คำเตือน: สินค้าในสต๊อก ({inventory} ชิ้น) อาจจะไม่พอขาย! แนะนำให้สั่งซื้อเพิ่มด่วน")
    elif predicted_sales < (inventory * 0.2):
        st.warning(f"📦 สต๊อกเหลือเยอะมาก ({inventory} ชิ้น) แต่คาดว่าจะขายได้น้อย แนะนำให้พิจารณาเพิ่มโปรโมชัน")
    else:
        st.success("✅ สต๊อกสินค้าอยู่ในระดับที่ปลอดภัย บริหารจัดการได้ดีมาก")