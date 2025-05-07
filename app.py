import streamlit as st
import joblib
import os
from googletrans import Translator

# Initialize translator
translator = Translator()

st.title("Hotel Review Sentiment Classifier")
st.markdown("พิมพ์ข้อความรีวิวโรงแรมเป็นภาษาไทย แล้วเลือกโมเดลที่ต้องการให้ AI วิเคราะห์")

# ค้นหาไฟล์โมเดล
model_files = sorted([f for f in os.listdir() if f.startswith("model_") and f.endswith(".pkl")])
if not model_files:
    st.error("ไม่พบไฟล์โมเดล (model_*.pkl) ในโฟลเดอร์นี้")
    st.stop()

# ตัวเลือกโมเดล
model_choice = st.selectbox("เลือกโมเดล", model_files)
user_input = st.text_area("กรอกข้อความรีวิวโรงแรมที่นี่ (ภาษาไทย)", height=150)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("กรุณากรอกข้อความก่อน")
    else:
        try:
            # แปลข้อความจากไทยเป็นอังกฤษ
            translation = translator.translate(user_input.strip(), src='th', dest='en')
            translated_text = translation.text
            # st.markdown(f"**ข้อความที่แปลเป็นอังกฤษ:** _{translated_text}_")

            # โหลดโมเดล + vectorizer
            model, vectorizer = joblib.load(model_choice)
            X_input = vectorizer.transform([translated_text])
            pred = model.predict(X_input)[0]

            # แสดงผล
            st.subheader("ผลการทำนาย:")
            if pred == 1:
                st.success("Positive (รีวิวเชิงบวก)")
            else:
                st.error("Negative (รีวิวเชิงลบ)")

            # ความมั่นใจ
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_input)[0][pred]
                st.write(f"ความมั่นใจ: {proba*100:.1f}%")
            elif hasattr(model, "decision_function"):
                margin = model.decision_function(X_input)[0]
                confidence = 1 / (1 + pow(2.718, -abs(margin)))  # sigmoid approx
                st.write(f"ความมั่นใจ: {confidence*100:.1f}%")
            else:
                st.write("โมเดลนี้ไม่รองรับการแสดงความมั่นใจ")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดระหว่างการแปลหรือทำนาย: {e}")
