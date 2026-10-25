"""ระบบประเมินความเสี่ยงสินเชื่อ (German Credit) — Streamlit

ตัวเลือกของทุกช่องแบบ dropdown ถูกอ่านจากตัวโมเดลที่เทรนไว้โดยตรง
จึงไม่เกิดปัญหาชื่อหมวดหมู่ไม่ตรงกับข้อมูลที่ใช้เทรน
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "best_credit_model.pkl"

LABELS = {
    "checking_status": "สถานะบัญชีเช็คกระแสรายวัน",
    "duration": "ระยะเวลาสินเชื่อ (เดือน)",
    "credit_history": "ประวัติเครดิต",
    "purpose": "วัตถุประสงค์การกู้",
    "credit_amount": "วงเงินสินเชื่อ (DM)",
    "savings_status": "สถานะบัญชีออมทรัพย์",
    "employment": "อายุงานปัจจุบัน (ปี)",
    "installment_commitment": "ภาระผ่อนชำระ (% ของรายได้ 1-4)",
    "personal_status": "สถานภาพส่วนบุคคลและเพศ",
    "other_parties": "ผู้ร่วมกู้/ผู้ค้ำประกัน",
    "residence_since": "อยู่อาศัยที่ปัจจุบันมาแล้ว (ปี 1-4)",
    "property_magnitude": "ทรัพย์สินที่มี",
    "age": "อายุ (ปี)",
    "other_payment_plans": "แผนผ่อนชำระอื่น",
    "housing": "ลักษณะที่อยู่อาศัย",
    "existing_credits": "จำนวนสินเชื่อที่มีอยู่ที่ธนาคารนี้",
    "job": "ลักษณะงาน",
    "num_dependents": "จำนวนผู้อยู่ในความอุปการะ",
    "own_telephone": "มีโทรศัพท์ในชื่อตนเอง",
    "foreign_worker": "แรงงานต่างชาติ",
}
NUMERIC_INPUTS = {  # คอลัมน์: (ต่ำสุด, สูงสุด, ค่าเริ่มต้น)
    "duration": (1, 72, 12),
    "credit_amount": (100, 20000, 3000),
    "installment_commitment": (1, 4, 2),
    "residence_since": (1, 4, 2),
    "age": (18, 100, 30),
    "existing_credits": (1, 4, 1),
    "num_dependents": (1, 2, 1),
}

st.set_page_config(page_title="Credit Risk Prediction", page_icon="💳", layout="centered")


@st.cache_resource
def load_model():
    package = joblib.load(MODEL_PATH)
    pre = package["model"].named_steps["preprocessor"]
    cat_cols, cat_options = [], {}
    for name, transformer, cols in pre.transformers_:
        if name == "cat":
            encoder = transformer.named_steps["onehot"]
            cat_cols = list(cols)
            cat_options = {c: list(cats) for c, cats in zip(cols, encoder.categories_)}
    num_cols = [c for name, _, cols in pre.transformers_ if name == "num" for c in cols]
    return package, num_cols, cat_cols, cat_options


try:
    package, num_cols, cat_cols, cat_options = load_model()
except Exception as exc:  # noqa: BLE001
    st.error(
        "โหลดไฟล์โมเดลไม่สำเร็จ — โดยมากเกิดจากเวอร์ชัน scikit-learn ไม่ตรงกับตอนเทรน "
        "(โมเดลนี้เทรนด้วย scikit-learn 1.6.1) หรือไม่ได้อัปโหลด best_credit_model.pkl ไว้ใน repository"
    )
    st.exception(exc)
    st.stop()

model, threshold, model_name = package["model"], package["threshold"], package["model_name"]

st.title("💳 ระบบประเมินความเสี่ยงสินเชื่อ")
st.write(f"โมเดล: **{model_name}** | Threshold: **{threshold:.2f}** (ต้นทุนผิดพลาด FN : FP = 5 : 1)")
st.markdown("---")
st.subheader("กรอกข้อมูลผู้ขอสินเชื่อ")

values = {}
left, right = st.columns(2)
for i, col in enumerate(cat_cols + num_cols):
    target = left if i % 2 == 0 else right
    with target:
        if col in cat_options:
            values[col] = st.selectbox(LABELS.get(col, col), cat_options[col], key=col)
        else:
            lo, hi, default = NUMERIC_INPUTS.get(col, (0, 100000, 1))
            values[col] = float(
                st.number_input(LABELS.get(col, col), min_value=lo, max_value=hi, value=default, key=col)
            )

if st.button("🎯 ประเมินความเสี่ยง"):
    df_input = pd.DataFrame([values])
    prob = float(model.predict_proba(df_input)[0, 1])

    st.markdown("---")
    st.subheader("ผลการวิเคราะห์")
    st.metric("โอกาสที่จะเป็นหนี้เสีย (Probability of Bad)", f"{prob:.2%}")
    st.progress(min(max(prob, 0.0), 1.0))

    if prob >= threshold:
        st.error("❌ ผลลัพธ์: ความเสี่ยงสูง — ไม่แนะนำให้อนุมัติ")
    else:
        st.success("✅ ผลลัพธ์: ความเสี่ยงต่ำ — อยู่ในเกณฑ์ที่ยอมรับได้")
    st.caption(f"เกณฑ์ตัดสินใจ: ความน่าจะเป็นตั้งแต่ {threshold:.2f} ขึ้นไปถือว่าความเสี่ยงสูง")

st.info(
    "ผลจากโมเดลเป็นเครื่องมือช่วยประกอบการพิจารณาเท่านั้น ไม่ควรใช้เป็นเกณฑ์เดียวในการตัดสินใจอนุมัติสินเชื่อ "
    "โดยเฉพาะตัวแปรอ่อนไหว เช่น เพศ อายุ และสถานะแรงงานต่างชาติ ควรผ่านการตรวจสอบด้านความเป็นธรรมก่อนนำไปใช้จริง"
)
