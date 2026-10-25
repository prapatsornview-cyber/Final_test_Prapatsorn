"""ระบบประเมินความเสี่ยงสินเชื่อ (German Credit) — Streamlit

โมเดล Logistic Regression ที่เทรนไว้ถูกแปลงเป็นไฟล์พารามิเตอร์ (model_params.json)
คำนวณด้วย numpy จึงไม่ขึ้นกับเวอร์ชัน scikit-learn หรือ Python
ตัวเลือก dropdown อ่านจากพารามิเตอร์ของโมเดลโดยตรง
"""
from pathlib import Path

import json

import numpy as np
import streamlit as st

MODEL_PATH = Path(__file__).parent / "model_params.json"

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
    with open(MODEL_PATH, encoding="utf-8") as f:
        return json.load(f)


def predict_proba_bad(params: dict, values: dict) -> float:
    """คำนวณซ้ำขั้นตอนของ Pipeline: standardize ตัวเลข + one-hot หมวดหมู่ + logistic regression"""
    num = np.array([values[c] for c in params["num_cols"]], dtype=float)
    num = (num - np.array(params["num_mean"])) / np.array(params["num_scale"])
    cat = []
    for col, cats in zip(params["cat_cols"], params["cat_categories"]):
        cat.extend(1.0 if values[col] == c else 0.0 for c in cats)
    x = np.concatenate([num, np.array(cat)])
    z = float(x @ np.array(params["coef"]) + params["intercept"])
    return 1.0 / (1.0 + np.exp(-z))


try:
    params = load_model()
except Exception as exc:  # noqa: BLE001
    st.error("โหลดไฟล์ model_params.json ไม่สำเร็จ — ตรวจสอบว่าอัปโหลดไฟล์นี้ไว้ใน repository แล้ว")
    st.exception(exc)
    st.stop()

num_cols, cat_cols = params["num_cols"], params["cat_cols"]
cat_options = dict(zip(cat_cols, params["cat_categories"]))
threshold, model_name = params["threshold"], params["model_name"]

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
    prob = predict_proba_bad(params, values)

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
