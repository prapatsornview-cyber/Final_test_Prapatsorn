# German Credit Risk — Model, Explainability & Fairness (Streamlit)

แอปพลิเคชัน Streamlit สำหรับวิเคราะห์ความเสี่ยงสินเชื่อด้วยชุดข้อมูล **Statlog German Credit** (OpenML `data_id=31`)
ครอบคลุมการสำรวจข้อมูล การสร้างและปรับจูนโมเดล การเลือก threshold ตามต้นทุน การอธิบายโมเดล และการประเมินความเป็นธรรม

## ขอบเขตงานตามโจทย์

| ข้อ | รายละเอียด | ตำแหน่งในโค้ด / แอป |
|---|---|---|
| 1 | โหลดข้อมูลด้วย `fetch_openml(data_id=31)` และทำ EDA | `core.load_data()` · แท็บ 1 |
| 2.1 | target: positive class = `'bad'`, แยก train/test แบบ stratified (`random_state=42`) | `core.split_data()` |
| 2.2 | `Pipeline(ColumnTransformer + โมเดล)` 3 อัลกอริทึม: Logistic Regression, Random Forest, Gradient Boosting | `core.build_preprocessor()`, `core.model_specs()` |
| 2.3 | ปรับ hyperparameter ด้วย `GridSearchCV` + `StratifiedKFold(k=5)` บนชุดฝึกเท่านั้น | `core.train_all()` |
| 2.4 | เลือก threshold ที่ลด expected cost (FN:FP = 5:1) จาก out-of-fold predictions | `core.threshold_curve()` · แท็บ 2 |
| 3 | Permutation importance และ SHAP บนชุดทดสอบ | `core.perm_importance()`, `core.shap_values()` · แท็บ 3 |
| 4 | Fairness metrics ตาม `sex` และ `age_group` | `core.fairness_by_group()` · แท็บ 4 |
| 5 | แอป Streamlit เชื่อมต่อ GitHub และ Deploy | `app.py` · หัวข้อด้านล่าง |

## โครงสร้างโปรเจกต์

```
credit-risk-app/
├── app.py              # หน้าแอป Streamlit (5 แท็บ)
├── src/
│   ├── __init__.py
│   └── core.py         # ข้อมูล, pipeline, CV, threshold, XAI, fairness
├── requirements.txt
├── .gitignore
└── README.md
```

## รายละเอียดวิธีการ

- **Preprocessing**: ตัวแปรเชิงปริมาณใช้ `StandardScaler` ตัวแปรเชิงกลุ่มใช้ `OneHotEncoder(handle_unknown="ignore")`
- **การจูน**: ใช้ `scoring="roc_auc"` บน `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Threshold**: นำโมเดลที่ดีที่สุดไปสร้าง out-of-fold probability ด้วย `cross_val_predict` (fold เดียวกัน) แล้วค้นหา threshold 0.01–0.99 ที่ทำให้ `cost = 5·FN + 1·FP` ต่ำสุด จากนั้นจึงนำไปใช้กับชุดทดสอบ
- **ตัวแปรกลุ่ม**: `sex` สร้างจาก `personal_status` (ขึ้นต้นด้วย female/male) และ `age_group` แบ่งเป็น `<=25` และ `>25`
- **Fairness metrics**: demographic parity difference, disparate impact ratio (อัตราอนุมัติ), equal opportunity difference (TPR), FPR difference, equalized odds difference
- **ตัวเลือกเพิ่มเติม**: สามารถตัด `personal_status` และ `age` ออกจากโมเดลเพื่อเปรียบเทียบผลด้านความเป็นธรรม

## การรันบนเครื่อง

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

การเปิดแอปครั้งแรกจะใช้เวลาประมาณ 1–2 นาทีสำหรับการจูนโมเดล หลังจากนั้นผลลัพธ์จะถูกแคชไว้

## การเชื่อมต่อ GitHub และ Deploy บน Streamlit Community Cloud

1. สร้าง repository ใหม่บน GitHub (เช่น `german-credit-streamlit`) แบบ Public
2. อัปโหลดไฟล์ทั้งหมดในโฟลเดอร์นี้ หรือใช้คำสั่ง
   ```bash
   git init
   git add .
   git commit -m "Initial commit: German credit Streamlit app"
   git branch -M main
   git remote add origin https://github.com/<username>/german-credit-streamlit.git
   git push -u origin main
   ```
3. เข้า https://share.streamlit.io แล้วลงชื่อเข้าใช้ด้วยบัญชี GitHub
4. เลือก **Create app** → เลือก repository, branch `main` และ Main file path = `app.py`
5. กด **Deploy** ระบบจะติดตั้งแพ็กเกจจาก `requirements.txt` และให้ URL สำหรับส่งงาน
6. เมื่อ `git push` โค้ดใหม่ แอปจะอัปเดตอัตโนมัติ

## ข้อจำกัด

- ชุดทดสอบมีประมาณ 200 ราย บางกลุ่มย่อย (เช่น female, อายุ ≤25) มีขนาดเล็ก ค่าความต่างด้านความเป็นธรรมจึงมีความแปรปรวนสูง
- OOF predictions มาจาก fold เดียวกับที่ใช้จูน จึงอาจประเมินผลในแง่ดีเล็กน้อย ทั้งนี้ไม่มีการใช้ชุดทดสอบในการเลือก hyperparameter หรือ threshold
- `personal_status` ในชุดข้อมูลนี้ผสมสถานะสมรสกับเพศ การสร้างตัวแปร `sex` จึงเป็นการประมาณ
- งานนี้จัดทำเพื่อการศึกษา ไม่ใช่ระบบตัดสินใจสินเชื่อจริง
