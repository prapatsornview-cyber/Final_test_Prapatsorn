"""
German Credit Risk — Model, Explainability & Fairness
ชุดข้อมูล: Statlog German Credit (OpenML data_id=31)
ไฟล์เดียว ใช้เพียง streamlit, pandas, numpy, scikit-learn
"""
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="German Credit Risk", page_icon="💳", layout="wide")

# ตรวจว่าติดตั้ง scikit-learn แล้วหรือยัง (ป้องกัน ModuleNotFoundError)
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.datasets import fetch_openml
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                         cross_val_predict, train_test_split)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ModuleNotFoundError:
    st.error("ไม่พบ scikit-learn — โปรดตรวจว่ามีไฟล์ชื่อ requirements.txt "
             "(สะกดตรงตัว) อยู่ระดับเดียวกับไฟล์นี้ใน GitHub แล้วกด Reboot app")
    st.stop()

# ------------------------------------------------------------------
# ค่าคงที่ตามโจทย์
# ------------------------------------------------------------------
RANDOM_STATE = 42
TARGET, POS_LABEL = "class", "bad"
COST_FN, COST_FP = 5, 1
K = 5


# ------------------------------------------------------------------
# 1) โหลดข้อมูล
# ------------------------------------------------------------------
@st.cache_data(show_spinner="กำลังโหลดข้อมูลจาก OpenML (data_id=31)...")
def load_data():
    try:
        df = fetch_openml(data_id=31, as_frame=True).frame
    except Exception:
        df = pd.read_csv("data/credit-g.csv")  # สำรองกรณีออฟไลน์ (ถ้ามีไฟล์)
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(float)
        else:
            df[c] = df[c].astype(str)
    # ตัวแปรกลุ่มสำหรับ fairness (ไม่ใช้เป็น feature)
    df["sex"] = np.where(df["personal_status"].str.lower().str.startswith("female"),
                         "female", "male")
    df["age_group"] = np.where(df["age"] <= 25, "<=25", ">25")
    return df


# ------------------------------------------------------------------
# 2) โมเดล
# ------------------------------------------------------------------
def make_models():
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000),
            {"clf__C": [0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
            {"clf__max_depth": [None, 6, 12], "clf__min_samples_leaf": [1, 5]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
             "clf__max_depth": [2, 3]},
        ),
    }


def cost_of(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return COST_FN * fn + COST_FP * fp, fn, fp


def cost_curve(y_true, proba):
    rows = []
    for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
        c, fn, fp = cost_of(y_true, (proba >= t).astype(int))
        rows.append({"threshold": t, "cost": int(c), "FN": int(fn), "FP": int(fp)})
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="กำลังฝึกและจูนโมเดล (StratifiedKFold k=5) ประมาณ 1–2 นาที...")
def train(df):
    # 2.1 target และแบ่งข้อมูลแบบ stratified
    y = (df[TARGET] == POS_LABEL).astype(int)
    X = df.drop(columns=[TARGET, "sex", "age_group"])
    groups = df[["sex", "age_group"]]
    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
        X, y, groups, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    for name, (est, grid) in make_models().items():
        # 2.2 Pipeline = ColumnTransformer + โมเดล
        prep = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ])
        pipe = Pipeline([("prep", prep), ("clf", est)])
        # 2.3 จูนด้วย StratifiedKFold บนชุดฝึกเท่านั้น
        gs = GridSearchCV(pipe, grid, cv=skf, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        # 2.4 OOF predictions -> threshold ที่ cost ต่ำสุด
        oof = cross_val_predict(gs.best_estimator_, X_tr, y_tr, cv=skf,
                                method="predict_proba", n_jobs=-1)[:, 1]
        curve = cost_curve(y_tr, oof)
        best_t = float(curve.loc[curve["cost"].idxmin(), "threshold"])
        results[name] = {
            "model": gs.best_estimator_,
            "params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
            "cv_auc": gs.best_score_,
            "curve": curve,
            "threshold": best_t,
            "oof_cost": int(curve["cost"].min()),
            "oof_cost_05": int(curve.loc[curve["threshold"] == 0.5, "cost"].iloc[0]),
            "test_proba": gs.best_estimator_.predict_proba(X_te)[:, 1],
        }
    return X_tr, X_te, y_tr, y_te, g_te, results


def metrics(y_true, proba, t):
    y_pred = (proba >= t).astype(int)
    c, fn, fp = cost_of(y_true, y_pred)
    return {
        "threshold": t,
        "ROC-AUC": roc_auc_score(y_true, proba),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision(bad)": precision_score(y_true, y_pred, zero_division=0),
        "Recall(bad)": recall_score(y_true, y_pred, zero_division=0),
        "F1(bad)": f1_score(y_true, y_pred, zero_division=0),
        "FN": int(fn), "FP": int(fp), "Expected cost": int(c),
    }


# ------------------------------------------------------------------
# 4) ความเป็นธรรม
# ------------------------------------------------------------------
def fairness_table(y_true, y_pred, group):
    rows = []
    y_true, y_pred, group = np.asarray(y_true), np.asarray(y_pred), np.asarray(group)
    for g in sorted(set(group)):
        m = group == g
        tn, fp, fn, tp = confusion_matrix(y_true[m], y_pred[m], labels=[0, 1]).ravel()
        rows.append({
            "group": g, "n": int(m.sum()),
            "actual bad rate": y_true[m].mean(),
            "predicted bad rate": y_pred[m].mean(),
            "approval rate": 1 - y_pred[m].mean(),
            "TPR": tp / (tp + fn) if tp + fn else np.nan,
            "FPR": fp / (fp + tn) if fp + tn else np.nan,
            "accuracy": (tp + tn) / m.sum(),
        })
    return pd.DataFrame(rows)


def fairness_summary(tbl):
    gap = lambda c: tbl[c].max() - tbl[c].min()
    return {
        "Demographic parity diff.": gap("predicted bad rate"),
        "Disparate impact ratio": tbl["approval rate"].min() / tbl["approval rate"].max(),
        "Equal opportunity diff. (TPR)": gap("TPR"),
        "Equalized odds diff.": max(gap("TPR"), gap("FPR")),
    }


# ==================================================================
# UI
# ==================================================================
st.title("💳 German Credit Risk — Model, Explainability & Fairness")
st.caption("Statlog German Credit (OpenML data_id=31) · positive class = 'bad' · "
           "random_state=42 · StratifiedKFold k=5 · cost FN:FP = 5:1")

df = load_data()
X_tr, X_te, y_tr, y_te, g_te, results = train(df)
best_name = min(results, key=lambda n: results[n]["oof_cost"])
name = st.sidebar.selectbox("เลือกโมเดลสำหรับแท็บ 3–4", list(results),
                            index=list(results).index(best_name))
R = results[name]
st.sidebar.info(f"โมเดลที่ expected cost (OOF) ต่ำสุด: **{best_name}**")

tab1, tab2, tab3, tab4 = st.tabs(["1) EDA", "2) โมเดลและ Threshold",
                                  "3) Permutation Importance", "4) Fairness"])

# ---------------- 1) EDA ----------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("จำนวนแถว", df.shape[0])
    c2.metric("จำนวนตัวแปร (ไม่รวม target)", df.shape[1] - 3)
    c3.metric("ค่าว่าง", int(df.isna().sum().sum()))
    c4.metric("สัดส่วน bad", f"{(df[TARGET] == 'bad').mean():.1%}")

    st.subheader("ตัวอย่างข้อมูล")
    st.dataframe(df.head(10))

    st.subheader("สถิติเชิงพรรณนา (ตัวแปรเชิงปริมาณ)")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    st.dataframe(df[num_cols].describe().T.round(2))

    a, b = st.columns(2)
    with a:
        st.markdown("**การกระจายของ target**")
        st.bar_chart(df[TARGET].value_counts())
    with b:
        st.markdown("**อัตรา bad ตาม sex และ age_group**")
        st.bar_chart(pd.crosstab(df["sex"], df["age_group"],
                                 values=(df[TARGET] == "bad"), aggfunc="mean"))

    st.subheader("ค่าเฉลี่ยตัวแปรเชิงปริมาณแยกตาม class")
    st.dataframe(df.groupby(TARGET)[num_cols].mean().T.round(2))

    st.subheader("อัตรา bad ตามตัวแปรเชิงกลุ่ม")
    cat_cols = [c for c in df.columns if c not in num_cols + [TARGET]]
    col = st.selectbox("เลือกตัวแปร", cat_cols)
    st.bar_chart(df.groupby(col)[TARGET].apply(lambda s: (s == "bad").mean()))

    st.subheader("Correlation")
    st.dataframe(df[num_cols].corr().round(2))

# ---------------- 2) Model ----------------
with tab2:
    st.write(f"Train = {len(y_tr)} ราย (bad {y_tr.mean():.1%}) · "
             f"Test = {len(y_te)} ราย (bad {y_te.mean():.1%})")

    st.subheader("ผลการจูนบนชุดฝึก (5-fold CV) และ threshold จาก OOF")
    st.dataframe(pd.DataFrame([{
        "model": n, "CV ROC-AUC": round(r["cv_auc"], 3), "best params": str(r["params"]),
        "OOF cost @0.5": r["oof_cost_05"], "best threshold": r["threshold"],
        "OOF cost @best": r["oof_cost"]} for n, r in results.items()]), hide_index=True)

    st.markdown("**Expected cost (5·FN + 1·FP) บน OOF ตาม threshold**")
    st.line_chart(pd.DataFrame({n: r["curve"].set_index("threshold")["cost"]
                                for n, r in results.items()}))

    st.subheader("ผลบนชุดทดสอบ")
    rows = []
    for n, r in results.items():
        for t in (0.5, r["threshold"]):
            rows.append({"model": n, **metrics(y_te, r["test_proba"], t)})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)

    yp = (R["test_proba"] >= R["threshold"]).astype(int)
    st.markdown(f"**Confusion matrix — {name} @ threshold {R['threshold']:.2f}**")
    st.dataframe(pd.DataFrame(confusion_matrix(y_te, yp, labels=[0, 1]),
                              index=["actual good", "actual bad"],
                              columns=["pred good", "pred bad"]))

# ---------------- 3) Permutation importance ----------------
with tab3:
    st.subheader(f"Permutation importance บนชุดทดสอบ — {name}")
    st.caption("ค่าที่ลดลงของ ROC-AUC เมื่อสลับค่าตัวแปรแบบสุ่ม (n_repeats=10)")

    @st.cache_data(show_spinner="กำลังคำนวณ permutation importance...")
    def perm(model_name):
        pi = permutation_importance(results[model_name]["model"], X_te, y_te,
                                    scoring="roc_auc", n_repeats=10,
                                    random_state=RANDOM_STATE, n_jobs=-1)
        return (pd.DataFrame({"feature": X_te.columns,
                              "importance_mean": pi.importances_mean,
                              "importance_std": pi.importances_std})
                .sort_values("importance_mean", ascending=False))

    pi_df = perm(name)
    st.bar_chart(pi_df.set_index("feature")["importance_mean"])
    st.dataframe(pi_df.round(4), hide_index=True)

# ---------------- 4) Fairness ----------------
with tab4:
    st.subheader(f"Fairness metrics บนชุดทดสอบ — {name}")
    t = st.slider("threshold", 0.05, 0.95, float(R["threshold"]), 0.01)
    yp = (R["test_proba"] >= t).astype(int)
    for attr in ["sex", "age_group"]:
        st.markdown(f"#### ตาม `{attr}`")
        tbl = fairness_table(y_te, yp, g_te[attr])
        st.dataframe(tbl.round(3), hide_index=True)
        summ = fairness_summary(tbl)
        cols = st.columns(len(summ))
        for c, (k, v) in zip(cols, summ.items()):
            c.metric(k, f"{v:.3f}")
        if summ["Disparate impact ratio"] < 0.8:
            st.warning("Disparate impact ratio ต่ำกว่า 0.8 (four-fifths rule)")
    st.caption("sex สร้างจาก personal_status · age_group แบ่งที่อายุ 25 ปี · "
               "positive = bad ดังนั้น approval rate = สัดส่วนที่ถูกทำนายว่า good")
