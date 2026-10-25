"""
German Credit Risk — Model, Explainability & Fairness (ไฟล์เดียว)
รัน: streamlit run app.py
"""
from __future__ import annotations

# ====================== ส่วนที่ 1: ฟังก์ชันหลัก ======================

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------------------------------
# ค่าคงที่
# ------------------------------------------------------------------
RANDOM_STATE = 42
TARGET = "class"
POS_LABEL = "bad"          # positive class = 'bad'
COST_FN = 5                # ปล่อยกู้ให้ลูกค้าที่ผิดนัด (พลาด bad)
COST_FP = 1                # ปฏิเสธลูกค้าที่ดี
N_SPLITS = 5
SENSITIVE_RAW = ["personal_status", "age"]   # คอลัมน์ต้นทางของ sex และ age_group
LOCAL_CSV = Path(__file__).resolve().parent / "data" / "credit-g.csv"


# ------------------------------------------------------------------
# 1) ข้อมูล
# ------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """โหลดจาก OpenML; หากออฟไลน์จะใช้ไฟล์สำรองใน data/ (ถ้ามี)"""
    try:
        df = fetch_openml(data_id=31, as_frame=True).frame
    except Exception:
        if LOCAL_CSV.exists():
            df = pd.read_csv(LOCAL_CSV)
        else:
            raise
    df = df.copy()
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str)
    return df


def add_sensitive_groups(df: pd.DataFrame, age_cut: int = 25) -> pd.DataFrame:
    """สร้าง sex จาก personal_status และ age_group จาก age"""
    out = df.copy()
    ps = out["personal_status"].astype(str).str.lower()
    out["sex"] = np.where(ps.str.startswith("female"), "female", "male")
    out["age_group"] = np.where(out["age"].astype(float) <= age_cut,
                                f"<={age_cut}", f">{age_cut}")
    return out


def split_data(df: pd.DataFrame, test_size: float = 0.2,
               exclude_sensitive: bool = False):
    """2.1 กำหนด target และแยก train/test แบบ stratified (random_state=42)"""
    y = (df[TARGET] == POS_LABEL).astype(int)
    X = df.drop(columns=[TARGET])
    if exclude_sensitive:
        X = X.drop(columns=SENSITIVE_RAW)
    groups = add_sensitive_groups(df)[["sex", "age_group"]]
    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
        X, y, groups, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    return X_tr, X_te, y_tr, y_te, g_tr, g_te


# ------------------------------------------------------------------
# 2) Pipeline + โมเดล
# ------------------------------------------------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


def model_specs() -> dict:
    """2.2 อัลกอริทึม 3 แบบ พร้อม grid ของ hyperparameter"""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000),
            {"clf__C": [0.01, 0.1, 1, 10],
             "clf__class_weight": [None, "balanced"]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
            {"clf__max_depth": [None, 6, 12],
             "clf__min_samples_leaf": [1, 5],
             "clf__class_weight": [None, "balanced_subsample"]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"clf__n_estimators": [100, 200],
             "clf__learning_rate": [0.05, 0.1],
             "clf__max_depth": [2, 3]},
        ),
    }


def expected_cost(y_true, y_pred) -> tuple[int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(COST_FN * fn + COST_FP * fp), int(fn), int(fp)


def threshold_curve(y_true, proba) -> pd.DataFrame:
    """คำนวณ expected cost ทุก threshold ตั้งแต่ 0.01–0.99"""
    rows = []
    for t in np.round(np.arange(0.01, 1.00, 0.01), 2):
        cost, fn, fp = expected_cost(y_true, (proba >= t).astype(int))
        rows.append({"threshold": t, "cost": cost, "FN": fn, "FP": fp,
                     "cost_per_case": cost / len(y_true)})
    return pd.DataFrame(rows)


def best_threshold(curve: pd.DataFrame) -> float:
    return float(curve.loc[curve["cost"].idxmin(), "threshold"])


def train_all(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    2.3 จูน hyperparameter ด้วย StratifiedKFold (k=5) เฉพาะชุดฝึก
    2.4 สร้าง out-of-fold predictions แล้วเลือก threshold ที่ลด expected cost
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, (est, grid) in model_specs().items():
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("clf", est)])
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=skf, n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)

        oof = cross_val_predict(gs.best_estimator_, X_train, y_train, cv=skf,
                                method="predict_proba", n_jobs=-1)[:, 1]
        curve = threshold_curve(y_train, oof)
        t_star = best_threshold(curve)
        results[name] = {
            "model": gs.best_estimator_,
            "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
            "cv_auc": float(gs.best_score_),
            "cv_auc_std": float(gs.cv_results_["std_test_score"][gs.best_index_]),
            "oof_proba": oof,
            "oof_auc": float(roc_auc_score(y_train, oof)),
            "threshold_curve": curve,
            "threshold": t_star,
            "oof_cost": int(curve["cost"].min()),
            "oof_cost_at_05": int(curve.loc[curve["threshold"] == 0.5, "cost"].iloc[0]),
        }
    return results


def evaluate(y_true, proba, threshold: float) -> dict:
    y_pred = (proba >= threshold).astype(int)
    cost, fn, fp = expected_cost(y_true, y_pred)
    tn, _, _, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "ROC-AUC": roc_auc_score(y_true, proba),
        "PR-AUC": average_precision_score(y_true, proba),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (bad)": precision_score(y_true, y_pred, zero_division=0),
        "Recall (bad)": recall_score(y_true, y_pred, zero_division=0),
        "F1 (bad)": f1_score(y_true, y_pred, zero_division=0),
        "TN": int(tn), "FP": fp, "FN": fn, "TP": int(tp),
        "Expected cost": cost,
        "Cost per case": cost / len(y_true),
    }


# ------------------------------------------------------------------
# 3) การอธิบายโมเดล
# ------------------------------------------------------------------
def perm_importance(model, X_test, y_test, n_repeats: int = 10) -> pd.DataFrame:
    """Permutation importance บนชุดทดสอบ (ระดับตัวแปรต้นฉบับ, scoring=ROC-AUC)"""
    r = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                               n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1)
    return (pd.DataFrame({"feature": X_test.columns,
                          "importance_mean": r.importances_mean,
                          "importance_std": r.importances_std})
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True))


def shap_values(model, X_train, X_test):
    """SHAP บนชุดทดสอบ (ระดับตัวแปรหลัง one-hot) — คืน (values, Xt_df) หรือ None"""
    try:
        import shap
    except ImportError:
        return None
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    names = prep.get_feature_names_out()
    Xt = pd.DataFrame(prep.transform(X_test), columns=names, index=X_test.index)
    if isinstance(clf, LogisticRegression):
        bg = pd.DataFrame(prep.transform(X_train), columns=names)
        sv = shap.LinearExplainer(clf, bg).shap_values(Xt)
    else:
        sv = shap.TreeExplainer(clf).shap_values(Xt)
    if isinstance(sv, list):          # shap รุ่นเก่า: list ต่อ class
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # shap รุ่นใหม่: (n, features, classes)
        sv = sv[:, :, 1]
    return sv, Xt


# ------------------------------------------------------------------
# 4) ความเป็นธรรม
# ------------------------------------------------------------------
def _rate(num, den):
    return float(num) / float(den) if den else np.nan


def fairness_by_group(y_true, y_pred, group: pd.Series) -> pd.DataFrame:
    rows = []
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    g = np.asarray(group)
    for level in sorted(pd.unique(g)):
        m = g == level
        yt, yp = y_true[m], y_pred[m]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        rows.append({
            "group": level,
            "n": int(m.sum()),
            "actual bad rate": yt.mean(),
            "predicted bad rate": yp.mean(),
            "approval rate (pred good)": 1 - yp.mean(),
            "TPR (recall bad)": _rate(tp, tp + fn),
            "FPR (good ถูกปฏิเสธ)": _rate(fp, fp + tn),
            "FNR (bad หลุดผ่าน)": _rate(fn, fn + tp),
            "precision (bad)": _rate(tp, tp + fp),
            "accuracy": _rate(tp + tn, len(yt)),
            "cost per case": _rate(COST_FN * fn + COST_FP * fp, len(yt)),
        })
    return pd.DataFrame(rows)


def fairness_summary(tbl: pd.DataFrame) -> dict:
    def gap(col):
        return float(tbl[col].max() - tbl[col].min())
    appr = tbl["approval rate (pred good)"]
    return {
        "Demographic parity difference": gap("predicted bad rate"),
        "Disparate impact ratio (approval)": float(appr.min() / appr.max()) if appr.max() > 0 else np.nan,
        "Equal opportunity difference (TPR)": gap("TPR (recall bad)"),
        "FPR difference": gap("FPR (good ถูกปฏิเสธ)"),
        "Equalized odds difference": max(gap("TPR (recall bad)"), gap("FPR (good ถูกปฏิเสธ)")),
        "Accuracy difference": gap("accuracy"),
    }


import types as _types
core = _types.SimpleNamespace(**{k: v for k, v in dict(globals()).items() if not k.startswith('_')})

# ====================== ส่วนที่ 2: หน้าแอป Streamlit ======================

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="German Credit Risk", page_icon="💳", layout="wide")


# ------------------------------------------------------------------
# Cache
# ------------------------------------------------------------------
@st.cache_data(show_spinner="กำลังโหลดข้อมูลจาก OpenML (data_id=31)...")
def get_data():
    return core.load_data()


@st.cache_resource(show_spinner="กำลังจูนโมเดลด้วย StratifiedKFold (k=5)... ครั้งแรกใช้เวลาประมาณ 1–2 นาที")
def get_trained(test_size: float, exclude_sensitive: bool):
    df = get_data()
    X_tr, X_te, y_tr, y_te, g_tr, g_te = core.split_data(df, test_size, exclude_sensitive)
    results = core.train_all(X_tr, y_tr)
    for r in results.values():
        r["test_proba"] = r["model"].predict_proba(X_te)[:, 1]
    return dict(X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
                g_tr=g_tr, g_te=g_te, results=results)


@st.cache_data(show_spinner="กำลังคำนวณ permutation importance...")
def get_perm(_model, X_te, y_te, key):
    return core.perm_importance(_model, X_te, y_te)


@st.cache_data(show_spinner="กำลังคำนวณ SHAP values...")
def get_shap(_model, X_tr, X_te, key):
    return core.shap_values(_model, X_tr, X_te)


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.header("⚙️ การตั้งค่า")
test_size = st.sidebar.slider("สัดส่วนชุดทดสอบ", 0.15, 0.35, 0.20, 0.05)
exclude_sensitive = st.sidebar.checkbox(
    "ไม่ใช้ personal_status และ age เป็นตัวแปรในโมเดล", value=False,
    help="ใช้เปรียบเทียบผลด้านความเป็นธรรมเมื่อตัดตัวแปรอ่อนไหวออก")
st.sidebar.markdown(
    f"""
**ค่าคงที่ตามโจทย์**
- positive class = `'{core.POS_LABEL}'`
- random_state = {core.RANDOM_STATE}
- StratifiedKFold k = {core.N_SPLITS}
- ต้นทุน FN : FP = {core.COST_FN} : {core.COST_FP}
"""
)

st.title("💳 German Credit Risk — Model, Explainability & Fairness")
st.caption("ชุดข้อมูล Statlog German Credit (OpenML data_id=31) · 1,000 รายการ · 20 ตัวแปร")

df = get_data()
df_g = core.add_sensitive_groups(df)

tab_eda, tab_model, tab_xai, tab_fair, tab_pred = st.tabs([
    "1) ข้อมูลและ EDA", "2) โมเดลและ Threshold", "3) การอธิบายโมเดล",
    "4) ความเป็นธรรม", "5) ทดลองทำนาย",
])

# ==================================================================
# 1) EDA
# ==================================================================
with tab_eda:
    st.subheader("ภาพรวมข้อมูล")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("จำนวนแถว", f"{df.shape[0]:,}")
    c2.metric("จำนวนคอลัมน์", df.shape[1])
    c3.metric("ค่าว่างทั้งหมด", int(df.isna().sum().sum()))
    c4.metric("สัดส่วน bad", f"{(df[core.TARGET] == 'bad').mean():.1%}")

    st.dataframe(df.head(20))

    with st.expander("ชนิดข้อมูลและสถิติเบื้องต้น"):
        info = pd.DataFrame({"dtype": df.dtypes.astype(str),
                             "missing": df.isna().sum(),
                             "unique": df.nunique()})
        st.dataframe(info)
        st.dataframe(df.describe().T)

    left, right = st.columns(2)
    with left:
        cnt = df[core.TARGET].value_counts().reset_index()
        cnt.columns = ["class", "count"]
        st.plotly_chart(px.bar(cnt, x="class", y="count", color="class",
                               title="การกระจายของ target"))
    with right:
        grp = (df_g.groupby(["sex", "age_group"])[core.TARGET]
               .apply(lambda s: (s == "bad").mean()).reset_index(name="bad rate"))
        st.plotly_chart(px.bar(grp, x="sex", y="bad rate", color="age_group",
                               barmode="group", title="อัตรา bad ตาม sex และ age_group"))

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols + [core.TARGET]]

    st.subheader("ตัวแปรเชิงปริมาณ")
    num_sel = st.selectbox("เลือกตัวแปร", num_cols, index=num_cols.index("credit_amount"))
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(df, x=num_sel, color=core.TARGET, barmode="overlay",
                                nbins=30, title=f"Histogram: {num_sel}"))
    b.plotly_chart(px.box(df, x=core.TARGET, y=num_sel, color=core.TARGET,
                          title=f"Boxplot: {num_sel}"))

    corr = df[num_cols].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, title="Correlation (ตัวแปรเชิงปริมาณ)"))

    st.subheader("ตัวแปรเชิงกลุ่ม")
    cat_sel = st.selectbox("เลือกตัวแปร", cat_cols)
    rate = (df.groupby(cat_sel)[core.TARGET]
            .agg(n="size", bad_rate=lambda s: (s == "bad").mean())
            .reset_index().sort_values("bad_rate", ascending=False))
    st.plotly_chart(px.bar(rate, x=cat_sel, y="bad_rate", hover_data=["n"],
                           title=f"อัตรา bad ตาม {cat_sel}"))

# ------------------------------------------------------------------
# Train (ใช้ร่วมกันทุกแท็บถัดไป)
# ------------------------------------------------------------------
T = get_trained(test_size, exclude_sensitive)
results = T["results"]
best_name = min(results, key=lambda n: results[n]["oof_cost"])
model_name = st.sidebar.selectbox(
    "โมเดลที่ใช้วิเคราะห์ (แท็บ 3–5)", list(results),
    index=list(results).index(best_name),
    help="ค่าเริ่มต้นคือโมเดลที่มี expected cost จาก OOF ต่ำสุด")
R = results[model_name]
key = f"{model_name}|{test_size}|{exclude_sensitive}"

# ==================================================================
# 2) Model & threshold
# ==================================================================
with tab_model:
    st.subheader("การแบ่งข้อมูล")
    c1, c2, c3 = st.columns(3)
    c1.metric("Train", f"{len(T['y_tr'])} ราย", f"bad {T['y_tr'].mean():.1%}", delta_color="off")
    c2.metric("Test", f"{len(T['y_te'])} ราย", f"bad {T['y_te'].mean():.1%}", delta_color="off")
    c3.metric("โมเดลที่ cost OOF ต่ำสุด", best_name)

    st.subheader("ผลการจูน (CV บนชุดฝึก) และ threshold จาก OOF")
    rows = []
    for n, r in results.items():
        rows.append({"model": n,
                     "CV ROC-AUC (mean±sd)": f"{r['cv_auc']:.3f} ± {r['cv_auc_std']:.3f}",
                     "best params": str(r["best_params"]),
                     "threshold*": r["threshold"],
                     "OOF cost @0.5": r["oof_cost_at_05"],
                     "OOF cost @threshold*": r["oof_cost"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    fig = go.Figure()
    for n, r in results.items():
        cv = r["threshold_curve"]
        fig.add_trace(go.Scatter(x=cv["threshold"], y=cv["cost"], mode="lines", name=n))
        fig.add_vline(x=r["threshold"], line_dash="dot")
    fig.update_layout(title="Expected cost (5·FN + 1·FP) บน out-of-fold predictions",
                      xaxis_title="threshold", yaxis_title="cost")
    st.plotly_chart(fig)
    st.caption("เส้นประแสดง threshold ที่ให้ cost ต่ำสุดของแต่ละโมเดล "
               "(ค่าทางทฤษฎีเมื่อความน่าจะเป็นถูก calibrate คือ 1/(1+5) ≈ 0.167)")

    st.subheader("ประสิทธิภาพบนชุดทดสอบ")
    ev = []
    for n, r in results.items():
        for t, lab in [(0.5, "0.5"), (r["threshold"], "threshold*")]:
            e = core.evaluate(T["y_te"], r["test_proba"], t)
            ev.append({"model": n, "ใช้ threshold": lab, **e})
    ev = pd.DataFrame(ev)
    st.dataframe(ev.style.format({c: "{:.3f}" for c in
                 ["threshold", "ROC-AUC", "PR-AUC", "Accuracy", "Precision (bad)",
                  "Recall (bad)", "F1 (bad)", "Cost per case"]}), hide_index=True)

    e = core.evaluate(T["y_te"], R["test_proba"], R["threshold"])
    cm = np.array([[e["TN"], e["FP"]], [e["FN"], e["TP"]]])
    st.plotly_chart(px.imshow(cm, text_auto=True, x=["pred good", "pred bad"],
                              y=["actual good", "actual bad"], color_continuous_scale="Blues",
                              title=f"Confusion matrix — {model_name} @ {R['threshold']:.2f}"))

# ==================================================================
# 3) Explainability
# ==================================================================
with tab_xai:
    st.subheader(f"Permutation importance บนชุดทดสอบ — {model_name}")
    st.caption("วัดการลดลงของ ROC-AUC เมื่อสลับค่าตัวแปรแบบสุ่ม (n_repeats=10)")
    pi = get_perm(R["model"], T["X_te"], T["y_te"], key)
    st.plotly_chart(px.bar(pi.sort_values("importance_mean"), x="importance_mean", y="feature",
                           error_x="importance_std", orientation="h", height=600))

    st.subheader("SHAP บนชุดทดสอบ")
    out = get_shap(R["model"], T["X_tr"], T["X_te"], key)
    if out is None:
        st.info("ไม่พบไลบรารี shap — ติดตั้งด้วย `pip install shap`")
    else:
        sv, Xt = out
        imp = (pd.DataFrame({"feature": Xt.columns, "mean|SHAP|": np.abs(sv).mean(0)})
               .sort_values("mean|SHAP|", ascending=False).head(20))
        st.plotly_chart(px.bar(imp.sort_values("mean|SHAP|"), x="mean|SHAP|", y="feature",
                               orientation="h", height=600,
                               title="Top 20 ตัวแปร (หลัง one-hot) ตามค่า mean |SHAP|"))
        try:
            import matplotlib.pyplot as plt
            import shap
            plt.figure()
            shap.summary_plot(sv, Xt, max_display=15, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
        except Exception as ex:  # แสดงผลต่อได้แม้ beeswarm ล้มเหลว
            st.caption(f"ไม่สามารถแสดง beeswarm plot: {ex}")

        i = st.number_input("ดูคำอธิบายรายบุคคล: ลำดับลูกค้าในชุดทดสอบ", 0, len(Xt) - 1, 0)
        local = (pd.DataFrame({"feature": Xt.columns, "SHAP": sv[i]})
                 .assign(abs=lambda d: d["SHAP"].abs())
                 .sort_values("abs", ascending=False).head(12))
        st.write(f"ความน่าจะเป็น bad = **{R['test_proba'][i]:.3f}** · "
                 f"ค่าจริง = **{'bad' if T['y_te'].iloc[i] else 'good'}**")
        st.plotly_chart(px.bar(local.sort_values("SHAP"), x="SHAP", y="feature",
                               orientation="h", color=local.sort_values("SHAP")["SHAP"] > 0,
                               labels={"color": "เพิ่มความเสี่ยง"}))

# ==================================================================
# 4) Fairness
# ==================================================================
with tab_fair:
    st.subheader(f"Fairness metrics บนชุดทดสอบ — {model_name}")
    t_use = st.slider("threshold ที่ใช้ประเมิน", 0.05, 0.95, float(R["threshold"]), 0.01)
    y_pred = (R["test_proba"] >= t_use).astype(int)
    if exclude_sensitive:
        st.success("โหมดนี้ไม่ได้ใช้ personal_status และ age ในการฝึกโมเดล")

    for attr in ["sex", "age_group"]:
        st.markdown(f"#### ตาม `{attr}`")
        tbl = core.fairness_by_group(T["y_te"], y_pred, T["g_te"][attr])
        summ = core.fairness_summary(tbl)
        st.dataframe(tbl.style.format({c: "{:.3f}" for c in tbl.columns
                                       if c not in ("group", "n")}), hide_index=True)
        cols = st.columns(len(summ))
        for col, (k, v) in zip(cols, summ.items()):
            col.metric(k, f"{v:.3f}")
        di = summ["Disparate impact ratio (approval)"]
        if di < 0.8:
            st.warning(f"Disparate impact ratio = {di:.2f} ต่ำกว่าเกณฑ์ 0.8 (four-fifths rule)")
        long = tbl.melt(id_vars="group", value_vars=["predicted bad rate", "TPR (recall bad)",
                                                     "FPR (good ถูกปฏิเสธ)"])
        st.plotly_chart(px.bar(long, x="variable", y="value", color="group", barmode="group",
                               title=f"เปรียบเทียบอัตราสำคัญตาม {attr}"))

    with st.expander("ความหมายของตัวชี้วัด"):
        st.markdown("""
- **Demographic parity difference**: ส่วนต่างของสัดส่วนที่ถูกทำนายว่า bad ระหว่างกลุ่ม (ยิ่งใกล้ 0 ยิ่งดี)
- **Disparate impact ratio**: อัตราอนุมัติของกลุ่มต่ำสุด ÷ กลุ่มสูงสุด (ควร ≥ 0.8)
- **Equal opportunity difference**: ส่วนต่าง TPR — ความสามารถในการจับ bad ได้เท่ากันหรือไม่
- **Equalized odds difference**: ค่าสูงสุดของส่วนต่าง TPR และ FPR
- ข้อควรระวัง: ชุดทดสอบมีขนาดเล็ก บางกลุ่มมีจำนวนน้อย ค่าความต่างจึงมีความแปรปรวนสูง
- `sex` สร้างจาก `personal_status` (female/male) และ `age_group` แบ่งที่อายุ 25 ปี
""")

# ==================================================================
# 5) Prediction
# ==================================================================
with tab_pred:
    st.subheader(f"ทดลองประเมินผู้ขอสินเชื่อ 1 ราย — {model_name}")
    X_ref = T["X_tr"]
    with st.form("pred"):
        cols = st.columns(3)
        row = {}
        for i, c in enumerate(X_ref.columns):
            with cols[i % 3]:
                if pd.api.types.is_numeric_dtype(X_ref[c]):
                    row[c] = st.number_input(c, value=float(X_ref[c].median()),
                                             min_value=float(X_ref[c].min()),
                                             max_value=float(X_ref[c].max()))
                else:
                    opts = sorted(X_ref[c].unique())
                    mode = X_ref[c].mode().iloc[0]
                    row[c] = st.selectbox(c, opts, index=opts.index(mode))
        submitted = st.form_submit_button("ประเมินความเสี่ยง")
    if submitted:
        x = pd.DataFrame([row])[X_ref.columns]
        for c in X_ref.columns:
            x[c] = x[c].astype(X_ref[c].dtype)
        p = float(R["model"].predict_proba(x)[0, 1])
        decision = "bad → ปฏิเสธ/ทบทวนเพิ่มเติม" if p >= R["threshold"] else "good → อนุมัติ"
        c1, c2, c3 = st.columns(3)
        c1.metric("P(bad)", f"{p:.3f}")
        c2.metric("threshold", f"{R['threshold']:.2f}")
        c3.metric("ผลการตัดสิน", decision)

st.divider()
st.caption("จัดทำเพื่อการศึกษา · ผลลัพธ์ไม่ใช่คำแนะนำการปล่อยสินเชื่อจริง")
