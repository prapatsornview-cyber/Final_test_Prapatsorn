"""
core.py — ฟังก์ชันหลักสำหรับข้อมูล โมเดล threshold การอธิบาย และความเป็นธรรม
ชุดข้อมูล: Statlog German Credit (OpenML data_id=31)
"""
from __future__ import annotations

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
LOCAL_CSV = Path(__file__).resolve().parent.parent / "data" / "credit-g.csv"


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
