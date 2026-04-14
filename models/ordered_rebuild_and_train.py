import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "02_processed"
FINAL_DIR = DATA_DIR / "03_final"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

KEY_COLS = ["co_ref", "prospect_renewal_date", "cutoff_date"]


def to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def yes_no_to_int(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["yes", "y", "true", "1"]).astype(int)


def parse_contact_count(value) -> float:
    """Simple internship-friendly parser for messy count values."""
    if pd.isna(value):
        return 0.0

    s = str(value).strip().lower()
    if s == "":
        return 0.0

    # Treat date-like strings as unknown count.
    if re.search(r"january|february|march|april|may|june|july|august|september|october|november|december", s):
        return 0.0

    # Common textual buckets.
    if any(k in s for k in ["no_email", "no email", "not discussed", "not specified", "not applicable"]):
        return 0.0
    if "less" in s:
        return 1.0
    if "few" in s:
        return 2.0
    if "several" in s:
        return 3.0
    if "multiple" in s:
        return 5.0
    if "many" in s:
        return 8.0

    nums = re.findall(r"\d+", s)
    if nums:
        n = float(nums[0])
        if "more than" in s or "+" in s:
            n += 1.0
        return float(min(n, 20.0))

    return 0.0


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    billings = pd.read_csv(PROCESSED_DIR / "processed_billings.csv", low_memory=False)
    renewal_calls = pd.read_csv(PROCESSED_DIR / "processed_renewal_calls.csv", low_memory=False)
    cc_calls = pd.read_csv(PROCESSED_DIR / "processed_cc_calls.csv", low_memory=False)
    emails = pd.read_csv(PROCESSED_DIR / "processed_emails.csv", low_memory=False)
    return billings, renewal_calls, cc_calls, emails


def build_master_snapshot(billings: pd.DataFrame) -> pd.DataFrame:
    df = billings.copy()
    df["prospect_renewal_date"] = to_datetime(df["prospect_renewal_date"])
    df["datetime_out"] = to_datetime(df["datetime_out"])

    # Keep only closed outcomes that can be used as supervised labels.
    base = df[df["prospect_outcome"].isin(["Won", "Churned"])].copy()
    base = base.dropna(subset=["co_ref", "prospect_renewal_date"])

    # Drop obvious date outliers that break time splits (only one row in this dataset).
    base = base[(base["prospect_renewal_date"].dt.year >= 2018) & (base["prospect_renewal_date"].dt.year <= 2035)]

    base["cutoff_date"] = base["prospect_renewal_date"] - pd.Timedelta(days=14)
    base["churn_14"] = (base["prospect_outcome"] == "Churned").astype(int)
    base["renewal_year"] = base["prospect_renewal_date"].dt.year

    keep = [
        "co_ref",
        "prospect_renewal_date",
        "cutoff_date",
        "renewal_year",
        "churn_14",
        "prospect_outcome",
        "tenure_years",
        "payment_method",
        "band",
        "proforma_auto_renewal",
        "proforma_world_pay_token",
    ]
    master = base[keep].copy()
    master = master.drop_duplicates(subset=["co_ref", "prospect_renewal_date"])
    return master


def build_billings_features(master: pd.DataFrame, billings: pd.DataFrame) -> pd.DataFrame:
    tx = billings[["co_ref", "datetime_out", "amount"]].copy()
    tx["datetime_out"] = to_datetime(tx["datetime_out"])
    tx["amount"] = to_num(tx["amount"]).fillna(0)

    feat = master[KEY_COLS].merge(tx, on="co_ref", how="left")
    lookback_start = feat["prospect_renewal_date"] - pd.Timedelta(days=365)
    feat = feat[(feat["datetime_out"] <= feat["cutoff_date"]) & (feat["datetime_out"] >= lookback_start)]

    grp = feat.groupby(KEY_COLS, dropna=False)
    out = grp.agg(
        total_spent=("amount", "sum"),
        avg_payment=("amount", "mean"),
        num_payments=("amount", "count"),
        last_payment_date=("datetime_out", "max"),
    ).reset_index()

    feat["days_before_cutoff"] = (feat["cutoff_date"] - feat["datetime_out"]).dt.days
    for w in [30, 90]:
        tmp = feat[feat["days_before_cutoff"] <= w].groupby(KEY_COLS, dropna=False).agg(
            **{f"payments_last_{w}": ("amount", "count"), f"spend_last_{w}": ("amount", "sum")}
        )
        out = out.merge(tmp.reset_index(), on=KEY_COLS, how="left")

    prev30 = feat[(feat["days_before_cutoff"] > 30) & (feat["days_before_cutoff"] <= 60)].groupby(KEY_COLS, dropna=False)[
        "amount"
    ].sum().rename("spend_prev_30")
    out = out.merge(prev30.reset_index(), on=KEY_COLS, how="left")
    out["payment_trend"] = out["spend_last_30"].fillna(0) - out["spend_prev_30"].fillna(0)
    out["days_since_last_payment"] = (out["cutoff_date"] - out["last_payment_date"]).dt.days

    out = out.drop(columns=["spend_prev_30"], errors="ignore")
    return out


def build_renewal_call_features(master: pd.DataFrame, renewal_calls: pd.DataFrame) -> pd.DataFrame:
    rc = renewal_calls.copy()
    rc["call_date"] = to_datetime(rc["call_date"])

    for c in [
        "serious_complaint",
        "explicit_switching_intent",
        "desire_to_cancel",
        "discussion_on_price_increase",
        "discount_or_waiver_requested",
    ]:
        rc[c] = yes_no_to_int(rc[c])

    feat = master[KEY_COLS].merge(rc, on="co_ref", how="left")
    lookback_start = feat["prospect_renewal_date"] - pd.Timedelta(days=365)
    feat = feat[(feat["call_date"] <= feat["cutoff_date"]) & (feat["call_date"] >= lookback_start)]

    grp = feat.groupby(KEY_COLS, dropna=False)
    out = grp.agg(
        total_calls=("call_id", "count"),
        last_call_date=("call_date", "max"),
        serious_complaints=("serious_complaint", "sum"),
        switch_intent=("explicit_switching_intent", "sum"),
        cancel_intent=("desire_to_cancel", "sum"),
        price_discussions=("discussion_on_price_increase", "sum"),
        discount_requests=("discount_or_waiver_requested", "sum"),
    ).reset_index()

    feat["days_before_cutoff"] = (feat["cutoff_date"] - feat["call_date"]).dt.days
    for w in [7, 14, 30]:
        tmp = feat[feat["days_before_cutoff"] <= w].groupby(KEY_COLS, dropna=False)["call_id"].count().rename(f"calls_last_{w}")
        out = out.merge(tmp.reset_index(), on=KEY_COLS, how="left")

    out["days_since_last_call"] = (out["cutoff_date"] - out["last_call_date"]).dt.days
    out["recent_call_ratio"] = out["calls_last_30"].fillna(0) / (out["total_calls"].replace(0, np.nan))
    out["recent_call_ratio"] = out["recent_call_ratio"].fillna(0)
    return out


def build_cc_call_features(master: pd.DataFrame, cc_calls: pd.DataFrame) -> pd.DataFrame:
    cc = cc_calls.copy()
    cc["call_date"] = to_datetime(cc["call_date"])

    for c in [
        "cc_customer_issues_concerns",
        "cc_business_struggles_financial_hardship",
        "cc_pricing_mentioned",
        "cc_contractor_complained",
    ]:
        cc[c] = yes_no_to_int(cc[c])

    cc["cc_contractor_sentiment_overall_score"] = to_num(cc["cc_contractor_sentiment_overall_score"])

    feat = master[KEY_COLS].merge(cc, on="co_ref", how="left")
    lookback_start = feat["prospect_renewal_date"] - pd.Timedelta(days=365)
    feat = feat[(feat["call_date"] <= feat["cutoff_date"]) & (feat["call_date"] >= lookback_start)]

    grp = feat.groupby(KEY_COLS, dropna=False)
    out = grp.agg(
        total_cc_calls=("contact_id", "count"),
        last_cc_call=("call_date", "max"),
        total_complaints=("cc_contractor_complained", "sum"),
        customer_issues=("cc_customer_issues_concerns", "sum"),
        financial_issues=("cc_business_struggles_financial_hardship", "sum"),
        pricing_mentions=("cc_pricing_mentioned", "sum"),
        avg_cc_sentiment=("cc_contractor_sentiment_overall_score", "mean"),
    ).reset_index()

    feat["days_before_cutoff"] = (feat["cutoff_date"] - feat["call_date"]).dt.days
    for w in [7, 30]:
        tmp = feat[feat["days_before_cutoff"] <= w].groupby(KEY_COLS, dropna=False)["contact_id"].count().rename(f"cc_calls_last_{w}")
        out = out.merge(tmp.reset_index(), on=KEY_COLS, how="left")

    out["days_since_last_cc_call"] = (out["cutoff_date"] - out["last_cc_call"]).dt.days
    out["repeat_call_ratio"] = out["cc_calls_last_30"].fillna(0) / (out["total_cc_calls"].replace(0, np.nan))
    out["repeat_call_ratio"] = out["repeat_call_ratio"].fillna(0)
    out["high_call_volume"] = (out["total_cc_calls"].fillna(0) >= 5).astype(int)
    return out


def build_email_features(master: pd.DataFrame, emails: pd.DataFrame) -> pd.DataFrame:
    em = emails.copy()
    em["year"] = to_num(em["year"]).astype("Int64")

    for c in [
        "crm_customer_complained",
        "crm_negative_customer_experience",
        "crm_dissatisfaction_with_support",
        "crm_financial_hardship_mentioned",
        "crm_dissatisified_with_renewal_price",
    ]:
        em[c] = yes_no_to_int(em[c])

    em["crm_contractor_sentiment_score"] = to_num(em["crm_contractor_sentiment_score"])
    em["crm_agent_chase_count"] = em["crm_agent_chase_count"].apply(parse_contact_count)

    feat = master[KEY_COLS + ["renewal_year"]].merge(em, left_on=["co_ref", "renewal_year"], right_on=["co_ref", "year"], how="left")

    feat["is_14_days_before"] = (feat["time_to_renewal"].astype(str).str.lower() == "14_out").astype(int)
    feat["is_pre_renewal"] = (feat["time_to_renewal"].astype(str).str.lower() == "pre_renewal").astype(int)

    grp = feat.groupby(KEY_COLS, dropna=False)
    out = grp.agg(
        total_interactions=("time_to_renewal", "count"),
        interactions_14_days=("is_14_days_before", "sum"),
        pre_renewal_interactions=("is_pre_renewal", "sum"),
        complaints=("crm_customer_complained", "sum"),
        negative_experience=("crm_negative_customer_experience", "sum"),
        support_issues=("crm_dissatisfaction_with_support", "sum"),
        financial_stress=("crm_financial_hardship_mentioned", "sum"),
        price_dissatisfaction=("crm_dissatisified_with_renewal_price", "sum"),
        avg_sentiment=("crm_contractor_sentiment_score", "mean"),
        agent_followups=("crm_agent_chase_count", "sum"),
    ).reset_index()

    out["last_moment_engagement_ratio"] = out["interactions_14_days"].fillna(0) / (
        out["total_interactions"].replace(0, np.nan)
    )
    out["last_moment_engagement_ratio"] = out["last_moment_engagement_ratio"].fillna(0)
    out["engagement_score"] = out["pre_renewal_interactions"].fillna(0) + 2 * out["interactions_14_days"].fillna(0)
    return out


def merge_dataset(master: pd.DataFrame, bf: pd.DataFrame, rf: pd.DataFrame, cf: pd.DataFrame, ef: pd.DataFrame) -> pd.DataFrame:
    data = master.copy()

    for feat in [bf, rf, cf, ef]:
        data = data.merge(feat, on=KEY_COLS, how="left")

    num_cols = data.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in data.columns if c not in num_cols and c not in ["prospect_renewal_date", "cutoff_date"]]

    data[num_cols] = data[num_cols].fillna(0)
    for c in cat_cols:
        data[c] = data[c].astype(str).replace("nan", "Unknown").fillna("Unknown")

    data["days_since_last_payment"] = np.maximum(data.get("days_since_last_payment", 0), 0)
    data["days_since_last_call"] = np.maximum(data.get("days_since_last_call", 0), 0)
    data["days_since_last_cc_call"] = np.maximum(data.get("days_since_last_cc_call", 0), 0)

    return data


def simplify_sparse_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop clearly non-informative numeric columns for a simple mini-project pipeline."""
    protected = {"churn_14"}
    dropped: list[str] = []

    num_cols = data.select_dtypes(include=["number"]).columns.tolist()
    for c in num_cols:
        if c in protected:
            continue

        s = data[c]
        nunique = int(s.nunique(dropna=True))
        zero_pct = float((s == 0).mean() * 100)

        # Always drop constant numeric columns.
        if nunique <= 1:
            dropped.append(c)
            continue

        # Drop extremely sparse low-cardinality columns to reduce noisy features.
        if zero_pct >= 99.5 and nunique <= 5:
            dropped.append(c)

    if dropped:
        data = data.drop(columns=sorted(set(dropped)), errors="ignore")

    return data, sorted(set(dropped))


def quality_report(data: pd.DataFrame, dropped_features: list[str] | None = None) -> dict:
    report = {
        "rows": int(len(data)),
        "unique_key_rows": int(data[KEY_COLS].drop_duplicates().shape[0]),
        "duplicate_key_rows": int(data.duplicated(subset=KEY_COLS).sum()),
        "label_distribution": data["churn_14"].value_counts(dropna=False).to_dict(),
        "negative_day_counts": {},
        "dropped_sparse_or_constant_features": dropped_features or [],
    }

    for c in ["days_since_last_payment", "days_since_last_call", "days_since_last_cc_call"]:
        if c in data.columns:
            report["negative_day_counts"][c] = int((data[c] < 0).sum())

    return report


def find_best_threshold(y_true: pd.Series, probs: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    if len(thresholds) == 0:
        return 0.5

    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-9, None)
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx])


def evaluate(y_true: pd.Series, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "threshold": float(threshold),
    }


def grouped_time_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Split by customer groups to avoid train/test customer leakage."""
    d = data.sort_values("prospect_renewal_date").copy()

    cust_dates = d.groupby("co_ref")["prospect_renewal_date"].max().sort_values().reset_index()
    n_cust = len(cust_dates)

    c70 = int(n_cust * 0.70)
    c85 = int(n_cust * 0.85)

    train_ids = set(cust_dates.iloc[:c70]["co_ref"])
    valid_ids = set(cust_dates.iloc[c70:c85]["co_ref"])
    test_ids = set(cust_dates.iloc[c85:]["co_ref"])

    train = d[d["co_ref"].isin(train_ids)].copy()
    valid = d[d["co_ref"].isin(valid_ids)].copy()
    test = d[d["co_ref"].isin(test_ids)].copy()

    # Fallback to row split if group split gets too small.
    if len(valid) < 500 or len(test) < 500:
        n = len(d)
        train = d.iloc[: int(n * 0.7)].copy()
        valid = d.iloc[int(n * 0.7) : int(n * 0.85)].copy()
        test = d.iloc[int(n * 0.85) :].copy()
        overlap = {
            "train_valid": int(len(set(train["co_ref"]).intersection(set(valid["co_ref"])))),
            "train_test": int(len(set(train["co_ref"]).intersection(set(test["co_ref"])))),
            "valid_test": int(len(set(valid["co_ref"]).intersection(set(test["co_ref"])))),
        }
        return train, valid, test, overlap

    overlap = {
        "train_valid": int(len(train_ids.intersection(valid_ids))),
        "train_test": int(len(train_ids.intersection(test_ids))),
        "valid_test": int(len(valid_ids.intersection(test_ids))),
    }
    return train, valid, test, overlap


def train_models(data: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    id_cols = ["co_ref", "prospect_renewal_date", "cutoff_date", "prospect_outcome"]
    target_col = "churn_14"
    feature_cols = [c for c in data.columns if c not in id_cols + [target_col]]

    train_df, valid_df, test_df, overlap = grouped_time_split(data)

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_valid, y_valid = valid_df[feature_cols], valid_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    cat_cols = X_train.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    metrics = {
        "split_sizes": {"train": int(len(train_df)), "valid": int(len(valid_df)), "test": int(len(test_df))},
        "customer_overlap": overlap,
        "models": {},
    }

    logit = Pipeline(
        steps=[
            ("prep", preprocess),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
        ]
    )
    logit.fit(X_train, y_train)
    val_prob = logit.predict_proba(X_valid)[:, 1]
    th = find_best_threshold(y_valid, val_prob)
    test_prob = logit.predict_proba(X_test)[:, 1]
    metrics["models"]["logistic"] = {
        "validation": evaluate(y_valid, val_prob, th),
        "test": evaluate(y_test, test_prob, th),
    }

    pred_df = test_df[["co_ref", "prospect_renewal_date", "cutoff_date", "churn_14"]].copy()
    pred_df["logistic_prob"] = test_prob

    if HAS_XGBOOST:
        pos = int(y_train.sum())
        neg = int((y_train == 0).sum())
        scale_pos_weight = float(neg / max(pos, 1))

        xgb = Pipeline(
            steps=[
                ("prep", preprocess),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=450,
                        learning_rate=0.05,
                        max_depth=5,
                        min_child_weight=2,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        reg_lambda=1.0,
                        random_state=42,
                        n_jobs=4,
                        scale_pos_weight=scale_pos_weight,
                    ),
                ),
            ]
        )

        xgb.fit(X_train, y_train)
        xgb_val_prob = xgb.predict_proba(X_valid)[:, 1]
        xgb_th = find_best_threshold(y_valid, xgb_val_prob)
        xgb_test_prob = xgb.predict_proba(X_test)[:, 1]

        metrics["models"]["xgboost"] = {
            "validation": evaluate(y_valid, xgb_val_prob, xgb_th),
            "test": evaluate(y_test, xgb_test_prob, xgb_th),
        }
        pred_df["xgboost_prob"] = xgb_test_prob

    return metrics, pred_df


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    billings, renewal_calls, cc_calls, emails = load_processed()

    master = build_master_snapshot(billings)
    bf = build_billings_features(master, billings)
    rf = build_renewal_call_features(master, renewal_calls)
    cf = build_cc_call_features(master, cc_calls)
    ef = build_email_features(master, emails)

    data = merge_dataset(master, bf, rf, cf, ef)
    data, dropped_features = simplify_sparse_features(data)
    q = quality_report(data, dropped_features=dropped_features)

    dataset_path = FINAL_DIR / "master_model_dataset.csv"
    data.to_csv(dataset_path, index=False)

    quality_path = REPORTS_DIR / "feature_quality_report.json"
    quality_path.write_text(json.dumps(q, indent=2), encoding="utf-8")

    metrics, pred_df = train_models(data)
    metrics["xgboost_available"] = HAS_XGBOOST

    metrics_path = REPORTS_DIR / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    preds_path = REPORTS_DIR / "test_predictions.csv"
    pred_df.to_csv(preds_path, index=False)

    print("Built dataset:", dataset_path)
    print("Quality report:", quality_path)
    print("Metrics:", metrics_path)
    print("Predictions:", preds_path)


if __name__ == "__main__":
    main()
