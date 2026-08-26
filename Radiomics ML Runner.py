#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radiomics ML Batch Runner
=========================
Batch training and external validation for EPE prediction.

The script reads one experiment-plan Excel file, builds radiomics/clinical
features, performs fold-safe feature selection and optional SMOTE, trains the
configured models, and writes model metrics, figures, final estimators, and
predictions.npz files for downstream subgroup analysis.
"""

import os
import re
import sys
import json
import argparse
import datetime
from pathlib import Path
from contextlib import contextmanager

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from sklearn.feature_selection import mutual_info_classif

try:
    import xgboost as xgb
    from xgboost.sklearn import XGBClassifier
    xgb.set_config(verbosity=0)
    XGBOOST_AVAILABLE = True
except ImportError:
    try:
        from xgboost import XGBClassifier
        XGBOOST_AVAILABLE = True
    except ImportError:
        XGBOOST_AVAILABLE = False

import joblib


# ============================================================
# 0) Defaults
# ============================================================
DEFAULT_CONFIG = {
    "OUTCOME_XLSX": "./Feature Data/Clinical Outcomes.xlsx",
    "OUTCOME_SHEET": "Sheet1",
    "SUV_XLSX": "./Feature Data/SUV-feature.xlsx",
    "T2W_XLSX": "./Feature Data/T2W-feature.xlsx",
    "ADC_XLSX": "./Feature Data/ADC-feature.xlsx",

    "SUV_SHEETS_DEFAULT": ["prostate"],
    "SUV_IMAGE_TYPES_DEFAULT": ["original"],
    "T2W_SHEETS_DEFAULT": ["prostate"],
    "T2W_IMAGE_TYPES_DEFAULT": ["original"],
    "ADC_SHEETS_DEFAULT": ["prostate"],
    "ADC_IMAGE_TYPES_DEFAULT": ["original"],

    "INCLUDE_CLINICAL": True,
    "CLINICAL_BASE_FEATURES": ["Age", "tPSA", "SUVmax"],
    "INCLUDE_bSIUP_GG": True,

    "RANDOM_STATE": 42,
    "N_SPLITS": 5,

    "RUN_SHAP": True,

    "Internal_MIN": None,      # Minimum internal sample ID
    "Internal_MAX": None,      # Maximum internal sample ID
    "Divide": None,            # Split ratio, e.g. "5:3"
    "Divide_Random": 42,       # Random seed for splitting

    "EXTERNAL_ID_MIN": None,   # Minimum external-validation sample ID
    "EXTERNAL_ID_MAX": None,   # Maximum external-validation sample ID

    "FS_TOPK": 30,                 # Final number of selected features
    "FS_MRMR_TOPK": 25,            # mRMR features retained per modality
    "FS_LASSO_TOPK": 7,            # LASSO features retained per modality
    "FS_CORR_THRESHOLD": 0.8,      # Spearman correlation threshold

    "SMOTE": False,
}

# ============================================================
# 1) Data Loading And Feature Assembly
# ============================================================

def _standardize_null(x):
    """Normalize NULL-like values to np.nan."""
    if x is None:
        return np.nan
    if isinstance(x, float) and np.isnan(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.upper() == "NULL" or s.upper() == "NAN":
        return np.nan
    return x


def clean_exception_message(msg):
    """Remove non-printable characters from exception messages."""
    if msg is None:
        return ""
    return ''.join(filter(lambda x: x.isprintable() or x in '\n\t\r', str(msg)))

def _to_numeric_or_nan(x):
    x = _standardize_null(x)
    if pd.isna(x):
        return np.nan
    try:
        return float(x)
    except Exception:
        s = str(x).strip()
        if s in ["0", "1"]:
            return float(s)
        m = re.findall(r"[-+]?\d*\.?\d+", s)
        if len(m) == 1:
            try:
                return float(m[0])
            except Exception:
                return np.nan
        return np.nan

def make_unique(names):
    seen = {}
    out = []
    for n in names:
        if n not in seen:
            seen[n] = 0
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}__dup{seen[n]}")
    return out

def read_outcomes(path, sheet_name=DEFAULT_CONFIG["OUTCOME_SHEET"]):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Outcomes 文件不存在：{path}")
    df = pd.read_excel(path, sheet_name=sheet_name)
    if "ID" not in df.columns:
        raise ValueError("Outcomes 表找不到 'ID' 列。")

    def _clean_id(x):
        x = _standardize_null(x)
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        try:
            return int(float(s))
        except Exception:
            digits = re.sub(r"\D", "", s)
            return int(digits) if digits != "" else np.nan

    df = df.copy()
    df["ID"] = df["ID"].apply(_clean_id)
    df = df.dropna(subset=["ID"]).astype({"ID": int})
    return df

def build_y_module(out_df):
    """
    Return EPE labels only: dict["EPE"] = Series(index=ID, values in {0,1} or NaN).
    """
    y_dict = {}
    tmp = out_df.set_index("ID")

    if "EPE" in tmp.columns:
        s = tmp["EPE"].map(_to_numeric_or_nan)
        y_dict["EPE"] = s.where(s.isin([0.0, 1.0]), np.nan).astype("float")
    else:
        print("[WARN] Outcomes 无列 EPE")

    return y_dict

def parse_pyradiomics_sheet(path, sheet_name, image_types=None, drop_diagnostics=True):
    """
    Parse a pyradiomics export sheet.

    Expected layout:
      - Row 0, columns 3 onward: patient IDs
      - Row 1, columns 3 onward: patient names
      - Row 3 onward: image type, feature class, feature name, then patient values

    Returns a patient-by-feature table indexed by ID.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature 文件不存在：{path}")

    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)

    patient_ids = raw.iloc[0, 3:].tolist()
    patient_names = raw.iloc[1, 3:].tolist()

    pid_clean = []
    for pid in patient_ids:
        pid = _standardize_null(pid)
        if pd.isna(pid):
            pid_clean.append(np.nan)
            continue
        s = str(pid).strip()
        try:
            pid_clean.append(int(float(s)))
        except Exception:
            digits = re.sub(r"\D", "", s)
            pid_clean.append(int(digits) if digits != "" else np.nan)

    meta = raw.iloc[3:, 0:3].copy()
    meta.columns = ["Image_type", "Feature_class", "Feature_name"]

    vals = raw.iloc[3:, 3:].copy()
    vals.columns = pid_clean

    image_type = meta["Image_type"].astype(str)
    mask = np.ones(len(meta), dtype=bool)

    if drop_diagnostics:
        mask &= (image_type != "diagnostics")

    if image_types is not None:
        if isinstance(image_types, str):
            image_types = [image_types]
        image_types = set([str(x) for x in image_types])
        mask &= meta["Image_type"].astype(str).isin(image_types)

    meta_f = meta.loc[mask].reset_index(drop=True)
    vals_f = vals.loc[mask].reset_index(drop=True)

    feat_names = (
        meta_f["Image_type"].astype(str)
        + "__" + meta_f["Feature_class"].astype(str)
        + "__" + meta_f["Feature_name"].astype(str)
    ).tolist()
    feat_names = make_unique(feat_names)

    vals_f = vals_f.apply(pd.to_numeric, errors="coerce")
    vals_f.index = feat_names

    X = vals_f.T
    X.index.name = "ID"
    X = X.dropna(axis=1, how="all")

    pid_map = pd.DataFrame({"ID": pid_clean, "Patient Name": patient_names}).set_index("ID")
    return X, pid_map, meta_f

def build_radiomics_module(
    path,
    sheets=("prostate",),
    image_types=("original",),
    modality_prefix="SUV"
):
    """
    Build one radiomics feature table from selected sheets and image types.

    Feature names are prefixed as {modality_prefix}__{sheet}__{feature_name}.
    """
    all_X = []
    for sh in sheets:
        X, _, _ = parse_pyradiomics_sheet(path, sh, image_types=image_types, drop_diagnostics=True)
        X = X.copy()
        X.columns = [f"{modality_prefix}__{sh}__{c}" for c in X.columns]
        all_X.append(X)

    if len(all_X) == 0:
        return pd.DataFrame()

    X_all = all_X[0]
    for X in all_X[1:]:
        X_all = X_all.join(X, how="outer")
    return X_all

def build_clinical_feature_module(
    out_df,
    include_bsiup=True,
    base_features=DEFAULT_CONFIG["CLINICAL_BASE_FEATURES"]
):
    """
    Build clinical features from the outcome table.

    Uses DEFAULT_CONFIG["CLINICAL_BASE_FEATURES"] plus optional bISUP GG.
    """
    tmp = out_df.set_index("ID").copy()
    cols_present = set(tmp.columns)
    clinical_cols = []

    for c in base_features:
        if c in cols_present:
            clinical_cols.append(c)

    Xc = tmp[clinical_cols].copy() if len(clinical_cols) else pd.DataFrame(index=tmp.index)

    for c in Xc.columns:
        Xc[c] = Xc[c].map(_to_numeric_or_nan)

    if include_bsiup:
        if "bISUP GG" in cols_present:
            Xc["bISUP GG"] = tmp["bISUP GG"].map(_to_numeric_or_nan)
        else:
            print("[WARN] Outcomes 无列 bISUP GG，已跳过。")

    if Xc.shape[1] > 0:
        Xc = Xc.rename(columns={c: f"CLIN__{c}" for c in Xc.columns})
    return Xc

def build_full_feature_table_custom(
    out_df,
    suv_path,
    t2w_path,
    adc_path,
    suv_sheets,
    suv_image_types,
    t2w_sheets,
    t2w_image_types,
    adc_sheets,
    adc_image_types,
    include_clinical,
    include_bsiup,
    clinical_base_features
):
    """Build the complete radiomics and clinical feature table."""
    parts = {}

    X_suv = build_radiomics_module(
        path=suv_path,
        sheets=suv_sheets,
        image_types=suv_image_types,
        modality_prefix="SUV"
    )
    parts["SUV"] = X_suv

    if t2w_path and os.path.exists(t2w_path):
        X_t2w = build_radiomics_module(
            path=t2w_path,
            sheets=t2w_sheets,
            image_types=t2w_image_types,
            modality_prefix="T2W"
        )
    else:
        print(f"[WARN] 找不到 T2W 文件：{t2w_path}（本次将不纳入 T2W 特征）")
        X_t2w = pd.DataFrame(index=X_suv.index if X_suv.shape[0] else out_df["ID"].unique())
    parts["T2W"] = X_t2w

    if adc_path and os.path.exists(adc_path):
        X_adc = build_radiomics_module(
            path=adc_path,
            sheets=adc_sheets,
            image_types=adc_image_types,
            modality_prefix="ADC"
        )
    else:
        print(f"[WARN] 找不到 ADC 文件：{adc_path}（本次将不纳入 ADC 特征）")
        X_adc = pd.DataFrame(index=X_suv.index if X_suv.shape[0] else out_df["ID"].unique())
    parts["ADC"] = X_adc

    if include_clinical:
        X_clin = build_clinical_feature_module(
            out_df,
            include_bsiup=include_bsiup,
            base_features=clinical_base_features
        )
    else:
        X_clin = pd.DataFrame(index=out_df["ID"].unique())
    parts["CLIN"] = X_clin

    ids = pd.Index(sorted(out_df["ID"].unique()), name="ID")
    X_all = pd.DataFrame(index=ids)

    for k, df_part in parts.items():
        if df_part is None or df_part.shape[1] == 0:
            continue
        df_part = df_part.copy()
        df_part.index = df_part.index.astype(int)
        X_all = X_all.join(df_part, how="left")

    return X_all, parts

def describe_label_and_features(y, X, title_prefix=""):
    vc = y.value_counts(dropna=False).sort_index()
    print(f"\n{title_prefix} Label counts (including NaN):")
    display(vc)

    miss = X.isna().mean().sort_values(ascending=False)
    print(f"{title_prefix} Feature missing rate: max={miss.max():.3f}, median={miss.median():.3f}")

def get_feature_names_after_preproc(pipe, original_feature_names):
    names = np.array(original_feature_names)
    if isinstance(pipe, Pipeline) and "var" in pipe.named_steps:
        var_step = pipe.named_steps["var"]
        if hasattr(var_step, "get_support"):
            mask = var_step.get_support()
            names = names[mask]
    return names.tolist()

class PerDatasetZScoreScaler(BaseEstimator, TransformerMixin):
    """
    Apply z-score scaling using the statistics of each input dataset.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0, ddof=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
        return (arr - mean) / std

def build_models():
    """
    Build the configured model dictionary.

    LinearSVC is wrapped with CalibratedClassifierCV so it can output
    calibrated probabilities.
    """
    models = {}

    models["LogReg_ElasticNet"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
        ("scaler", PerDatasetZScoreScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=8000,
            class_weight="balanced", random_state=RANDOM_STATE
        ))
    ])

    base_linsvc = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
        ("scaler", PerDatasetZScoreScaler()),
        ("clf", LinearSVC(
            C=1.0, dual="auto", class_weight="balanced", random_state=RANDOM_STATE
        ))
    ])
    models["Calib_LinearSVC"] = CalibratedClassifierCV(base_linsvc, method="sigmoid", cv=3)

    base_svm = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
        ("scaler", PerDatasetZScoreScaler()),
        ("clf", SVC(
            kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", random_state=RANDOM_STATE, probability=True
        ))
    ])
    models["SVM"] = base_svm

    models["RandomForest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
        ("clf", RandomForestClassifier(
            n_estimators=800, min_samples_leaf=1,
            class_weight="balanced", random_state=RANDOM_STATE
        ))
    ])

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(0.0)),
            ("clf", XGBClassifier(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,
                random_state=RANDOM_STATE,
                verbosity=0,
                eval_metric="logloss",
                tree_method="hist"
            ))
        ])

    return models

def safe_import_shap():
    try:
        import shap
        return shap
    except Exception as e:
        print("\n[WARN] shap import failed:", repr(e))
        print("如需 SHAP：可尝试：")
        print("  pip install -U shap numba")
        print('如仍冲突：pip install "coverage<7"')
        return None

# ============================================================
# 2) Batch Utilities
# ============================================================

def _is_nan(x) -> bool:
    try:
        return pd.isna(x)
    except Exception:
        return x is None

def _parse_bool(x, default=None):
    if _is_nan(x):
        return default
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    return default

def _parse_int(x, default=None):
    if _is_nan(x):
        return default
    try:
        return int(float(x))
    except Exception:
        return default



def _parse_list(x, default=None):
    """
    Parse list-like configuration values.

    Supports NaN/default values, Python lists or tuples, JSON list strings,
    and strings separated by commas, semicolons, or vertical bars.
    """
    if _is_nan(x):
        return default
    if isinstance(x, (list, tuple)):
        return list(x)
    s = str(x).strip()
    if s == "":
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            v = json.loads(s)
            return v if isinstance(v, list) else default
        except Exception:
            pass
    parts = re.split(r"[;,|]", s)
    parts = [p.strip() for p in parts if p.strip() != ""]
    return parts

def _sanitize_filename(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = re.sub(r"\s+", "_", s)
    return s

def _resolve_path(p: str, base_dir: Path) -> str:
    if p is None:
        return p
    p = str(p).strip()
    if p == "":
        return p
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((base_dir / pp).resolve())

class Tee:
    """Write output to multiple streams."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

class OutputManager:
    """
    Save display() tables and plt.show() figures under the active output tag.
    """
    def __init__(self, root_dir: Path, dpi: int = 300):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.current_dir = self.root_dir
        self.tag = "misc"
        self.fig_i = 0
        self.tbl_i = 0
        self.dpi = int(dpi)

    def set_dir(self, d: Path):
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        self.current_dir = d

    def set_tag(self, tag: str, reset_counter: bool = True):
        self.tag = _sanitize_filename(tag)
        if reset_counter:
            self.fig_i = 0
            self.tbl_i = 0

    def _next_fig_path(self, ext: str = "png") -> Path:
        self.fig_i += 1
        name = f"{self.tag}__fig{self.fig_i:03d}.{ext}"
        return self.current_dir / name

    def _next_tbl_path(self, ext: str = "xlsx") -> Path:
        self.tbl_i += 1
        name = f"{self.tag}__table{self.tbl_i:03d}.{ext}"
        return self.current_dir / name

    def save_all_open_figures(self):
        fignums = list(plt.get_fignums())
        if len(fignums) == 0:
            return
        figs = []
        for n in fignums:
            figs.append(plt.figure(n))
        for fig in figs:
            p = self._next_fig_path("png")
            fig.savefig(p, dpi=self.dpi, bbox_inches="tight")
        for fig in figs:
            try:
                plt.close(fig)
            except Exception:
                pass

    def save_table(self, obj):
        if isinstance(obj, pd.Series):
            df = obj.to_frame(name="value")
        elif isinstance(obj, pd.DataFrame):
            df = obj
        else:
            return
        p = self._next_tbl_path("xlsx")
        try:
            df.to_excel(p, index=True)
        except Exception:
            p = self._next_tbl_path("csv")
            df.to_csv(p, index=True, encoding="utf-8")

_OUTPUT_MANAGER = None

@contextmanager
def capture_outputs(out_mgr: OutputManager):
    global _OUTPUT_MANAGER
    orig_show = plt.show

    def _patched_show(*args, **kwargs):
        if _OUTPUT_MANAGER is None:
            return orig_show(*args, **kwargs)
        _OUTPUT_MANAGER.save_all_open_figures()

    plt.show = _patched_show
    _OUTPUT_MANAGER = out_mgr
    try:
        yield
    finally:
        plt.show = orig_show
        _OUTPUT_MANAGER = None

try:
    from IPython.display import display as _ipy_display
except Exception:
    _ipy_display = None

def display(obj):
    """Save tabular objects during capture and then display or print them."""
    if _OUTPUT_MANAGER is not None:
        try:
            _OUTPUT_MANAGER.save_table(obj)
        except Exception:
            pass
    if _ipy_display is not None:
        _ipy_display(obj)
    else:
        print(obj)

def write_settings_txt(out_dir: Path, cfg: dict, base_dir: Path = None):
    """
    Write runtime settings and package versions for one experiment.

    Spreadsheet paths are stored relative to base_dir when possible.
    """
    out_dir = Path(out_dir)
    if base_dir is None:
        base_dir = Path.cwd()

    def _rel_if_under_base(v: str) -> str:
        if not isinstance(v, str) or v.strip() == "":
            return v
        try:
            return str(Path(v).resolve().relative_to(base_dir.resolve()))
        except Exception:
            return v

    lines = []
    lines.append(f"timestamp\t{datetime.datetime.now().isoformat()}")
    lines.append(f"python\t{sys.version.replace(os.linesep, ' ')}")
    lines.append(f"pandas\t{pd.__version__}")
    lines.append(f"numpy\t{np.__version__}")
    try:
        import sklearn
        lines.append(f"sklearn\t{sklearn.__version__}")
    except Exception:
        pass
    lines.append("")
    lines.append("[CONFIG]")

    rel_keys = {"OUTCOME_XLSX", "SUV_XLSX", "T2W_XLSX", "ADC_XLSX"}
    for k in sorted(cfg.keys()):
        v = cfg[k]
        if k in rel_keys:
            v = _rel_if_under_base(v)
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)
        lines.append(f"{k}\t{v}")

    (out_dir / "settings.txt").write_text("\n".join(lines), encoding="utf-8")

def read_plan_xlsx(plan_xlsx: str, sheet_name: str = None) -> pd.DataFrame:
    plan_xlsx = str(plan_xlsx)
    if sheet_name is None:
        sheet_name = 0
    df = pd.read_excel(plan_xlsx, sheet_name=sheet_name)
    if "RUN" in df.columns:
        df = df[df["RUN"].map(lambda x: _parse_bool(x, True))].copy()
    df = df.reset_index(drop=True)
    return df

def normalize_config_row(row: pd.Series, base_dir: Path) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    for col, val in row.items():
        if col is None:
            continue
        c = str(col).strip()
        if c == "":
            continue
        cfg[c] = val

    for k in ["OUTCOME_XLSX", "SUV_XLSX", "T2W_XLSX", "ADC_XLSX"]:
        cfg[k] = _resolve_path(cfg.get(k, DEFAULT_CONFIG.get(k)), base_dir)

    if _is_nan(cfg.get("OUTCOME_SHEET")) or str(cfg.get("OUTCOME_SHEET")).strip() == "":
        cfg["OUTCOME_SHEET"] = DEFAULT_CONFIG["OUTCOME_SHEET"]
    else:
        cfg["OUTCOME_SHEET"] = str(cfg["OUTCOME_SHEET"]).strip()

    for k in [
        "SUV_SHEETS_DEFAULT", "SUV_IMAGE_TYPES_DEFAULT",
        "T2W_SHEETS_DEFAULT", "T2W_IMAGE_TYPES_DEFAULT",
        "ADC_SHEETS_DEFAULT", "ADC_IMAGE_TYPES_DEFAULT",
        "CLINICAL_BASE_FEATURES"
    ]:
        cfg[k] = _parse_list(cfg.get(k), DEFAULT_CONFIG.get(k))

    for k in ["INCLUDE_CLINICAL", "INCLUDE_bSIUP_GG", "RUN_SHAP", "SMOTE"]:
        cfg[k] = _parse_bool(cfg.get(k), DEFAULT_CONFIG.get(k))

    cfg["RANDOM_STATE"] = DEFAULT_CONFIG["RANDOM_STATE"]
    cfg["N_SPLITS"] = DEFAULT_CONFIG["N_SPLITS"]
    cfg["FS_TOPK"] = DEFAULT_CONFIG["FS_TOPK"]
    cfg["FS_MRMR_TOPK"] = DEFAULT_CONFIG["FS_MRMR_TOPK"]
    cfg["FS_LASSO_TOPK"] = DEFAULT_CONFIG["FS_LASSO_TOPK"]

    cfg["Internal_MIN"] = _parse_int(cfg.get("Internal_MIN"), None)
    cfg["Internal_MAX"] = _parse_int(cfg.get("Internal_MAX"), None)
    if _is_nan(cfg.get("Divide")):
        cfg["Divide"] = None
    else:
        cfg["Divide"] = str(cfg.get("Divide")).strip()
    cfg["Divide_Random"] = _parse_int(cfg.get("Divide_Random"), 42)
    cfg["EXTERNAL_ID_MIN"] = _parse_int(cfg.get("EXTERNAL_ID_MIN"), None)
    cfg["EXTERNAL_ID_MAX"] = _parse_int(cfg.get("EXTERNAL_ID_MAX"), None)

    cfg["ALGORITHMS"] = _parse_list(cfg.get("ALGORITHMS"), None)

    return cfg


# ============================================================
# 3) Train/Test split by internal ID interval and optional random division
# ============================================================

def split_train_test_by_closed_intervals(ids: pd.Index, cfg: dict, y=None):
    """
    Select the internal ID interval and optionally split it into train/test sets.

    If Divide is absent, all selected internal samples are used for training.
    When labels are provided, the random split is stratified by class.
    """
    ids = pd.Index(ids.astype(int), name="ID")
    lo_all = int(ids.min())
    hi_all = int(ids.max())

    internal_min = cfg.get("Internal_MIN")
    internal_max = cfg.get("Internal_MAX")
    divide_ratio = cfg.get("Divide")
    divide_random = cfg.get("Divide_Random", 42)

    if internal_min is not None and internal_max is not None:
        internal_min, internal_max = int(internal_min), int(internal_max)
        if internal_min > internal_max:
            raise ValueError(f"Bad internal interval: [{internal_min},{internal_max}] (min > max)")
        
        if internal_min < lo_all or internal_max > hi_all:
            raise ValueError(
                f"Internal ID interval out of bounds for this task. "
                f"Available ID range=[{lo_all},{hi_all}] but got [{internal_min},{internal_max}]"
            )
        
        filtered_ids = ids[(ids >= internal_min) & (ids <= internal_max)]
        if len(filtered_ids) == 0:
            raise ValueError(
                f"Empty sample after applying internal interval. "
                f"Internal range=[{internal_min},{internal_max}]"
            )
    else:
        filtered_ids = ids

    if divide_ratio is None:
        return filtered_ids, pd.Index([], name="ID"), False

    parts = divide_ratio.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid Divide format: {divide_ratio}. Expected format like '5:3'")
    
    try:
        train_ratio = int(parts[0])
        test_ratio = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid Divide ratio values: {divide_ratio}. Expected integers like '5:3'")

    if train_ratio <= 0 or test_ratio <= 0:
        raise ValueError(f"Divide ratio must be positive integers: {divide_ratio}")

    total = train_ratio + test_ratio
    train_size = int(len(filtered_ids) * train_ratio / total)
    
    if train_size == 0 or train_size == len(filtered_ids):
        raise ValueError(
            f"Divide ratio {divide_ratio} results in empty train or test set. "
            f"Total samples: {len(filtered_ids)}"
        )

    rng = np.random.RandomState(divide_random)
    
    if y is not None:
        y_filtered = y.loc[filtered_ids].values
        
        unique_classes = np.unique(y_filtered)
        train_indices = []
        test_indices = []
        
        for cls in unique_classes:
            cls_indices = np.where(y_filtered == cls)[0]
            cls_size = len(cls_indices)
            
            shuffled_cls_indices = rng.permutation(cls_indices)
            
            cls_train_size = int(cls_size * train_ratio / total)
            if cls_train_size == 0 and cls_size > 0:
                cls_train_size = 1
            if cls_train_size >= cls_size:
                cls_train_size = cls_size - 1
            
            train_indices.extend(shuffled_cls_indices[:cls_train_size])
            test_indices.extend(shuffled_cls_indices[cls_train_size:])
        
        train_mask = np.zeros(len(filtered_ids), dtype=bool)
        train_mask[train_indices] = True
    else:
        shuffled_indices = rng.permutation(len(filtered_ids))
        train_mask = np.zeros(len(filtered_ids), dtype=bool)
        train_mask[shuffled_indices[:train_size]] = True
    
    train_ids = filtered_ids[train_mask]
    test_ids = filtered_ids[~train_mask]

    return train_ids, test_ids, True


# ============================================================
# 4) Feature selection
# ============================================================



def fs_lasso(X_tr: pd.DataFrame, y_tr: np.ndarray, topk: int, random_state: int):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", PerDatasetZScoreScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=8000,
            class_weight="balanced", random_state=random_state
        ))
    ])
    
    param_grid = {
        "clf__C": [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512,
                   1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0],
        "clf__l1_ratio": [0.3]
    }
    
    vals, cnts = np.unique(y_tr, return_counts=True)
    if len(cnts) < 2 or int(cnts.min()) < 2:
        return list(X_tr.columns[:min(topk, X_tr.shape[1])])
    cv_splits = min(5, int(cnts.min()))

    grid_search = GridSearchCV(
        pipe, param_grid, cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state), 
        scoring="roc_auc", n_jobs=-1, verbose=0
    )
    grid_search.fit(X_tr, y_tr)
    
    best_pipe = grid_search.best_estimator_
    coef = best_pipe.named_steps["clf"].coef_.ravel()
    
    sorted_idx = np.argsort(np.abs(coef))[::-1]
    
    topk_idx = sorted_idx[:min(topk, len(coef))]
    
    sel = topk_idx[np.abs(coef[topk_idx]) > 1e-8]
    if len(sel) == 0:
        sel = topk_idx
    
    cols = list(X_tr.columns[sel])
    return cols

def fs_mrmr(X_tr: pd.DataFrame, y_tr: np.ndarray, topk: int, random_state: int):
    imp = SimpleImputer(strategy="median")
    Ximp = pd.DataFrame(imp.fit_transform(X_tr), columns=X_tr.columns)
    mi = mutual_info_classif(Ximp.values, y_tr, random_state=random_state)
    order = list(np.argsort(mi)[::-1])

    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(Ximp.values, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    selected = []
    for idx in order:
        if len(selected) == 0:
            selected.append(idx)
        else:
            selected.append(idx)
        if len(selected) >= topk:
            break

    cols = [X_tr.columns[i] for i in selected[:topk]]
    return cols

def fs_spearman_correlation(X_modality: pd.DataFrame, y_tr: np.ndarray, threshold: float = 0.8, random_state: int = 42):
    """
    Reduce correlated features within one modality using Spearman correlation.

    For highly correlated feature groups, the feature with the strongest
    univariate association with the outcome is retained.
    """
    if len(X_modality.columns) == 0:
        return []
    
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X_modality), columns=X_modality.columns)
    
    from scipy.stats import spearmanr
    feature_corr_with_y = {}
    for col in X_imputed.columns:
        corr, _ = spearmanr(X_imputed[col], y_tr)
        feature_corr_with_y[col] = abs(corr)
    
    corr_matrix = X_imputed.corr(method="spearman")
    
    selected = []
    remaining = list(X_imputed.columns)
    
    while remaining:
        current = max(remaining, key=lambda x: (feature_corr_with_y.get(x, 0), x))
        selected.append(current)
        remaining.remove(current)
        
        highly_correlated = []
        for col in remaining:
            if abs(corr_matrix.loc[current, col]) >= threshold:
                highly_correlated.append(col)
        
        for col in highly_correlated:
            remaining.remove(col)
    
    return selected



def run_feature_selection(X_tr: pd.DataFrame, y_tr: np.ndarray, cfg: dict):
    topk = int(cfg.get("FS_TOPK", 30))
    mrmr_topk = int(cfg.get("FS_MRMR_TOPK", 25))
    lasso_topk = int(cfg.get("FS_LASSO_TOPK", 7))
    rs = int(cfg.get("RANDOM_STATE", 42))
    
    suv_cols = [col for col in X_tr.columns if col.startswith("SUV__")]
    t2w_cols = [col for col in X_tr.columns if col.startswith("T2W__")]
    adc_cols = [col for col in X_tr.columns if col.startswith("ADC__")]
    clin_cols = [col for col in X_tr.columns if col.startswith("CLIN__")]
    
    selected_cols = []
    modality_steps = []
    
    if suv_cols:
        suv_corr_threshold = float(cfg.get("FS_CORR_THRESHOLD", 0.8))
        suv_X = X_tr[suv_cols]
        
        suv_corr_cols = fs_spearman_correlation(suv_X, y_tr, threshold=suv_corr_threshold, random_state=rs)
        
        modality_steps.append({
            "modality": "SUV", "method": "Spearman",
            "input_n": len(suv_cols), "selected_n": len(suv_corr_cols),
            "selected_features": suv_corr_cols
        })
        
        if suv_corr_cols:
            suv_corr_X = X_tr[suv_corr_cols]
            suv_mrmr_cols = fs_mrmr(suv_corr_X, y_tr, topk=mrmr_topk, random_state=rs)
            
            modality_steps.append({
                "modality": "SUV", "method": "mRMR",
                "input_n": len(suv_corr_cols), "selected_n": len(suv_mrmr_cols),
                "selected_features": suv_mrmr_cols
            })
            
            if suv_mrmr_cols:
                suv_mrmr_X = X_tr[suv_mrmr_cols]
                suv_lasso_cols = fs_lasso(suv_mrmr_X, y_tr, topk=lasso_topk, random_state=rs)
                selected_cols.extend(suv_lasso_cols)
                
                modality_steps.append({
                    "modality": "SUV", "method": "LASSO",
                    "input_n": len(suv_mrmr_cols), "selected_n": len(suv_lasso_cols),
                    "selected_features": suv_lasso_cols
                })
    
    if t2w_cols:
        t2w_corr_threshold = float(cfg.get("FS_CORR_THRESHOLD", 0.8))
        t2w_X = X_tr[t2w_cols]
        
        t2w_corr_cols = fs_spearman_correlation(t2w_X, y_tr, threshold=t2w_corr_threshold, random_state=rs)
        
        modality_steps.append({
            "modality": "T2W", "method": "Spearman",
            "input_n": len(t2w_cols), "selected_n": len(t2w_corr_cols),
            "selected_features": t2w_corr_cols
        })
        
        if t2w_corr_cols:
            t2w_corr_X = X_tr[t2w_corr_cols]
            t2w_mrmr_cols = fs_mrmr(t2w_corr_X, y_tr, topk=mrmr_topk, random_state=rs)
            
            modality_steps.append({
                "modality": "T2W", "method": "mRMR",
                "input_n": len(t2w_corr_cols), "selected_n": len(t2w_mrmr_cols),
                "selected_features": t2w_mrmr_cols
            })
            
            if t2w_mrmr_cols:
                t2w_mrmr_X = X_tr[t2w_mrmr_cols]
                t2w_lasso_cols = fs_lasso(t2w_mrmr_X, y_tr, topk=lasso_topk, random_state=rs)
                selected_cols.extend(t2w_lasso_cols)
                
                modality_steps.append({
                    "modality": "T2W", "method": "LASSO",
                    "input_n": len(t2w_mrmr_cols), "selected_n": len(t2w_lasso_cols),
                    "selected_features": t2w_lasso_cols
                })
    
    if adc_cols:
        adc_corr_threshold = float(cfg.get("FS_CORR_THRESHOLD", 0.8))
        adc_X = X_tr[adc_cols]
        
        adc_corr_cols = fs_spearman_correlation(adc_X, y_tr, threshold=adc_corr_threshold, random_state=rs)
        
        modality_steps.append({
            "modality": "ADC", "method": "Spearman",
            "input_n": len(adc_cols), "selected_n": len(adc_corr_cols),
            "selected_features": adc_corr_cols
        })
        
        if adc_corr_cols:
            adc_corr_X = X_tr[adc_corr_cols]
            adc_mrmr_cols = fs_mrmr(adc_corr_X, y_tr, topk=mrmr_topk, random_state=rs)
            
            modality_steps.append({
                "modality": "ADC", "method": "mRMR",
                "input_n": len(adc_corr_cols), "selected_n": len(adc_mrmr_cols),
                "selected_features": adc_mrmr_cols
            })
            
            if adc_mrmr_cols:
                adc_mrmr_X = X_tr[adc_mrmr_cols]
                adc_lasso_cols = fs_lasso(adc_mrmr_X, y_tr, topk=lasso_topk, random_state=rs)
                selected_cols.extend(adc_lasso_cols)
                
                modality_steps.append({
                    "modality": "ADC", "method": "LASSO",
                    "input_n": len(adc_mrmr_cols), "selected_n": len(adc_lasso_cols),
                    "selected_features": adc_lasso_cols
                })
    
    selected_cols.extend(clin_cols)
    
    if len(selected_cols) > topk:
        final_cols = selected_cols[:topk]
    else:
        final_cols = selected_cols
    
    meta = {
        "selected_n": len(final_cols),
        "detail": {
            "modality_steps": modality_steps,
            "total_selected_before_truncation": len(selected_cols)
        },
        "mode": "per_modality",
        "selected_features": final_cols,
    }
    return final_cols, meta


# ============================================================
# 5) SMOTE after feature selection
# ============================================================

def _maybe_smote_xy(X_num: np.ndarray, y: np.ndarray, random_state: int):
    """
    Apply SMOTE when class counts allow it.

    If the minority class has fewer than two samples, or if SMOTE fails, the
    original data are returned unchanged.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except Exception as e:
        print("[WARN] imblearn/SMOTE not available, skipping SMOTE. err=", repr(e))
        return X_num, y

    try:
        y = np.asarray(y)
        vals, cnts = np.unique(y, return_counts=True)
        if len(cnts) < 2:
            return X_num, y
        minority = int(cnts.min())

        if minority < 2:
            print(f"[SMOTE] Skip: minority_count={minority} (<2).")
            return X_num, y

        k = min(5, minority - 1)
        sm = SMOTE(random_state=random_state, k_neighbors=k)
        X_res, y_res = sm.fit_resample(X_num, y)
        return X_res, y_res

    except Exception as e:
        print("[WARN] SMOTE failed, skipping SMOTE. err=", repr(e))
        return X_num, y



def _impute_df(X_fit_df: pd.DataFrame, X_apply_df: pd.DataFrame):
    """
    Fit SimpleImputer on X_fit_df and transform X_apply_df.

    The result is returned as a DataFrame with feature names and indices.
    """
    imp = SimpleImputer(strategy="median")
    imp.fit(X_fit_df)
    X_out = imp.transform(X_apply_df)
    return pd.DataFrame(X_out, columns=X_apply_df.columns, index=X_apply_df.index)


# ============================================================
# 6) OOF scoring helper with optional SMOTE inside CV fold
# ============================================================

def oof_scores_with_optional_smote(estimator, X_df, y, cv, use_smote: bool, random_state: int,
                                   fs_cfg: dict = None, fs_cache: dict = None):
    """
    Compute out-of-fold scores with optional fold-local feature selection and SMOTE.

    SMOTE is applied only to each fold's training subset.
    """
    scores = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in cv.split(X_df, y):
        X_tr = X_df.iloc[train_idx]
        y_tr = y[train_idx]
        X_te = X_df.iloc[test_idx]

        if fs_cfg is not None:
            cache_key = tuple(X_tr.index.tolist())
            if fs_cache is not None and cache_key in fs_cache:
                sel_cols = fs_cache[cache_key]
            else:
                sel_cols, _ = run_feature_selection(X_tr, y_tr, fs_cfg)
                if fs_cache is not None:
                    fs_cache[cache_key] = sel_cols
            if len(sel_cols) == 0:
                raise ValueError("Feature selection selected 0 features inside CV fold.")
            X_tr = X_tr[sel_cols]
            X_te = X_te[sel_cols]

        est = clone(estimator)

        if use_smote:
            imp = SimpleImputer(strategy="median")
            X_tr_imp = imp.fit_transform(X_tr)
            X_te_imp = imp.transform(X_te)

            X_tr_sm, y_tr_sm = _maybe_smote_xy(X_tr_imp, y_tr, random_state=random_state)
            est.fit(X_tr_sm, y_tr_sm)
            X_eval = X_te_imp
        else:
            est.fit(X_tr, y_tr)
            X_eval = X_te

        if hasattr(est, "predict_proba"):
            s = est.predict_proba(X_eval)[:, 1]
        elif hasattr(est, "decision_function"):
            s = est.decision_function(X_eval)
        else:
            s = est.predict(X_eval).astype(float)

        scores[test_idx] = s

    return scores


def build_elasticnet_pipeline(C: float, l1_ratio: float, random_state: int):
    """Build the elastic-net logistic regression pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
        ("scaler", PerDatasetZScoreScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga", C=C, l1_ratio=l1_ratio,
            max_iter=8000, class_weight="balanced", random_state=random_state
        ))
    ])


def build_linsvc_pipeline(C: float, random_state: int):
    """Build the base LinearSVC pipeline used for calibration."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
        ("scaler", PerDatasetZScoreScaler()),
        ("clf", LinearSVC(C=C, dual="auto", class_weight="balanced", random_state=random_state))
    ])


def build_calib_linsvc(C: float, random_state: int):
    """Build a probability-calibrated LinearSVC model."""
    base = build_linsvc_pipeline(C, random_state)
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def build_svm_pipeline(C: float, gamma, class_weight, random_state: int):
    """Build the RBF-kernel SVM pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
        ("scaler", PerDatasetZScoreScaler()),
        ("clf", SVC(kernel="rbf", C=C, gamma=gamma,
                     class_weight=class_weight, random_state=random_state, probability=True))
    ])


def build_rf_pipeline(n_estimators: int, max_depth, min_samples_leaf: int, random_state: int):
    """Build the random forest pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced", random_state=random_state
        ))
    ])


def build_xgb_pipeline(n_estimators: int, max_depth: int, learning_rate: float,
                       subsample: float, colsample_bytree: float, random_state: int):
    """Build the XGBoost pipeline."""
    kwargs = dict(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=subsample,
        colsample_bytree=colsample_bytree, scale_pos_weight=1,
        random_state=random_state, verbosity=0, eval_metric="logloss"
    )
    if hasattr(xgb, "set_config"):
        kwargs["tree_method"] = "hist"
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
        ("clf", XGBClassifier(**kwargs))
    ])


def _iter_grid(grid_dict):
    """Yield all parameter combinations from a grid dictionary."""
    keys = list(grid_dict.keys())
    if len(keys) == 0:
        yield {}
        return

    first_key = keys[0]
    first_vals = grid_dict[first_key]
    for combo in _iter_grid({k: v for k, v in grid_dict.items() if k != first_key}):
        for val in first_vals:
            yield {first_key: val, **combo}


def _params_sort_key(model_name: str, params: dict):
    """Return a deterministic tie-break key for parameter search."""
    if model_name in ("LogReg_ElasticNet", "Calib_LinearSVC", "SVM"):
        return params.get("C", 0)
    elif model_name == "RandomForest":
        return (params.get("n_estimators", 0), params.get("max_depth", 0) if params.get("max_depth") is not None else 10**9)
    elif model_name == "XGBoost":
        return (params.get("learning_rate", 1.0), params.get("n_estimators", 0))
    return ()


def tune_model_params(model_name: str, builder, grid: dict, fallback_params: dict,
                      X_tr: pd.DataFrame, y_tr: np.ndarray,
                      cv, use_smote: bool, random_state: int):
    """
    Tune hyperparameters by cross-validated OOF AUC.

    Returns the best parameter dictionary, its OOF AUC, and the corresponding
    OOF scores.
    """
    combos = list(_iter_grid(grid))
    combos.sort(key=lambda p: _params_sort_key(model_name, p))

    best_auc = -1.0
    best_params = combos[0] if combos else fallback_params
    best_scores = None

    n_combos = len(combos)
    for idx, params in enumerate(combos, start=1):
        try:
            est = builder(**params, random_state=random_state)
            scores = oof_scores_with_optional_smote(
                est, X_tr, y_tr, cv=cv,
                use_smote=use_smote, random_state=random_state
            )
            fpr_, tpr_, _ = roc_curve(y_tr, scores)
            auc_val = auc(fpr_, tpr_)
            print(f"[TUNE] {model_name} ({idx}/{n_combos}) {params} -> OOF AUC={auc_val:.4f}")

            if auc_val > best_auc:
                best_auc = auc_val
                best_params = params
                best_scores = scores
        except Exception as e:
            print(f"[TUNE] {model_name} {params} failed: {clean_exception_message(repr(e))}")

    if best_scores is None:
        print(f"[TUNE] {model_name} All combos failed, falling back to {fallback_params}")
        est = builder(**fallback_params, random_state=random_state)
        best_scores = oof_scores_with_optional_smote(
            est, X_tr, y_tr, cv=cv, use_smote=use_smote, random_state=random_state
        )
        fpr_, tpr_, _ = roc_curve(y_tr, best_scores)
        best_auc = auc(fpr_, tpr_)
        best_params = fallback_params

    return best_params, float(best_auc), best_scores


def nested_cv_tune_and_score(model_name: str, builder, grid: dict, fallback_params: dict,
                              X_df: pd.DataFrame, y: np.ndarray,
                              outer_cv, inner_cv_folds: int = 3,
                              use_smote: bool = False, random_state: int = 42,
                              fs_cfg: dict = None):
    """
    Compute OOF scores with nested cross-validation.

    Hyperparameters are tuned inside each outer fold. Feature selection and
    optional SMOTE are also confined to the relevant training folds.
    """
    n_samples = len(y)
    oof_scores = np.zeros(n_samples, dtype=float)
    oof_scores[:] = np.nan

    fold_best_params = []
    fs_cache = {} if fs_cfg is not None else None

    inner_cv = StratifiedKFold(n_splits=inner_cv_folds, shuffle=True, random_state=random_state)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_df, y)):
        X_tr_fold = X_df.iloc[train_idx]
        y_tr_fold = y[train_idx]
        X_te_fold = X_df.iloc[test_idx]
        y_te_fold = y[test_idx]

        combos = list(_iter_grid(grid))
        combos.sort(key=lambda p: _params_sort_key(model_name, p))

        best_inner_auc = -1.0
        best_inner_params = combos[0] if combos else fallback_params

        for params in combos:
            try:
                inner_scores = oof_scores_with_optional_smote(
                    builder(**params, random_state=random_state),
                    X_tr_fold, y_tr_fold, cv=inner_cv,
                    use_smote=use_smote, random_state=random_state,
                    fs_cfg=fs_cfg, fs_cache=fs_cache
                )
                fpr_, tpr_, _ = roc_curve(y_tr_fold, inner_scores)
                inner_auc = auc(fpr_, tpr_)

                if inner_auc > best_inner_auc:
                    best_inner_auc = inner_auc
                    best_inner_params = params
            except Exception:
                pass

        if best_inner_auc < 0:
            best_inner_params = fallback_params

        fold_best_params.append(best_inner_params)

        est = builder(**best_inner_params, random_state=random_state)

        if fs_cfg is not None:
            cache_key = tuple(X_tr_fold.index.tolist())
            if fs_cache is not None and cache_key in fs_cache:
                sel_cols_fold = fs_cache[cache_key]
            else:
                sel_cols_fold, _ = run_feature_selection(X_tr_fold, y_tr_fold, fs_cfg)
                if fs_cache is not None:
                    fs_cache[cache_key] = sel_cols_fold
            if len(sel_cols_fold) == 0:
                raise ValueError("Feature selection selected 0 features in outer CV fold.")
            X_tr_fold = X_tr_fold[sel_cols_fold]
            X_te_fold = X_te_fold[sel_cols_fold]

        if use_smote:
            imp = SimpleImputer(strategy="median")
            X_tr_imp = imp.fit_transform(X_tr_fold)
            X_te_imp = imp.transform(X_te_fold)
            X_tr_sm, y_tr_sm = _maybe_smote_xy(X_tr_imp, y_tr_fold, random_state=random_state)
            est.fit(X_tr_sm, y_tr_sm)
            X_eval = X_te_imp
        else:
            est.fit(X_tr_fold, y_tr_fold)
            X_eval = X_te_fold

        if hasattr(est, "predict_proba"):
            s = est.predict_proba(X_eval)[:, 1]
        elif hasattr(est, "decision_function"):
            s = est.decision_function(X_eval)
        else:
            s = est.predict(X_eval).astype(float)

        oof_scores[test_idx] = s
        print(f"[NESTED-CV] {model_name} Fold {fold_idx+1}/{outer_cv.get_n_splits()}: "
              f"best_params={best_inner_params}, inner_auc={best_inner_auc:.4f}")

    print(f"[NESTED-CV] {model_name} Outer OOF scoring complete. "
          "Final params will be searched on the all-training selected feature set.")

    return oof_scores, None, float('nan')


def save_model_to_file(model, model_name: str, task_name: str, exp_dir, suffix: str = "final"):
    """Save a fitted model to the task-level models directory."""
    task_s = _sanitize_filename(task_name)
    model_s = _sanitize_filename(model_name)
    task_dir = exp_dir / f"task_{task_s}" / "models"
    task_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = task_dir / f"{model_s}_{suffix}.joblib"
    joblib.dump(model, model_path)
    print(f"[SAVE] Model saved: {model_path}")
    return model_path


TUNE_CONFIGS = {
    "LogReg_ElasticNet": (
        build_elasticnet_pipeline,
        {"C": [0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1.0, 2.0, 4.0, 10.0],
         "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0]},
        {"C": 1.0, "l1_ratio": 0.5},
    ),
    "Calib_LinearSVC": (
        build_calib_linsvc,
        {"C": [0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1.0, 2.0, 4.0, 10.0]},
        {"C": 1.0},
    ),
    "SVM": (
        build_svm_pipeline,
        {"C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
         "gamma": ["scale", 0.001, 0.01, 0.1, 1.0],
         "class_weight": ["balanced"]},
        {"C": 1.0, "gamma": "scale", "class_weight": "balanced"},
    ),
    "RandomForest": (
        build_rf_pipeline,
        {"n_estimators": [50, 100, 200, 300],
         "max_depth": [3, 5, 8, None],
         "min_samples_leaf": [1, 2, 3]},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
    ),
    "XGBoost": (
        build_xgb_pipeline,
        {"n_estimators": [100, 200, 300],
         "max_depth": [1, 2, 3],
         "learning_rate": [0.02, 0.05, 0.1],
         "subsample": [0.5, 0.8],
         "colsample_bytree": [0.5, 0.8]},
        {"n_estimators": 300, "max_depth": 2, "learning_rate": 0.02,
         "subsample": 0.8, "colsample_bytree": 0.8},
    ),
}


def bootstrap_auc_ci(y_true, y_score, n_bootstrap=1000, alpha=0.05, random_state=None):
    """Estimate AUC, bootstrap confidence interval, and bootstrap standard deviation."""
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    auc_values = []
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_mean = auc(fpr, tpr)
    
    for i in range(n_bootstrap):
        indices = rng.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]
        
        if len(np.unique(y_true_boot)) == 2:
            fpr_boot, tpr_boot, _ = roc_curve(y_true_boot, y_score_boot)
            auc_boot = auc(fpr_boot, tpr_boot)
            auc_values.append(auc_boot)
    
    if auc_values:
        auc_values = np.array(auc_values)
        lower = np.percentile(auc_values, 100 * alpha / 2)
        upper = np.percentile(auc_values, 100 * (1 - alpha / 2))
        auc_std = np.std(auc_values)
    else:
        lower, upper, auc_std = None, None, None
    
    return auc_mean, (lower, upper), auc_std


def bootstrap_pr_auc_ci(y_true, y_score, n_bootstrap=1000, alpha=0.05, random_state=None):
    """Estimate PR-AUC, bootstrap confidence interval, and bootstrap standard deviation."""
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    pr_auc_values = []
    
    pr_auc_mean = average_precision_score(y_true, y_score)
    
    for i in range(n_bootstrap):
        indices = rng.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]
        
        if len(np.unique(y_true_boot)) == 2:
            pr_auc_boot = average_precision_score(y_true_boot, y_score_boot)
            pr_auc_values.append(pr_auc_boot)
    
    if pr_auc_values:
        pr_auc_values = np.array(pr_auc_values)
        lower = np.percentile(pr_auc_values, 100 * alpha / 2)
        upper = np.percentile(pr_auc_values, 100 * (1 - alpha / 2))
        pr_auc_std = np.std(pr_auc_values)
    else:
        lower, upper, pr_auc_std = None, None, None
    
    return pr_auc_mean, (lower, upper), pr_auc_std


def compute_youden_index(y_true, y_score):
    """Find the Youden-optimal threshold and compute classification metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    youden_indices = tpr - fpr
    best_idx = np.argmax(youden_indices)
    best_threshold = thresholds[best_idx]
    best_youden = youden_indices[best_idx]
    
    y_pred = (y_score >= best_threshold).astype(int)
    
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    except:
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    npv = tn / (tn + fn) if (tn + fn) > 0 else float('nan')
    
    return {
        'threshold': float(best_threshold),
        'youden_index': float(best_youden),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'ppv': float(ppv),
        'npv': float(npv),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def compute_metrics_at_threshold(y_true, y_score, threshold):
    """
    Compute classification metrics at a pre-specified threshold.
    This is used for test/external cohorts to avoid selecting thresholds on validation data.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("A finite training-derived threshold is required.")

    y_pred = (y_score >= threshold).astype(int)

    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    except Exception:
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    npv = tn / (tn + fn) if (tn + fn) > 0 else float('nan')
    youden_index = sensitivity + specificity - 1 if np.isfinite(sensitivity) and np.isfinite(specificity) else float('nan')

    return {
        'threshold': threshold,
        'youden_index': float(youden_index),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'ppv': float(ppv),
        'npv': float(npv),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


# ============================================================
# 7) SHAP for one model/dataset
# ============================================================

def _is_xgboost_estimator(clf) -> bool:
    cls_name = clf.__class__.__name__.lower()
    return "xgb" in cls_name or clf.__class__.__module__.lower().startswith("xgboost")


def _xgboost_pred_contribs_shap_values(clf, X_trans):
    """
    XGBoost 2.x returns UBJSON from Booster.save_raw() by default, while
    SHAP 0.42.1 still expects the older JSON text payload in TreeExplainer.
    Use XGBoost's native TreeSHAP path as a compatibility fallback.
    """
    if not XGBOOST_AVAILABLE or "xgb" not in globals():
        raise RuntimeError("xgboost is not available")
    if not hasattr(clf, "get_booster"):
        raise TypeError("estimator does not expose get_booster()")

    booster = clf.get_booster()
    X_np = np.asarray(X_trans)
    contribs = booster.predict(xgb.DMatrix(X_np), pred_contribs=True)
    contribs = np.asarray(contribs)
    n_features = X_np.shape[1]

    if contribs.ndim == 2:
        if contribs.shape[1] < n_features:
            raise ValueError(f"unexpected XGBoost SHAP shape: {contribs.shape}")
        return contribs[:, :n_features]

    if contribs.ndim == 3:
        if contribs.shape[2] >= n_features:
            class_idx = 1 if contribs.shape[1] > 1 else 0
            return contribs[:, class_idx, :n_features]
        if contribs.shape[1] >= n_features:
            class_idx = 1 if contribs.shape[2] > 1 else 0
            return contribs[:, :n_features, class_idx]

    raise ValueError(f"unexpected XGBoost SHAP shape: {contribs.shape}")


def plot_shap_for_estimator(model_name: str, estimator, X_df: pd.DataFrame, dataset_tag: str, task_name: str):
    """Create SHAP bar and beeswarm plots for one fitted model and dataset."""
    shap = safe_import_shap()
    if shap is None:
        return

    if X_df is None or len(X_df) == 0:
        print(f"[SHAP] Skip empty dataset: task={task_name} model={model_name} dataset={dataset_tag}")
        return

    try:
        if isinstance(estimator, Pipeline):
            preproc = estimator[:-1]
            clf = estimator.named_steps["clf"]
            X_trans = preproc.transform(X_df)
            feat_names = get_feature_names_after_preproc(preproc, X_df.columns.tolist())
        else:
            preproc = None
            clf = estimator
            X_trans = X_df.values
            feat_names = X_df.columns.tolist()

        model_cls = clf.__class__.__name__.lower()

        if any(k in model_cls for k in ["randomforest", "xgb"]):
            try:
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_trans)
            except UnicodeDecodeError as e:
                if _is_xgboost_estimator(clf):
                    print(f"[WARN] SHAP TreeExplainer UnicodeDecodeError for {model_name}; using XGBoost pred_contribs fallback.")
                    shap_values = _xgboost_pred_contribs_shap_values(clf, X_trans)
                else:
                    print(f"[WARN] SHAP TreeExplainer UnicodeDecodeError for {model_name}, skip.")
                    return
            except Exception as e:
                if _is_xgboost_estimator(clf):
                    try:
                        print(f"[WARN] SHAP TreeExplainer failed for {model_name}; using XGBoost pred_contribs fallback. err={clean_exception_message(repr(e))}")
                        shap_values = _xgboost_pred_contribs_shap_values(clf, X_trans)
                    except Exception as e2:
                        print(f"[WARN] XGBoost pred_contribs fallback failed for {model_name}, skip. err={clean_exception_message(repr(e2))}")
                        return
                else:
                    print(f"[WARN] SHAP TreeExplainer failed for {model_name}, skip. err={clean_exception_message(repr(e))}")
                    return

            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values_pos = shap_values[1]
            else:
                shap_values_pos = shap_values

        elif "logisticregression" in model_cls:
            explainer = shap.LinearExplainer(clf, X_trans)
            shap_values_pos = explainer.shap_values(X_trans)

        else:
            bg_n = min(30, X_trans.shape[0])
            bg = shap.sample(pd.DataFrame(X_trans), bg_n, random_state=RANDOM_STATE).values
            if hasattr(clf, "predict_proba"):
                f = lambda X_: clf.predict_proba(X_)[:, 1]
            else:
                f = lambda X_: clf.decision_function(X_)
            explainer = shap.KernelExplainer(f, bg)
            shap_values_pos = explainer.shap_values(X_trans, nsamples=200)

        plt.figure(figsize=(30, 8))
        shap.summary_plot(shap_values_pos, X_trans, feature_names=feat_names, plot_type="bar", show=False)
        plt.gcf().subplots_adjust(left=0.4, right=0.95, top=0.95, bottom=0.07)
        ax = plt.gca()
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(1.0)
            ax.spines[spine].set_edgecolor('black')
        plt.title(f"SHAP bar | task={task_name} | model={model_name} | {dataset_tag}")
        plt.show()

        plt.figure(figsize=(30, 8))
        shap.summary_plot(shap_values_pos, X_trans, feature_names=feat_names, show=False)
        plt.gcf().subplots_adjust(left=0.4, right=0.95, top=0.95, bottom=0.07)
        ax = plt.gca()
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(1.0)
            ax.spines[spine].set_edgecolor('black')
        plt.title(f"SHAP beeswarm | task={task_name} | model={model_name} | {dataset_tag}")
        plt.show()

        print(f"[SHAP] Done: task={task_name} model={model_name} dataset={dataset_tag}")

    except Exception as e:
        print(f"[WARN] SHAP failed: task={task_name} model={model_name} dataset={dataset_tag} err={repr(e)}")


def plot_auc_forest(auc_df, task_name):
    """Create a forest plot from an AUC summary table."""
    if auc_df is None or len(auc_df) == 0:
        return
    
    if "auc_test" in auc_df.columns:
        auc_col = "auc_test"
    elif "auc_oof_train" in auc_df.columns:
        auc_col = "auc_oof_train"
    else:
        auc_col = [col for col in auc_df.columns if "auc" in col.lower()][0]
    
    auc_df = auc_df.sort_values(auc_col, ascending=True)
    
    models = auc_df["model"].tolist()
    auc_values = auc_df[auc_col].tolist()
    ci_lower = auc_df["ci_lower"].tolist()
    ci_upper = auc_df["ci_upper"].tolist()
    
    ci_length = [auc - lower for auc, lower in zip(auc_values, ci_lower)]
    
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(models))
    
    plt.errorbar(auc_values, y_pos, xerr=ci_length, fmt='o', capsize=5, color='blue', alpha=0.7)
    
    plt.scatter(auc_values, y_pos, color='blue', s=50, zorder=3)
    
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.yticks(y_pos, models)
    plt.xlabel('AUC with 95% CI')
    plt.ylabel('Models')
    plt.title(f'Test AUC Forest Plot - {task_name}')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    plt.show()


# ============================================================
# 8) Prediction Probability Distribution Histogram
# ============================================================

def plot_prediction_histogram(model_name: str, task_name: str,
                              y_true_train, y_score_train,
                              y_true_test=None, y_score_test=None,
                              y_true_ext=None, y_score_ext=None):
    """Plot prediction-score distributions by class for available cohorts."""
    datasets = []
    
    if y_score_train is not None and len(y_score_train) > 0:
        datasets.append(("Train (OOF)", y_true_train, y_score_train))
    
    if y_score_test is not None and len(y_score_test) > 0 and y_true_test is not None:
        datasets.append(("Test", y_true_test, y_score_test))
    
    if y_score_ext is not None and len(y_score_ext) > 0 and y_true_ext is not None:
        datasets.append(("External", y_true_ext, y_score_ext))
    
    if len(datasets) == 0:
        return
    
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5), squeeze=False)
    axes = axes.flatten()
    
    for idx, (ds_name, y_true, y_score) in enumerate(datasets):
        ax = axes[idx]
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        
        score_min, score_max = np.min(y_score), np.max(y_score)
        if score_min < -0.1 or score_max > 1.1:
            print(f"[HIST] {model_name} {ds_name}: scores out of [0,1] range [{score_min:.2f}, {score_max:.2f}], "
                  f"using raw values")
            bins = 30
            range_min, range_max = None, None
            xlabel = "Raw Score"
        else:
            bins = np.linspace(0, 1, 31)
            range_min, range_max = 0, 1
            xlabel = "Predicted Probability"
        
        y_pos = y_score[y_true == 1]
        y_neg = y_score[y_true == 0]
        
        if len(y_pos) > 0 and len(y_neg) > 0:
            ax.hist(y_neg, bins=bins, range=(range_min, range_max), alpha=0.6, 
                    label=f'Negative (n={len(y_neg)})', color='#4472C4', density=False)
            ax.hist(y_pos, bins=bins, range=(range_min, range_max), alpha=0.6, 
                    label=f'Positive (n={len(y_pos)})', color='#ED7D31', density=False)
            ax.legend(fontsize=8, loc='upper right')
        elif len(y_pos) > 0:
            ax.hist(y_pos, bins=bins, range=(range_min, range_max), alpha=0.7, 
                    label=f'Positive (n={len(y_pos)})', color='#ED7D31', density=False)
            ax.legend(fontsize=8, loc='upper right')
        elif len(y_neg) > 0:
            ax.hist(y_neg, bins=bins, range=(range_min, range_max), alpha=0.7, 
                    label=f'Negative (n={len(y_neg)})', color='#4472C4', density=False)
            ax.legend(fontsize=8, loc='upper right')
        
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        
        stats_text = (f'Pos: mean={np.mean(y_pos):.3f}±{np.std(y_pos):.3f}\n'
                      f'Neg: mean={np.mean(y_neg):.3f}±{np.std(y_neg):.3f}'
                      if len(y_pos) > 0 and len(y_neg) > 0
                      else f'Mean={np.mean(y_score):.3f}±{np.std(y_score):.3f}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title(f'{ds_name} | N={len(y_score)}', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Prediction Distribution | {model_name} | task={task_name}', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================
# 9) Decision Curve Analysis
# ============================================================

def plot_combined_dca(dataset_tag: str, task_name: str, 
                       model_preds: list):
    """Plot combined decision curves for multiple models."""
    try:
        if not model_preds:
            print(f"[DCA] No models for {dataset_tag}, skip.")
            return
        
        all_thresholds = np.linspace(0, 1.0, 201)
        model_curves = []
        
        for model_name, y_true, y_score in model_preds:
            y_true = np.asarray(y_true, dtype=int)
            y_score = np.asarray(y_score, dtype=float)
            n_total = len(y_true)
            if n_total == 0:
                continue
            
            score_min, score_max = np.min(y_score), np.max(y_score)
            if score_min < -0.1 or score_max > 1.1:
                print(f"[DCA] Skip {model_name}: scores out of [0,1] range [{score_min:.2f}, {score_max:.2f}], "
                      f"likely not probabilities (use predict_proba instead of decision_function).")
                continue
            
            n_pos = np.sum(y_true == 1)
            n_neg = np.sum(y_true == 0)
            prevalence = n_pos / n_total if n_total > 0 else 0
            
            net_benefit_model = []
            net_benefit_all = []
            
            for pt in all_thresholds:
                if pt >= 1.0:
                    nb_model = np.nan
                    nb_all = np.nan
                else:
                    ratio = pt / (1 - pt) if pt < 1.0 else float('inf')
                    
                    tp = np.sum((y_score >= pt) & (y_true == 1))
                    fp = np.sum((y_score >= pt) & (y_true == 0))
                    nb_model = tp / n_total - fp / n_total * ratio
                    
                    nb_all = n_pos / n_total - n_neg / n_total * ratio
                    nb_all = max(0, nb_all)
                
                net_benefit_model.append(nb_model)
                net_benefit_all.append(nb_all)
            
            model_curves.append((model_name, net_benefit_model, net_benefit_all))
        
        if not model_curves:
            print(f"[DCA] No valid models for {dataset_tag}, skip.")
            return
        
        all_nb = []
        for _, nb_model, nb_all in model_curves:
            valid_nb = [x for x in nb_model if not np.isnan(x)]
            valid_all = [x for x in nb_all if not np.isnan(x)]
            all_nb.extend(valid_nb)
            all_nb.extend(valid_all)
        
        max_nb = max(all_nb) if all_nb else 0.1
        ylim_max = max(0.05, max_nb * 1.2)
        ylim_min = -0.05
        
        plt.figure(figsize=(9, 7))
        
        _, _, nb_all_ref = model_curves[0]
        plt.plot(all_thresholds, nb_all_ref, 'k--', linewidth=1.5, label='Treat All', alpha=0.7)
        
        plt.axhline(y=0, color='gray', linestyle=':', linewidth=1, label='Treat None')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_curves)))
        for i, (model_name, nb_model, _) in enumerate(model_curves):
            plt.plot(all_thresholds, nb_model, color=colors[i], linewidth=2, 
                    label=model_name, alpha=0.85)
        
        plt.xlim(0, 1.0)
        plt.ylim(ylim_min, ylim_max)
        plt.xlabel('Threshold Probability', fontsize=12)
        plt.ylabel('Net Benefit', fontsize=12)
        plt.title(f'Decision Curve Analysis | task={task_name} | {dataset_tag}', 
                  fontsize=12)
        plt.legend(fontsize=9, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"[DCA] Done: task={task_name} dataset={dataset_tag} models={len(model_curves)}")
        
    except Exception as e:
        print(f"[WARN] Combined DCA failed: task={task_name} dataset={dataset_tag} err={repr(e)}")


# ============================================================
# 10) Calibration Curve
# ============================================================

def plot_combined_calibration(dataset_tag: str, task_name: str,
                               model_preds: list, n_bins: int = 5):
    """Plot combined calibration curves for multiple models."""
    try:
        if not model_preds:
            print(f"[CAL] No models for {dataset_tag}, skip.")
            return
        
        plt.figure(figsize=(8, 7))
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_preds)))
        valid_count = 0
        
        for i, (model_name, y_true, y_score) in enumerate(model_preds):
            try:
                y_true = np.asarray(y_true, dtype=int)
                y_score = np.asarray(y_score, dtype=float)
                
                if len(y_true) == 0:
                    continue
                
                score_min, score_max = np.min(y_score), np.max(y_score)
                if score_min < -0.1 or score_max > 1.1:
                    print(f"[CAL] Skip {model_name}: scores out of [0,1] range [{score_min:.2f}, {score_max:.2f}]")
                    continue
                
                prob_true, prob_pred = calibration_curve(y_true, y_score, 
                                                        n_bins=n_bins, strategy='quantile')
                
                brier_score = np.mean((y_score - y_true) ** 2)
                
                if len(prob_true) > 0 and len(prob_pred) > 0:
                    plt.plot(prob_pred, prob_true, 'o-', color=colors[i], 
                            linewidth=2, markersize=5, 
                            label=f'{model_name} (Brier={brier_score:.3f})', alpha=0.85)
                    valid_count += 1
                    
            except Exception as e:
                print(f"[CAL] Failed for {model_name}: {repr(e)}")
                continue
        
        if valid_count > 0:
            plt.xlabel('Mean Predicted Probability', fontsize=12)
            plt.ylabel('Fraction of Positives', fontsize=12)
            plt.title(f'Calibration Curve | task={task_name} | {dataset_tag}', fontsize=12)
            plt.legend(fontsize=8, loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.show()
            print(f"[CAL] Done: task={task_name} dataset={dataset_tag} models={valid_count}")
        else:
            plt.close()
            print(f"[CAL] No valid calibration data for {dataset_tag}")
            
    except Exception as e:
        print(f"[WARN] Combined calibration failed: task={task_name} dataset={dataset_tag} err={repr(e)}")


def run_one_task_all_models(task_name: str, y_dict: dict, X_all: pd.DataFrame, X_ext: pd.DataFrame, out_df: pd.DataFrame, cfg: dict, out_mgr: OutputManager):
    if task_name not in y_dict:
        print(f"[SKIP] 未找到任务 {task_name}")
        return None

    y0 = y_dict[task_name].copy().dropna().astype(int)
    if len(y0) < 10:
        print(f"[SKIP] task={task_name} too few labeled samples: n={len(y0)}")
        return None

    X0 = X_all.loc[y0.index].copy()
    X0 = X0.dropna(axis=1, how="all")

    print(f"\n==================== Task: {task_name} ====================")
    describe_label_and_features(y0, X0, title_prefix=f"[{task_name}] ")

    train_ids, test_ids, split_enabled = split_train_test_by_closed_intervals(y0.index, cfg, y=y0)
    if cfg.get("Divide") is not None:
        print(f"[SPLIT] Enabled. Internal range=[{cfg.get('Internal_MIN','all')},{cfg.get('Internal_MAX','all')}], "
              f"Divide={cfg['Divide']}, Random={cfg['Divide_Random']}")
        print(f"[SPLIT] Train n={len(train_ids)}, Test n={len(test_ids)}")
    else:
        print("[SPLIT] Divide not specified. Using all samples as training set (no test set).")

    y_tr = y0.loc[train_ids].values
    X_tr = X0.loc[train_ids]
    y_te = y0.loc[test_ids].values if split_enabled else None
    X_te = X0.loc[test_ids] if split_enabled else None
    outcomes_by_id = out_df.set_index("ID", drop=False)
    train_outcomes = outcomes_by_id.loc[train_ids].copy()
    test_outcomes = outcomes_by_id.loc[test_ids].copy() if split_enabled else None

    if X_tr.shape[1] == 0:
        print(f"[SKIP] task={task_name}: X has 0 features (check modality include / paths / sheets).")
        return None

    task_s = _sanitize_filename(task_name)
    X_tr_full = X_tr.copy()
    X_te_full = X_te.copy() if X_te is not None else None

    sel_cols = None
    fs_meta = None
    X_tr_selected = None
    X_te_selected = None
    X_ext_selected = None
    X_ext_avail = None

    def ensure_final_feature_selection():
        nonlocal sel_cols, fs_meta, X_tr_selected, X_te_selected, X_ext_selected
        if sel_cols is not None:
            return

        print("[FS] Running final feature selection on all training samples after nested CV.")
        sel_cols, fs_meta = run_feature_selection(X_tr_full, y_tr, cfg)
        if len(sel_cols) == 0:
            raise ValueError(f"[FS] selected 0 features for task={task_name}.")

        X_tr_selected = X_tr_full[sel_cols]
        X_te_selected = X_te_full[sel_cols] if X_te_full is not None else None

        print(
            f"[FS] mode={fs_meta.get('mode')} | "
            f"selected={fs_meta.get('selected_n')} | detail={fs_meta.get('detail')}"
        )

        out_mgr.set_tag(f"{task_s}__split_features", reset_counter=True)
        X_tr_with_id = X_tr_selected.reset_index()
        X_tr_with_id.to_csv(out_mgr.current_dir / f"{task_s}_train_features.csv", index=False)
        print(f"[SPLIT] Saved training features to {out_mgr.current_dir / f'{task_s}_train_features.csv'}")

        if split_enabled and X_te_selected is not None:
            X_te_with_id = X_te_selected.reset_index()
            X_te_with_id.to_csv(out_mgr.current_dir / f"{task_s}_test_features.csv", index=False)
            print(f"[SPLIT] Saved test features to {out_mgr.current_dir / f'{task_s}_test_features.csv'}")

        if external_enabled and X_ext_avail is not None and len(X_ext_avail) > 0:
            X_ext_selected = X_ext_avail[sel_cols].copy()
            X_ext_with_id = X_ext_selected.reset_index()
            X_ext_with_id.to_csv(out_mgr.current_dir / f"{task_s}_external_features.csv", index=False)
            print(f"[EXTERNAL] Saved external validation features to {out_mgr.current_dir / f'{task_s}_external_features.csv'}")

        out_mgr.set_tag(f"{task_s}__fs_selected_features", reset_counter=True)
        display(pd.DataFrame({
            "feature_rank": np.arange(1, len(sel_cols) + 1),
            "feature": sel_cols,
        }))
        if fs_meta.get("mode") == "per_modality" and fs_meta.get("detail", {}).get("modality_steps"):
            display(pd.DataFrame([
                {
                    "modality": s["modality"],
                    "method": s["method"],
                    "input_n": s["input_n"],
                    "selected_n": s["selected_n"],
                }
                for s in fs_meta["detail"]["modality_steps"]
            ]))
        out_mgr.set_tag(f"{task_s}__main", reset_counter=False)

    use_smote = bool(cfg.get("SMOTE", False))
    if use_smote:
        print("[SMOTE] Enabled (applied after feature selection).")
    else:
        print("[SMOTE] Disabled.")

    models = build_models()

    selected_algorithms = cfg.get("ALGORITHMS")
    if selected_algorithms is not None and len(selected_algorithms) > 0:
        selected_algorithms = [str(a).strip() for a in selected_algorithms]
        models = {name: est for name, est in models.items() if name in selected_algorithms}
        print(f"[ALGORITHMS] Selected algorithms: {list(models.keys())}")
    else:
        print(f"[ALGORITHMS] Using all available algorithms: {list(models.keys())}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    plt.figure(figsize=(7, 6))
    train_auc_rows = []
    test_auc_rows = []
    test_roc_curves = []
    external_auc_rows = []
    external_roc_curves = []
    
    train_pr_auc_rows = []
    train_pr_curves = []
    test_pr_auc_rows = []
    test_pr_curves = []
    external_pr_auc_rows = []
    external_pr_curves = []
    
    train_youden_rows = []
    test_youden_rows = []
    external_youden_rows = []
    
    oof_scores_dict = {}
    test_scores_dict = {}
    ext_scores_dict = {}
    tuned_models = {}
    
    external_enabled = False
    y_ext = None
    X_ext_selected = None
    if X_ext is not None and len(X_ext) > 0:
        external_min = cfg.get("EXTERNAL_ID_MIN")
        external_max = cfg.get("EXTERNAL_ID_MAX")
        if external_min is not None and external_max is not None:
            print(f"[EXTERNAL] Preparing external validation set...")
            ext_df_ids = y_dict.get(task_name, pd.Series(dtype=int))
            if ext_df_ids is not None and len(ext_df_ids) > 0:
                y_ext_all = y0.copy()
                ext_mask = (y_ext_all.index >= external_min) & (y_ext_all.index <= external_max)
                y_ext_ids = y_ext_all[ext_mask].index.tolist()
                if len(y_ext_ids) > 0:
                    y_ext = y0.loc[y_ext_ids].dropna().values
                    X_ext_avail = X_ext.loc[X_ext.index.isin(y_ext_ids)]
                    external_outcomes = outcomes_by_id.loc[y_ext_ids].copy()
                    print(f"[EXTERNAL] External validation: {len(y_ext)} samples with labels")
                    print(f"[EXTERNAL DEBUG] External y shape: {y_ext.shape if y_ext is not None else 'None'}")
                    external_enabled = True
                else:
                    print("[WARN] External validation: No samples with valid labels in external range")
            else:
                print("[WARN] External validation: No y_dict for task")

    for name, est in models.items():
        try:
            if name in TUNE_CONFIGS:
                if name == "XGBoost" and not XGBOOST_AVAILABLE:
                    print(f"[TUNE] {name} not available (xgboost not installed), skip tuning.")
                    scores_tr = oof_scores_with_optional_smote(
                        est, X_tr_full, y_tr, cv=cv,
                        use_smote=use_smote, random_state=RANDOM_STATE,
                        fs_cfg=cfg
                    )
                    ensure_final_feature_selection()
                    best_params = {}
                else:
                    builder, grid, fallback = TUNE_CONFIGS[name]
                    train_random_state = RANDOM_STATE + 1936
                    print(f"[NESTED-CV] {name} Starting nested cross-validation (random_state={train_random_state})...")
                    scores_tr, best_params, best_inner_auc = nested_cv_tune_and_score(
                        name, builder, grid, fallback,
                        X_tr_full, y_tr, outer_cv=cv,
                        inner_cv_folds=3,
                        use_smote=use_smote, random_state=train_random_state,
                        fs_cfg=cfg
                    )
                    print(f"[NESTED-CV] {name} Nested CV done.")
                    ensure_final_feature_selection()
                    print(f"[TUNE] {name} Final hyperparameter search on all-training selected features...")
                    best_params, best_final_auc, _ = tune_model_params(
                        name, builder, grid, fallback,
                        X_tr_selected, y_tr, cv=cv,
                        use_smote=use_smote, random_state=train_random_state
                    )
                    print(f"[TUNE] {name} Final best params: {best_params}, CV AUC={best_final_auc:.4f}")
                    est = builder(**best_params, random_state=train_random_state)
            else:
                scores_tr = oof_scores_with_optional_smote(
                    est, X_tr_full, y_tr, cv=cv,
                    use_smote=use_smote, random_state=RANDOM_STATE,
                    fs_cfg=cfg
                )
                ensure_final_feature_selection()
                best_params = {}
            
            if X_tr_selected is None:
                ensure_final_feature_selection()
            X_tr = X_tr_selected
            X_te = X_te_selected

            tuned_models[name] = est
            
            oof_scores_dict[name] = scores_tr.copy()
            
            fpr, tpr, _ = roc_curve(y_tr, scores_tr)
            auc_tr = auc(fpr, tpr)
            
            _, auc_ci, auc_std = bootstrap_auc_ci(y_tr, scores_tr, random_state=RANDOM_STATE)
            
            train_auc_rows.append((name, float(auc_tr), auc_ci[0], auc_ci[1], auc_std))
            plt.plot(fpr, tpr, label=f"{name} (OOF AUC={auc_tr:.3f})")
            
            try:
                pr_auc_tr, pr_auc_ci_tr, pr_auc_std_tr = bootstrap_pr_auc_ci(y_tr, scores_tr, random_state=RANDOM_STATE)
                train_pr_auc_rows.append((name, float(pr_auc_tr), pr_auc_ci_tr[0] if pr_auc_ci_tr else None, 
                                          pr_auc_ci_tr[1] if pr_auc_ci_tr else None, pr_auc_std_tr))
                prec_tr, rec_tr, _ = precision_recall_curve(y_tr, scores_tr)
                train_pr_curves.append((name, prec_tr, rec_tr, float(pr_auc_tr)))
            except Exception:
                train_pr_auc_rows.append((name, float('nan'), None, None, None))
            
            train_threshold = float('nan')
            try:
                yi_tr = compute_youden_index(y_tr, scores_tr)
                train_threshold = yi_tr['threshold']
                train_youden_rows.append((name, yi_tr['threshold'], yi_tr['youden_index'], 
                                          yi_tr['sensitivity'], yi_tr['specificity'],
                                          yi_tr['ppv'], yi_tr['npv'],
                                          yi_tr['tp'], yi_tr['tn'], yi_tr['fp'], yi_tr['fn']))
            except Exception:
                train_youden_rows.append((name, float('nan'), float('nan'), float('nan'), 
                                          float('nan'), float('nan'), float('nan'), 0, 0, 0, 0))

            s_te = np.array([])
            est2 = None
            X_tr_imp = None
            
            if split_enabled and X_te is not None and len(X_te) > 0:
                est2 = clone(est)

                if use_smote:
                    X_tr_imp = _impute_df(X_tr, X_tr)
                    X_te_imp = _impute_df(X_tr, X_te)
                    X_tr_sm, y_tr_sm = _maybe_smote_xy(X_tr_imp.values, y_tr, random_state=RANDOM_STATE)
                    X_tr_sm_df = pd.DataFrame(X_tr_sm, columns=X_tr.columns)
                    est2.fit(X_tr_sm_df, y_tr_sm)
                    X_eval = X_te_imp
                else:
                    X_tr_imp = _impute_df(X_tr, X_tr)
                    est2.fit(X_tr, y_tr)
                    X_eval = X_te

                if hasattr(est2, "predict_proba"):
                    s_te = est2.predict_proba(X_eval)[:, 1]
                elif hasattr(est2, "decision_function"):
                    s_te = est2.decision_function(X_eval)
                else:
                    s_te = est2.predict(X_eval).astype(float)

                test_scores_dict[name] = s_te.copy()

                fpr2, tpr2, _ = roc_curve(y_te, s_te)
                auc_te = auc(fpr2, tpr2)
                
                _, auc_ci, auc_std = bootstrap_auc_ci(y_te, s_te, random_state=RANDOM_STATE)

                test_auc_rows.append((name, float(auc_te), auc_ci[0], auc_ci[1], auc_std))
                test_roc_curves.append((name, fpr2, tpr2, float(auc_te)))
                
                try:
                    pr_auc_te, pr_auc_ci_te, pr_auc_std_te = bootstrap_pr_auc_ci(y_te, s_te, random_state=RANDOM_STATE)
                    test_pr_auc_rows.append((name, float(pr_auc_te), pr_auc_ci_te[0] if pr_auc_ci_te else None,
                                              pr_auc_ci_te[1] if pr_auc_ci_te else None, pr_auc_std_te))
                    prec_te, rec_te, _ = precision_recall_curve(y_te, s_te)
                    test_pr_curves.append((name, prec_te, rec_te, float(pr_auc_te)))
                except Exception:
                    test_pr_auc_rows.append((name, float('nan'), None, None, None))
                
                try:
                    yi_te = compute_metrics_at_threshold(y_te, s_te, train_threshold)
                    test_youden_rows.append((name, yi_te['threshold'], yi_te['youden_index'],
                                              yi_te['sensitivity'], yi_te['specificity'],
                                              yi_te['ppv'], yi_te['npv'],
                                              yi_te['tp'], yi_te['tn'], yi_te['fp'], yi_te['fn']))
                except Exception:
                    test_youden_rows.append((name, float('nan'), float('nan'), float('nan'),
                                              float('nan'), float('nan'), float('nan'), 0, 0, 0, 0))
            
            s_ext = np.array([])
            if external_enabled and X_ext_selected is not None and len(X_ext_selected) > 0 and y_ext is not None:
                print(f"[EXTERNAL DEBUG] Model {name}: y_ext length = {len(y_ext)}, X_ext_selected length = {len(X_ext_selected)}")
                try:
                    est_ext = clone(est)
                    
                    if use_smote:
                        if X_tr_imp is None:
                            print(f"[EXTERNAL DEBUG] Initializing X_tr_imp for model {name}")
                            X_tr_imp = _impute_df(X_tr, X_tr)
                        X_tr_sm_ext, y_tr_sm_ext = _maybe_smote_xy(X_tr_imp.values, y_tr, random_state=RANDOM_STATE)
                        X_tr_sm_ext_df = pd.DataFrame(X_tr_sm_ext, columns=X_tr.columns)
                        est_ext.fit(X_tr_sm_ext_df, y_tr_sm_ext)
                    else:
                        est_ext.fit(X_tr, y_tr)
                    
                    if hasattr(est_ext, "predict_proba"):
                        s_ext = est_ext.predict_proba(X_ext_selected)[:, 1]
                    elif hasattr(est_ext, "decision_function"):
                        s_ext = est_ext.decision_function(X_ext_selected)
                    else:
                        s_ext = est_ext.predict(X_ext_selected).astype(float)
                    
                    ext_scores_dict[name] = s_ext.copy()
                    
                    print(f"[EXTERNAL DEBUG] Model {name}: s_ext length = {len(s_ext)}")
                    
                    fpr_ext, tpr_ext, _ = roc_curve(y_ext, s_ext)
                    auc_ext = auc(fpr_ext, tpr_ext)
                    
                    _, auc_ci_ext, auc_std_ext = bootstrap_auc_ci(y_ext, s_ext, random_state=RANDOM_STATE)
                    
                    external_auc_rows.append((name, float(auc_ext), auc_ci_ext[0], auc_ci_ext[1], auc_std_ext))
                    external_roc_curves.append((name, fpr_ext, tpr_ext, float(auc_ext)))
                    print(f"[EXTERNAL] Model {name}: AUC={auc_ext:.3f} (95% CI: {auc_ci_ext[0]:.3f}-{auc_ci_ext[1]:.3f})")
                    
                    try:
                        pr_auc_ext, pr_auc_ci_ext, pr_auc_std_ext = bootstrap_pr_auc_ci(y_ext, s_ext, random_state=RANDOM_STATE)
                        external_pr_auc_rows.append((name, float(pr_auc_ext), pr_auc_ci_ext[0] if pr_auc_ci_ext else None,
                                                      pr_auc_ci_ext[1] if pr_auc_ci_ext else None, pr_auc_std_ext))
                        prec_ext, rec_ext, _ = precision_recall_curve(y_ext, s_ext)
                        external_pr_curves.append((name, prec_ext, rec_ext, float(pr_auc_ext)))
                    except Exception:
                        external_pr_auc_rows.append((name, float('nan'), None, None, None))
                    
                    try:
                        yi_ext = compute_metrics_at_threshold(y_ext, s_ext, train_threshold)
                        external_youden_rows.append((name, yi_ext['threshold'], yi_ext['youden_index'],
                                                      yi_ext['sensitivity'], yi_ext['specificity'],
                                                      yi_ext['ppv'], yi_ext['npv'],
                                                      yi_ext['tp'], yi_ext['tn'], yi_ext['fp'], yi_ext['fn']))
                    except Exception:
                        external_youden_rows.append((name, float('nan'), float('nan'), float('nan'),
                                                      float('nan'), float('nan'), float('nan'), 0, 0, 0, 0))
                except Exception as e:
                    print(f"[WARN] External validation failed for {name}: {clean_exception_message(repr(e))}")

            save_model_predictions(
                out_mgr.root_dir, task_name, name,
                scores_tr, s_te, y_tr, y_te,
                external_probs=s_ext if external_enabled else None,
                external_y=y_ext if external_enabled else None,
                train_outcomes=train_outcomes,
                test_outcomes=test_outcomes,
                external_outcomes=external_outcomes if external_enabled else None,
            )
            
            try:
                if use_smote:
                    if X_tr_imp is None:
                        X_tr_imp = _impute_df(X_tr, X_tr)
                    X_tr_sm_final, y_tr_sm_final = _maybe_smote_xy(X_tr_imp.values, y_tr, random_state=RANDOM_STATE)
                    X_tr_sm_final_df = pd.DataFrame(X_tr_sm_final, columns=X_tr.columns)
                    est_final = clone(est)
                    est_final.fit(X_tr_sm_final_df, y_tr_sm_final)
                else:
                    est_final = clone(est)
                    est_final.fit(X_tr, y_tr)
                
                model_path = save_model_to_file(est_final, name, task_name, out_mgr.root_dir, suffix="final")
                
                if best_params:
                    task_s = _sanitize_filename(task_name)
                    model_s = _sanitize_filename(name)
                    task_dir = out_mgr.root_dir / f"task_{task_s}" / "models"
                    task_dir.mkdir(parents=True, exist_ok=True)
                    params_path = task_dir / f"{model_s}_best_params.json"
                    with open(params_path, 'w') as f:
                        json.dump(best_params, f, indent=2, default=str)
                    print(f"[SAVE] Best params saved: {params_path}")
            except Exception as e:
                print(f"[WARN] Failed to save model {name}: {clean_exception_message(repr(e))}")

        except Exception as e:
            print(f"[WARN] Model failed: {name} err={clean_exception_message(repr(e))}")
            continue

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC (Train OOF) - {task_name} | N_train={len(y_tr)}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.show()

    train_auc_df = pd.DataFrame(train_auc_rows, columns=["model", "auc_oof_train", "ci_lower", "ci_upper", "auc_std"]).sort_values("auc_oof_train", ascending=False)
    train_auc_df["auc_oof_train_ci"] = train_auc_df.apply(lambda row: f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]", axis=1)
    train_auc_df = train_auc_df[["model", "auc_oof_train", "auc_std", "auc_oof_train_ci", "ci_lower", "ci_upper"]]
    display(train_auc_df)
    
    plot_auc_forest(train_auc_df, f"{task_name} (Train OOF)")

    if train_auc_df is None or len(train_auc_df) == 0:
        print(f"[WARN] No successful models for task={task_name}. Skip ROC/SHAP and continue.")
        return None

    if split_enabled:
        plt.figure(figsize=(7, 6))
        for name, fpr2, tpr2, auc_te in test_roc_curves:
            plt.plot(fpr2, tpr2, label=f"{name} (Test AUC={auc_te:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"ROC (Test) - {task_name} | N_test={len(y_te)}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.show()

    external_auc_df = None
    if external_enabled and len(external_auc_rows) > 0:
        plt.figure(figsize=(7, 6))
        for name, fpr_ext, tpr_ext, auc_ext in external_roc_curves:
            plt.plot(fpr_ext, tpr_ext, label=f"{name} (External AUC={auc_ext:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"ROC (External) - {task_name} | N_ext={len(y_ext) if y_ext is not None else 0}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.show()
        
        external_auc_df = (
            pd.DataFrame(external_auc_rows, columns=["model", "auc_external", "ci_lower", "ci_upper", "auc_std"])
            .sort_values("auc_external", ascending=False)
        )
        external_auc_df["auc_external_ci"] = external_auc_df.apply(lambda row: f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]", axis=1)
        external_auc_df = external_auc_df[["model", "auc_external", "auc_std", "auc_external_ci", "ci_lower", "ci_upper"]]
        display(external_auc_df)
        
        plot_auc_forest(external_auc_df, f"{task_name} (External Validation)")

    test_auc_df = None
    if split_enabled and len(test_auc_rows) > 0:
        test_auc_df = (
            pd.DataFrame(test_auc_rows, columns=["model", "auc_test", "ci_lower", "ci_upper", "auc_std"])
            .sort_values("auc_test", ascending=False)
        )
        test_auc_df["auc_test_ci"] = test_auc_df.apply(lambda row: f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]", axis=1)
        test_auc_df = test_auc_df[["model", "auc_test", "auc_std", "auc_test_ci", "ci_lower", "ci_upper"]]
        display(test_auc_df)
        
        plot_auc_forest(test_auc_df, f"{task_name} (Test)")
        
        if split_enabled and test_auc_df is not None and len(test_auc_df) > 0:
            combined_auc_df = pd.merge(
                train_auc_df[['model', 'auc_oof_train']],
                test_auc_df[['model', 'auc_test']],
                on='model',
                how='inner'
            )
            combined_auc_df['auc_diff'] = combined_auc_df['auc_oof_train'] - combined_auc_df['auc_test']
            combined_auc_df = combined_auc_df.sort_values('auc_diff', ascending=False)
            display(combined_auc_df)
    
    if external_enabled and external_auc_df is not None and len(external_auc_df) > 0:
        print(f"[EXTERNAL] Saving external validation results for task {task_name}")
        out_mgr.set_tag(f"{task_s}__external_results", reset_counter=True)
        try:
            external_auc_df.to_excel(out_mgr.current_dir / "external_validation_AUC.xlsx", index=False)
            print(f"[EXTERNAL] Saved external validation AUC to {out_mgr.current_dir / 'external_validation_AUC.xlsx'}")
        except Exception as e:
            print(f"[WARN] Failed to save external validation AUC: {repr(e)}")

    train_pr_auc_df = None
    if len(train_pr_auc_rows) > 0:
        train_pr_auc_df = pd.DataFrame(train_pr_auc_rows, 
                                       columns=["model", "pr_auc_oof_train", "ci_lower", "ci_upper", "pr_auc_std"])
        train_pr_auc_df = train_pr_auc_df.sort_values("pr_auc_oof_train", ascending=False)
        train_pr_auc_df["pr_auc_oof_train_ci"] = train_pr_auc_df.apply(
            lambda row: f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]" if pd.notna(row['ci_lower']) else "N/A", axis=1)
        train_pr_auc_df = train_pr_auc_df[["model", "pr_auc_oof_train", "pr_auc_std", "pr_auc_oof_train_ci", "ci_lower", "ci_upper"]]
        out_mgr.set_tag(f"{task_s}__pr_auc_train", reset_counter=True)
        display(train_pr_auc_df)
    
    if len(train_pr_curves) > 0:
        plt.figure(figsize=(7, 6))
        for name, prec, rec, pr_auc_val in train_pr_curves:
            plt.plot(rec, prec, label=f"{name} (PR-AUC={pr_auc_val:.3f})")
        baseline = np.sum(y_tr == 1) / len(y_tr)
        plt.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.3f})')
        plt.title(f"PR Curve (Train OOF) - {task_name}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        out_mgr.set_tag(f"{task_s}__pr_curve_train", reset_counter=True)
        plt.show()
    
    test_pr_auc_df = None
    if split_enabled and len(test_pr_auc_rows) > 0:
        test_pr_auc_df = pd.DataFrame(test_pr_auc_rows, 
                                      columns=["model", "pr_auc_test", "ci_lower", "ci_upper", "pr_auc_std"])
        test_pr_auc_df = test_pr_auc_df.sort_values("pr_auc_test", ascending=False)
        test_pr_auc_df["pr_auc_test_ci"] = test_pr_auc_df.apply(
            lambda row: f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]" if pd.notna(row['ci_lower']) else "N/A", axis=1)
        test_pr_auc_df = test_pr_auc_df[["model", "pr_auc_test", "pr_auc_std", "pr_auc_test_ci", "ci_lower", "ci_upper"]]
        out_mgr.set_tag(f"{task_s}__pr_auc_test", reset_counter=True)
        display(test_pr_auc_df)
        
        if len(test_pr_curves) > 0:
            plt.figure(figsize=(7, 6))
            for name, prec, rec, pr_auc_val in test_pr_curves:
                plt.plot(rec, prec, label=f"{name} (PR-AUC={pr_auc_val:.3f})")
            baseline_te = np.sum(y_te == 1) / len(y_te) if len(y_te) > 0 else 0
            plt.axhline(y=baseline_te, color='gray', linestyle='--', label=f'Baseline ({baseline_te:.3f})')
            plt.title(f"PR Curve (Test) - {task_name}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
            out_mgr.set_tag(f"{task_s}__pr_curve_test", reset_counter=True)
            plt.show()
    
    external_pr_auc_df = None
    if external_enabled and len(external_pr_auc_rows) > 0:
        external_pr_auc_df = pd.DataFrame(external_pr_auc_rows, 
                                          columns=["model", "pr_auc_external", "ci_lower", "ci_upper", "pr_auc_std"])
        external_pr_auc_df = external_pr_auc_df.sort_values("pr_auc_external", ascending=False)
        external_pr_auc_df["pr_auc_external_ci"] = external_pr_auc_df.apply(
            lambda row: f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]" if pd.notna(row['ci_lower']) else "N/A", axis=1)
        external_pr_auc_df = external_pr_auc_df[["model", "pr_auc_external", "pr_auc_std", "pr_auc_external_ci", "ci_lower", "ci_upper"]]
        out_mgr.set_tag(f"{task_s}__pr_auc_external", reset_counter=True)
        display(external_pr_auc_df)
        
        if len(external_pr_curves) > 0:
            plt.figure(figsize=(7, 6))
            for name, prec, rec, pr_auc_val in external_pr_curves:
                plt.plot(rec, prec, label=f"{name} (PR-AUC={pr_auc_val:.3f})")
            baseline_ext = np.sum(y_ext == 1) / len(y_ext) if y_ext is not None and len(y_ext) > 0 else 0
            plt.axhline(y=baseline_ext, color='gray', linestyle='--', label=f'Baseline ({baseline_ext:.3f})')
            plt.title(f"PR Curve (External) - {task_name}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
            out_mgr.set_tag(f"{task_s}__pr_curve_external", reset_counter=True)
            plt.show()

    if len(train_youden_rows) > 0:
        train_yi_df = pd.DataFrame(train_youden_rows, columns=[
            "model", "threshold", "youden_index", "sensitivity", "specificity", 
            "ppv", "npv", "tp", "tn", "fp", "fn"
        ])
        train_yi_df = train_yi_df.sort_values("youden_index", ascending=False)
        out_mgr.set_tag(f"{task_s}__youden_train", reset_counter=True)
        display(train_yi_df)
    
    if split_enabled and len(test_youden_rows) > 0:
        test_yi_df = pd.DataFrame(test_youden_rows, columns=[
            "model", "threshold", "youden_index", "sensitivity", "specificity", 
            "ppv", "npv", "tp", "tn", "fp", "fn"
        ])
        test_yi_df = test_yi_df.sort_values("youden_index", ascending=False)
        out_mgr.set_tag(f"{task_s}__youden_test", reset_counter=True)
        display(test_yi_df)
    
    if external_enabled and len(external_youden_rows) > 0:
        external_yi_df = pd.DataFrame(external_youden_rows, columns=[
            "model", "threshold", "youden_index", "sensitivity", "specificity", 
            "ppv", "npv", "tp", "tn", "fp", "fn"
        ])
        external_yi_df = external_yi_df.sort_values("youden_index", ascending=False)
        out_mgr.set_tag(f"{task_s}__youden_external", reset_counter=True)
        display(external_yi_df)

    if tuned_models is not None and len(tuned_models) > 0:
        dca_train_data = []
        dca_test_data = []
        dca_external_data = []
        
        cal_train_data = []
        cal_test_data = []
        cal_external_data = []
        
        for name, est in tuned_models.items():
            est_fit = None
            try:
                est_fit = est

                if use_smote:
                    X_tr_imp = _impute_df(X_tr, X_tr)
                    X_tr_sm, y_tr_sm = _maybe_smote_xy(X_tr_imp.values, y_tr, random_state=RANDOM_STATE)
                    X_tr_sm_df = pd.DataFrame(X_tr_sm, columns=X_tr.columns)
                    est_fit.fit(X_tr_sm_df, y_tr_sm)
                    X_train_for_shap = X_tr
                else:
                    est_fit.fit(X_tr, y_tr)
                    X_train_for_shap = X_tr

                if bool(cfg.get("RUN_SHAP", True)):
                    out_mgr.set_tag(f"{_sanitize_filename(task_name)}__{_sanitize_filename(name)}__shap_train", reset_counter=True)
                    plot_shap_for_estimator(name, est_fit, X_train_for_shap, dataset_tag="train", task_name=task_name)

                    if split_enabled and X_te is not None and len(X_te) > 0:
                        if use_smote:
                            X_test_for_shap = _impute_df(X_tr, X_te)
                        else:
                            X_test_for_shap = X_te

                        out_mgr.set_tag(f"{_sanitize_filename(task_name)}__{_sanitize_filename(name)}__shap_test", reset_counter=True)
                        plot_shap_for_estimator(name, est_fit, X_test_for_shap, dataset_tag="test", task_name=task_name)
                    
                    if external_enabled and X_ext_selected is not None and len(X_ext_selected) > 0:
                        try:
                            X_ext_for_shap = _impute_df(X_tr, X_ext_selected)
                            
                            out_mgr.set_tag(f"{_sanitize_filename(task_name)}__{_sanitize_filename(name)}__shap_external", reset_counter=True)
                            plot_shap_for_estimator(name, est_fit, X_ext_for_shap, dataset_tag="external", task_name=task_name)
                        except Exception as e:
                            print(f"[WARN] External SHAP failed for {name}: {repr(e)}")

                s_tr_oof = oof_scores_dict.get(name)
                
                s_te_stored = test_scores_dict.get(name) if split_enabled else None
                s_ext_stored = ext_scores_dict.get(name) if external_enabled else None

                if s_tr_oof is not None:
                    dca_train_data.append((name, y_tr, s_tr_oof))
                    cal_train_data.append((name, y_tr, s_tr_oof))

                if s_te_stored is not None and len(s_te_stored) > 0:
                    dca_test_data.append((name, y_te, s_te_stored))
                    cal_test_data.append((name, y_te, s_te_stored))

                if s_ext_stored is not None and len(s_ext_stored) > 0 and y_ext is not None and len(y_ext) > 0:
                    dca_external_data.append((name, y_ext, s_ext_stored))
                    cal_external_data.append((name, y_ext, s_ext_stored))

                out_mgr.set_tag(f"{_sanitize_filename(task_name)}__{_sanitize_filename(name)}__histogram", reset_counter=True)
                plot_prediction_histogram(
                    model_name=name,
                    task_name=task_name,
                    y_true_train=y_tr,
                    y_score_train=s_tr_oof,
                    y_true_test=y_te if split_enabled else None,
                    y_score_test=s_te_stored,
                    y_true_ext=y_ext if external_enabled else None,
                    y_score_ext=s_ext_stored
                )

            except Exception as e:
                print(f"[WARN] Model analysis failed: task={task_name} model={name} err={repr(e)}")

        out_mgr.set_tag(f"{task_s}__dca_train", reset_counter=True)
        plot_combined_dca(dataset_tag="train (OOF)", task_name=task_name, model_preds=dca_train_data)
        
        if len(dca_test_data) > 0:
            out_mgr.set_tag(f"{task_s}__dca_test", reset_counter=True)
            plot_combined_dca(dataset_tag="test", task_name=task_name, model_preds=dca_test_data)
        
        if len(dca_external_data) > 0:
            out_mgr.set_tag(f"{task_s}__dca_external", reset_counter=True)
            plot_combined_dca(dataset_tag="external", task_name=task_name, model_preds=dca_external_data)
        
        out_mgr.set_tag(f"{task_s}__cal_train", reset_counter=True)
        plot_combined_calibration(dataset_tag="train (OOF)", task_name=task_name, model_preds=cal_train_data)
        
        if len(cal_test_data) > 0:
            out_mgr.set_tag(f"{task_s}__cal_test", reset_counter=True)
            plot_combined_calibration(dataset_tag="test", task_name=task_name, model_preds=cal_test_data)
        
        if len(cal_external_data) > 0:
            out_mgr.set_tag(f"{task_s}__cal_external", reset_counter=True)
            plot_combined_calibration(dataset_tag="external", task_name=task_name, model_preds=cal_external_data)

    best_by_train = None
    best_auc = None
    if len(train_auc_df) > 0:
        best_by_train = train_auc_df.iloc[0]["model"]
        best_auc = float(train_auc_df.iloc[0]["auc_oof_train"])

    ret = {
        "task": task_name,
        "best_model_by_train_oof": best_by_train,
        "best_auc_oof_train": best_auc,
        "fs_selected_n": int(fs_meta.get("selected_n", len(sel_cols))),
        "fs_mode": fs_meta.get("mode"),
        "fs_selected_features": list(fs_meta.get("selected_features", sel_cols)),
        "train_auc_table": train_auc_df,
        "test_auc_table": test_auc_df,
        "external_auc_table": external_auc_df,
        "split_enabled": split_enabled,
        "external_enabled": external_enabled,
        "n_train": int(len(train_ids)),
        "n_test": int(len(test_ids)) if split_enabled else 0,
        "n_external": int(len(y_ext)) if y_ext is not None else 0,
        "train_ids": train_ids,
        "test_ids": test_ids,
    }
    return ret


# ============================================================
# 11) Experiment Runner
# ============================================================

def run_one_experiment(cfg: dict, out_dir: Path) -> pd.DataFrame:
    """Run one experiment-plan row."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_fp = (out_dir / "run.log").open("w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = Tee(old_out, log_fp)
    sys.stderr = Tee(old_err, log_fp)

    globals()["RANDOM_STATE"] = int(cfg["RANDOM_STATE"])
    globals()["N_SPLITS"] = int(cfg["N_SPLITS"])
    np.random.seed(int(cfg["RANDOM_STATE"]))

    write_settings_txt(out_dir, cfg, base_dir=Path(cfg["OUTCOME_XLSX"]).resolve().parent)

    out_mgr = OutputManager(out_dir, dpi=300)

    try:
        with capture_outputs(out_mgr):
            out_df = read_outcomes(cfg["OUTCOME_XLSX"], sheet_name=cfg["OUTCOME_SHEET"])
            y_dict = build_y_module(out_df)

            X_all, parts = build_full_feature_table_custom(
                out_df=out_df,
                suv_path=cfg["SUV_XLSX"],
                t2w_path=cfg["T2W_XLSX"],
                adc_path=cfg.get("ADC_XLSX"),
                suv_sheets=cfg["SUV_SHEETS_DEFAULT"],
                suv_image_types=cfg["SUV_IMAGE_TYPES_DEFAULT"],
                t2w_sheets=cfg["T2W_SHEETS_DEFAULT"],
                t2w_image_types=cfg["T2W_IMAGE_TYPES_DEFAULT"],
                adc_sheets=cfg.get("ADC_SHEETS_DEFAULT", []),
                adc_image_types=cfg.get("ADC_IMAGE_TYPES_DEFAULT", []),
                include_clinical=bool(cfg["INCLUDE_CLINICAL"]),
                include_bsiup=bool(cfg["INCLUDE_bSIUP_GG"]),
                clinical_base_features=cfg["CLINICAL_BASE_FEATURES"],
            )

            print("Outcome rows:", out_df.shape)
            print("All features shape:", X_all.shape)
            print("\nModule shapes:")
            for k, dfp in parts.items():
                print(f"  {k}: {dfp.shape}")

            external_min = cfg.get("EXTERNAL_ID_MIN")
            external_max = cfg.get("EXTERNAL_ID_MAX")
            if external_min is not None and external_max is not None:
                print(f"[EXTERNAL] External validation enabled. ID range=[{external_min},{external_max}]")
                external_ids = out_df[(out_df["ID"] >= external_min) & (out_df["ID"] <= external_max)]["ID"].values
                print(f"[EXTERNAL] Found {len(external_ids)} external validation samples")
                if len(external_ids) == 0:
                    print("[WARN] No external validation samples found in the specified ID range")
                    X_ext = None
                else:
                    X_ext = X_all.loc[external_ids].copy()
                    print(f"[EXTERNAL] External features shape: {X_ext.shape}")
            else:
                print("[EXTERNAL] External validation not specified (EXTERNAL_ID_MIN/MAX not set)")
                X_ext = None
                external_ids = None

            tasks = ["EPE"]
            results_rows = []
            train_ids_global = None
            test_ids_global = None

            for task in tasks:
                task_s = _sanitize_filename(task)
                task_dir = out_dir / f"task_{task_s}"
                task_dir.mkdir(parents=True, exist_ok=True)

                out_mgr.set_dir(task_dir)
                out_mgr.set_tag(f"{task_s}__main", reset_counter=True)

                ret = run_one_task_all_models(task, y_dict, X_all, X_ext, out_df, cfg, out_mgr)
                if ret is None:
                    continue

                if train_ids_global is None:
                    train_ids_global = ret.get("train_ids")
                    test_ids_global = ret.get("test_ids")

                results_rows.append({
                    "task": ret["task"],
                    "best_model_by_train_oof": ret["best_model_by_train_oof"],
                    "best_auc_oof_train": ret["best_auc_oof_train"],
                    "split_enabled": ret["split_enabled"],
                    "n_train": ret["n_train"],
                    "n_test": ret["n_test"],
                    "fs_mode": ret["fs_mode"],
                    "fs_selected_n": ret["fs_selected_n"],
                })

            if train_ids_global is not None:
                train_data = out_df[out_df["ID"].isin(train_ids_global)]
                test_data = out_df[out_df["ID"].isin(test_ids_global)] if test_ids_global is not None else pd.DataFrame()
                
                clinical_outcomes_path = out_dir / "Clinical_Outcomes.xlsx"
                with pd.ExcelWriter(clinical_outcomes_path) as writer:
                    train_data.to_excel(writer, sheet_name="Training Set", index=True)
                    test_data.to_excel(writer, sheet_name="Validation Set", index=True)
                print(f"[OUTCOMES] Saved Clinical Outcomes to {clinical_outcomes_path}")

            out_mgr.set_dir(out_dir)

            if len(results_rows) > 0:
                summary_df = pd.DataFrame(results_rows).sort_values(
                    ["best_auc_oof_train", "task"], ascending=[False, True]
                )
                return summary_df
            else:
                print("No tasks finished. Please check missing labels / file paths / sheet names.")
                return pd.DataFrame(columns=[
                    "task", "best_model_by_train_oof", "best_auc_oof_train",
                    "split_enabled", "n_train", "n_test", "fs_mode", "fs_selected_n"
                ])
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        try:
            log_fp.close()
        except Exception:
            pass


# ============================================================
# 12) Prediction Saving
# ============================================================

def save_model_predictions(
    exp_dir, task_name, model_name,
    train_probs, test_probs, train_y, test_y,
    external_probs=None, external_y=None,
    train_outcomes=None, test_outcomes=None, external_outcomes=None,
):
    """
    Save model predictions, labels, IDs, and matched outcome rows.

    Outcome rows are stored as object arrays with companion column-name arrays.
    """
    task_s = _sanitize_filename(task_name)
    task_dir = exp_dir / f"task_{task_s}"
    task_dir.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        "train_probs": train_probs,
        "test_probs": test_probs,
        "train_y": train_y,
        "test_y": test_y
    }
    
    if external_probs is not None and external_y is not None:
        save_dict["external_probs"] = external_probs
        save_dict["external_y"] = external_y

    def _add_outcomes(prefix: str, df: pd.DataFrame):
        if df is None:
            return
        save_dict[f"{prefix}_ids"] = df.index.to_numpy()
        save_dict[f"{prefix}_outcome_columns"] = df.columns.to_numpy(dtype=object)
        save_dict[f"{prefix}_outcomes"] = df.to_numpy(dtype=object)

    _add_outcomes("train", train_outcomes)
    _add_outcomes("test", test_outcomes)
    _add_outcomes("external", external_outcomes)
    
    np.savez(task_dir / f"{model_name}_predictions.npz", **save_dict)


# ============================================================
# 13) Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_xlsx", type=str, required=True, help="批量实验计划表（Excel）")
    parser.add_argument("--plan_sheet", type=str, default=None, help="计划表 sheet 名（默认第一个）")
    parser.add_argument("--out_root", type=str, required=True, help="批量输出根目录")
    args = parser.parse_args()

    plan_path = Path(args.plan_xlsx).resolve()
    base_dir = plan_path.parent
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    plan_df = read_plan_xlsx(str(plan_path), sheet_name=args.plan_sheet)

    all_runs = []
    for i, row in plan_df.iterrows():
        job_id_raw = row.get("job_id", None) if isinstance(row, pd.Series) else None
        job_id_int = _parse_int(job_id_raw, None)
        if job_id_int is not None:
            job_id_str = f"{job_id_int:03d}"
        else:
            if _is_nan(job_id_raw) or str(job_id_raw).strip() == "":
                job_id_str = f"{i+1:03d}"
            else:
                job_id_str = _sanitize_filename(job_id_raw)

        exp_dir = out_root / job_id_str
        exp_dir.mkdir(parents=True, exist_ok=True)

        cfg = normalize_config_row(row, base_dir=base_dir)

        print("\n" + "=" * 90)
        print(f"[RUN] job_id={job_id_str}  ->  {exp_dir}")
        print("=" * 90)

        summary_df = run_one_experiment(cfg, exp_dir)
        if summary_df is not None and len(summary_df) > 0:
            tmp = summary_df.copy()
            tmp.insert(0, "job_id", job_id_str)
            all_runs.append(tmp)

    if len(all_runs) > 0:
        agg = pd.concat(all_runs, ignore_index=True)
        try:
            agg.to_excel(out_root / "ALL_EXPERIMENTS_summary.xlsx", index=False)
        except Exception:
            agg.to_csv(out_root / "ALL_EXPERIMENTS_summary.csv", index=False, encoding="utf-8")

    print("\n[DONE] All experiments finished.")
    print("Output root:", out_root)

if __name__ == "__main__":
    main()
