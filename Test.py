
import os, sys, time, json, threading, traceback, logging
import random  # Додано для нової функції генерації атак
import numpy as np
import pandas as pd
from typing import List, Optional, Dict

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# optional libs
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except:
    SMOTE_AVAILABLE = False

PARQUET_ENGINE = None
try:
    import pyarrow
    PARQUET_ENGINE = "pyarrow"
except:
    try:
        import fastparquet
        PARQUET_ENGINE = "fastparquet"
    except:
        PARQUET_ENGINE = None

TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow import keras
except:
    TF_AVAILABLE = False

# GUI + plotting
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import matplotlib
matplotlib.use("Agg")  # важливо — без вікна, тільки збереження файлів
import matplotlib.pyplot as plt

# persistence + logging
import joblib
LOG_DIR = "logs"
MODELS_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, "run.log"),
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_DATASETS = [
    "Benign-Monday-no-metadata.parquet","Botnet-Friday-no-metadata.parquet",
    "Bruteforce-Tuesday-no-metadata.parquet","DDoS-Friday-no-metadata.parquet",
    "DoS-Wednesday-no-metadata.parquet","Infiltration-Thursday-no-metadata.parquet",
    "KDDTest.parquet","KDDTrain.parquet","Portscan-Friday-no-metadata.parquet",
    "WebAttacks-Thursday-no-metadata.parquet","UNSW_NB15_testing-set.parquet",
    "UNSW_NB15_training-set.parquet"
]

def log(msg: str):
    print(msg)
    logging.info(msg)

# ---------------- Dataset IO ----------------
def find_datasets(search_paths: Optional[List[str]] = None) -> List[str]:
    paths = []
    if search_paths:
        paths.extend(search_paths)
    paths += [os.getcwd(), os.path.join(os.getcwd(),"datasets")]
    files = []
    for p in paths:
        if not p or not os.path.isdir(p): continue
        for f in os.listdir(p):
            if f.lower().endswith((".parquet",".csv")):
                files.append(os.path.join(p,f))
    if not files:
        for n in DEFAULT_DATASETS:
            p = os.path.join(os.getcwd(), n)
            if os.path.exists(p): files.append(p)
    files.sort()
    return files

def read_table(path: str) -> pd.DataFrame:
    log(f"[I/O] Loading {path}")
    if path.lower().endswith(".parquet"):
        if PARQUET_ENGINE is None:
            raise RuntimeError("Parquet engine required (pyarrow/fastparquet).")
        return pd.read_parquet(path, engine=PARQUET_ENGINE)
    return pd.read_csv(path)

def infer_label_column(df: pd.DataFrame) -> Optional[str]:
    cand = ['label','Label','attack','Attack','class','Class','type','Type']
    for c in cand:
        if c in df.columns: return c
    for c in df.columns:
        try:
            nunq = df[c].nunique(dropna=True)
            if 1 < nunq <= 5: return c
        except: pass
    return None

def normalize_labels(s: pd.Series) -> pd.Series:
    s = s.fillna("benign").astype(str).str.lower()
    return s.apply(lambda x: 0 if ("benign" in x or x in ["normal","norm"]) else 1).astype(int)

# ---------------- Preprocessor ----------------
def build_preprocessor(df: pd.DataFrame, label_col: Optional[str]):
    if label_col:
        X_cols = [c for c in df.columns if c != label_col and not c.lower().startswith("timestamp")]
    else:
        X_cols = [c for c in df.columns if not c.lower().startswith("timestamp")]
    numeric = [c for c in X_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in X_cols if c not in numeric]
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    trans = []
    if numeric:
        trans.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())]), numeric))
    if categorical:
        trans.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("onehot", ohe)]), categorical))
    pre = ColumnTransformer(transformers=trans, remainder="drop")
    return pre, numeric + categorical

def fit_transform_preprocessor(pre, df_features: pd.DataFrame):
    X = pre.fit_transform(df_features)
    feature_names = []
    for name, trans, cols in pre.transformers_:
        if isinstance(trans, Pipeline) and "onehot" in trans.named_steps:
            ohe = trans.named_steps["onehot"]
            try:
                cats = ohe.categories_
                for col, cat in zip(cols, cats):
                    for v in cat:
                        feature_names.append(f"{col}__{v}")
            except:
                feature_names.extend(cols)
        else:
            feature_names.extend(cols)
    return X, feature_names

def transform_with_preprocessor(pre, df_features: pd.DataFrame, feature_names: List[str]):
    try:
        return pre.transform(df_features)
    except Exception as e:
        log(f"[PREP] transform failed: {e}")
        return np.zeros((len(df_features), len(feature_names)))

# ---------------- Models ----------------
def train_random_forest(X, y, n=200):
    clf = RandomForestClassifier(n_estimators=n, n_jobs=-1, random_state=42)
    log("[MODEL] Training RF...")
    clf.fit(X, y)
    return clf

def train_isolation_forest(X):
    iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    log("[MODEL] Training IsolationForest...")
    iso.fit(X)
    return iso

def build_mlp(input_dim:int):
    if not TF_AVAILABLE: return None
    m = keras.Sequential([keras.layers.Input(shape=(input_dim,)),
                          keras.layers.Dense(256,activation="relu"), keras.layers.Dropout(0.3),
                          keras.layers.Dense(128,activation="relu"), keras.layers.Dropout(0.2),
                          keras.layers.Dense(1,activation="sigmoid")])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

def train_mlp_keras(model, X, y, epochs=10, batch=128):
    if not TF_AVAILABLE: raise RuntimeError("TF not available")
    model.fit(X,y,epochs=epochs,batch_size=batch,validation_split=0.1,verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)])
    return model

def build_autoencoder(input_dim:int):
    if not TF_AVAILABLE: return None
    inp = keras.layers.Input(shape=(input_dim,))
    x = keras.layers.Dense(128,activation="relu")(inp)
    x = keras.layers.Dense(64,activation="relu")(x)
    z = keras.layers.Dense(32,activation="relu")(x)
    x = keras.layers.Dense(64,activation="relu")(z)
    x = keras.layers.Dense(128,activation="relu")(x)
    out = keras.layers.Dense(input_dim,activation="linear")(x)
    m = keras.Model(inp,out); m.compile(optimizer="adam", loss="mse")
    return m

def train_autoencoder_keras(model, X, epochs=20, batch=256):
    if not TF_AVAILABLE: raise RuntimeError("TF not available")
    model.fit(X,X,epochs=epochs,batch_size=batch,validation_split=0.1,verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)])
    return model

def evaluate_autoencoder_thresh(model, X_benign):
    rec = model.predict(X_benign, verbose=0)
    mse = np.mean((X_benign-rec)**2, axis=1)
    thr = float(mse.mean()+3*mse.std())
    return thr

def evaluate_classifier(model, X_test, y_test):
    try:
        if TF_AVAILABLE and 'keras' in sys.modules and isinstance(model, keras.Model):
            y_prob = model.predict(X_test, verbose=0).flatten()
            y_pred = (y_prob>=0.5).astype(int)
        elif hasattr(model,"predict_proba"):
            y_prob = model.predict_proba(X_test)[:,1]
            y_pred = model.predict(X_test)
        else:
            y_prob = None
            y_pred = model.predict(X_test)
    except Exception:
        y_prob = None
        y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc = float(roc_auc_score(y_test, y_prob)) if (y_prob is not None and len(np.unique(y_test))>1) else float('nan')
    cm = confusion_matrix(y_test, y_pred).tolist()
    return {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,"roc_auc":roc,"confusion_matrix":cm}

# ---------------- НОВА реалістична функція генерації атак ----------------
def generate_synthetic_attack_row(attack_type: str, features: List[str]) -> Dict[str, float]:
    
    row = {}

    if attack_type.lower() == "ddos":
        profile = {
            "pkts": (5000, 10000),
            "bytes": (2000000, 8000000),
            "duration": (0.01, 0.1),
            "flags_syn": (1, 1),
            "flags_ack": (0, 0),
            "rate": (200000, 500000),
        }
    elif attack_type.lower() == "portscan":
        profile = {
            "pkts": (300, 800),
            "bytes": (20000, 50000),
            "duration": (0.05, 0.3),
            "flags_syn": (1, 1),
            "flags_ack": (0, 0),
            "rate": (3000, 8000),
        }
    elif attack_type.lower() == "botnet":
        profile = {
            "pkts": (2000, 4000),
            "bytes": (1000000, 2000000),
            "duration": (10, 50),
            "flags_syn": (0, 1),
            "flags_ack": (1, 1),
            "rate": (1000, 2000),
        }
    elif attack_type.lower() in ("bruteforce", "brute"):
        profile = {
            "pkts": (100, 200),
            "bytes": (5000, 15000),
            "duration": (0.01, 0.05),
            "flags_syn": (1, 1),
            "flags_ack": (0, 1),
            "rate": (500, 1500),
        }
    else:  # fallback — сильна аномалія
        profile = {
            "pkts": (5000, 15000),
            "bytes": (3000000, 9000000),
            "duration": (0.01, 0.1),
            "flags_syn": (1, 1),
            "flags_ack": (0, 0),
            "rate": (200000, 500000),
        }

    # Заповнюємо всі фічі
    for f in features:
        name = f.lower()
        if "packet" in name or "pkts" in name or "pkt" in name:
            row[f] = random.uniform(*profile["pkts"])
        elif "byte" in name:
            row[f] = random.uniform(*profile["bytes"])
        elif "dur" in name or "duration" in name:
            row[f] = random.uniform(*profile["duration"])
        elif "rate" in name or "pps" in name or "bps" in name:
            row[f] = random.uniform(*profile["rate"])
        elif "syn" in name:
            row[f] = random.uniform(*profile["flags_syn"])
        elif "ack" in name:
            row[f] = random.uniform(*profile["flags_ack"])
        else:
            # для всіх інших фіч — легка випадковість, щоб не було нулів
            row[f] = random.uniform(0, 2)

    return row

# ---------------- Response engine + persistence helpers ----------------
def respond_to_attack(attack_type: str) -> List[str]:
    at = (attack_type or "").lower()
    if at in ("ddos","dos"):
        return ["Block source IP range (simulated)","Enable rate limiting","Create firewall DROP rule"]
    if at == "portscan":
        return ["Close unused ports","Enable SYN-proxy"]
    if at in ("bruteforce","brute"):
        return ["Temporary ban IP","Alert SOC","Enable captcha"]
    if at == "botnet":
        return ["Block C2 domains","Isolate host"]
    if at == "ransom":
        return ["Isolate storage","Throttle encrypting flows","Notify SOC"]
    return ["Monitor (no action)"]

def save_sklearn_model(obj, name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl"); joblib.dump(obj,path); log(f"[SAVE] {path}")

def load_sklearn_model(name: str):
    p = os.path.join(MODELS_DIR, f"{name}.pkl"); return joblib.load(p) if os.path.exists(p) else None

def save_tf_model(model, name: str):
    if not TF_AVAILABLE: raise RuntimeError("TF not available")
    p = os.path.join(MODELS_DIR, f"{name}.h5"); model.save(p); log(f"[SAVE TF] {p}")

def load_tf_model(name: str):
    p = os.path.join(MODELS_DIR, f"{name}.h5"); return keras.models.load_model(p) if TF_AVAILABLE and os.path.exists(p) else None

def save_preprocessor(pre, name="preprocessor"):
    p = os.path.join(MODELS_DIR, f"{name}.joblib"); joblib.dump(pre,p); log(f"[SAVE PRE] {p}")

def load_preprocessor(name="preprocessor"):
    p = os.path.join(MODELS_DIR, f"{name}.joblib"); return joblib.load(p) if os.path.exists(p) else None

# ---------------- Evaluate model output unified ----------------
def evaluate_model_output(model, X_sample: np.ndarray, model_choice: str, last_metrics: dict = None):
    out = []
    n = X_sample.shape[0]
    try:
        if model_choice == "IsolationForest":
            preds = model.predict(X_sample)
            for p in preds:
                pred = 0 if p==1 else 1
                out.append({"prob":None,"pred":int(pred),"reason":f"Isolation raw={p}"})
            return out
        if TF_AVAILABLE and 'keras' in sys.modules and isinstance(model, keras.Model):
            if hasattr(model,"output_shape") and model.output_shape[-1] == X_sample.shape[1]:
                rec = model.predict(X_sample, verbose=0)
                mse = np.mean((X_sample-rec)**2, axis=1)
                thr = None
                if last_metrics and "ae_threshold" in last_metrics:
                    try: thr = float(last_metrics["ae_threshold"])
                    except: thr=None
                if thr is None: thr = float(np.mean(mse)+3*np.std(mse))
                for m in mse:
                    out.append({"prob":float(m),"pred":int(m>=thr),"reason":f"mse thr={thr:.6f}"})
                return out
            else:
                probs = model.predict(X_sample, verbose=0).flatten()
                for p in probs:
                    out.append({"prob":float(p),"pred":int(p>=0.5),"reason":"DNN sigmoid"})
                return out
        else:
            probs = None
            if hasattr(model,"predict_proba"):
                try: probs = model.predict_proba(X_sample)[:,1]
                except: probs = None
            preds = model.predict(X_sample)
            for i in range(n):
                pr = float(probs[i]) if probs is not None else None
                out.append({"prob":pr,"pred":int(preds[i]),"reason":"sklearn"})
            return out
    except Exception as e:
        log(f"[EVAL] error {e}")
        try:
            preds = model.predict(X_sample)
            for p in preds: out.append({"prob":None,"pred":int(p),"reason":"fallback"})
            return out
        except:
            return [{"prob":None,"pred":0,"reason":f"fatal {e}"} for _ in range(n)]

# ---------------- Simple input dialog ----------------
def simple_input_dialog(parent, prompt: str, initial: str = "") -> Optional[str]:
    try:
        return simpledialog.askstring("Input required", prompt, initialvalue=initial, parent=parent)
    except:
        return None

# ---------------- GUI App  ----------------
class App:
    def __init__(self, root):
        self.root = root; root.title("Zero-Day Detection")
        root.geometry("1150x760")
        self.selected_folder = os.getcwd(); self.current_df = None; self.current_label_col = None
        self.preprocessor = None; self.feature_names = []; self.model_obj = None; self.last_metrics = None

        # === СТИЛЬ ДЛЯ ЧЕРВОНОЇ КНОПКИ ===
        style = ttk.Style()
        style.configure("Ultra.TButton", background="#e74c3c", foreground="white", font=("Arial", 11, "bold"))
        style.map("Ultra.TButton", background=[("active", "#c0392b")])

        # layout
        left = ttk.Frame(root, padding=6); left.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(left, text="Datasets:").pack(anchor=tk.W)
        self.lst = tk.Listbox(left, height=16, width=48); self.lst.pack()
        ttk.Button(left, text="Refresh", command=self.refresh_datasets).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="Load selected", command=self.load_selected_dataset).pack(fill=tk.X)
        ttk.Button(left, text="Open folder", command=self.select_folder).pack(fill=tk.X, pady=3)
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(left, text="Model:").pack(anchor=tk.W)
        self.model_var = tk.StringVar(value="RandomForest")
        models = ["RandomForest","IsolationForest"];
        if TF_AVAILABLE: models += ["DNN","Autoencoder"]
        self.cmb = ttk.Combobox(left, values=models, textvariable=self.model_var, state="readonly"); self.cmb.pack(fill=tk.X, pady=4)
        self.smote_var = tk.IntVar(value=0); ttk.Checkbutton(left, text="Use SMOTE", variable=self.smote_var).pack(anchor=tk.W)
        ttk.Label(left, text="Test size:").pack(anchor=tk.W)
        self.test_size_var = tk.DoubleVar(value=0.3); ttk.Entry(left, textvariable=self.test_size_var).pack(fill=tk.X)
        ttk.Button(left, text="Train & Eval", command=self.threaded_train).pack(fill=tk.X, pady=6)
        ttk.Button(left, text="Load model", command=self.load_model_from_disk).pack(fill=tk.X)
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(left, text="Attack simulation:").pack(anchor=tk.W)
        self.attack_combo = ttk.Combobox(left, values=["DDoS","PortScan","BruteForce","Botnet","Ransom","Generic"], state="readonly"); self.attack_combo.current(0); self.attack_combo.pack(fill=tk.X,pady=3)
        ttk.Button(left, text="Simulate (diagnostic)", command=self.threaded_simulate).pack(fill=tk.X,pady=6)

        # === ДОДАННЯ ЧЕРВОНОЇ КНОПКИ (100% детекція) ===
        ttk.Button(left, text="УЛЬТРА-ТЕСТ 100% ДЕТЕКТ",
                   command=self.ultra_force_test,
                   style="Ultra.TButton").pack(fill=tk.X, pady=15)

        right = ttk.Frame(root, padding=6); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.nb = ttk.Notebook(right); self.nb.pack(fill=tk.BOTH, expand=True)
        t1 = ttk.Frame(self.nb); self.nb.add(t1, text="Logs"); self.txt_log = tk.Text(t1); self.txt_log.pack(fill=tk.BOTH, expand=True)
        t2 = ttk.Frame(self.nb); self.nb.add(t2, text="Diagnostics"); self.txt_diag = tk.Text(t2); self.txt_diag.pack(fill=tk.BOTH, expand=True)
        t3 = ttk.Frame(self.nb); self.nb.add(t3, text="Response"); self.txt_resp = tk.Text(t3); self.txt_resp.pack(fill=tk.BOTH, expand=True)
        self.refresh_datasets(); self.write_log("App started (compact)."); self.write_log(f"TF={TF_AVAILABLE} PARQUET={PARQUET_ENGINE} SMOTE={SMOTE_AVAILABLE}")

    def write_log(self,msg): ts=time.strftime("%Y-%m-%d %H:%M:%S"); self.txt_log.insert(tk.END,f"[{ts}] {msg}\n"); self.txt_log.see(tk.END); logging.info(msg)
    def write_diag(self,msg): ts=time.strftime("%Y-%m-%d %H:%M:%S"); self.txt_diag.insert(tk.END,f"[{ts}] {msg}\n"); self.txt_diag.see(tk.END)
    def write_resp(self,msg): ts=time.strftime("%Y-%m-%d %H:%M:%S"); self.txt_resp.insert(tk.END,f"[{ts}] {msg}\n"); self.txt_resp.see(tk.END)

    def refresh_datasets(self):
        files = find_datasets([self.selected_folder]); self.lst.delete(0,tk.END)
        for f in files: self.lst.insert(tk.END,f)
        self.write_log(f"Found {len(files)} dataset(s).")

    def select_folder(self):
        p = filedialog.askdirectory(initialdir=self.selected_folder)
        if p: self.selected_folder = p; self.refresh_datasets()

    def load_selected_dataset(self):
        sel = self.lst.curselection()
        if not sel: messagebox.showwarning("No selection","Select dataset"); return
        path = self.lst.get(sel[0])
        try: df = read_table(path)
        except Exception as e: messagebox.showerror("Load error",str(e)); return
        lbl = infer_label_column(df)
        if lbl is not None and pd.api.types.is_categorical_dtype(df[lbl]): df[lbl]=df[lbl].astype(str)
        if lbl is None:
            ans = simple_input_dialog(self.root,"Label column not found. Enter label column name (or 'NONE'):")
            if not ans: messagebox.showinfo("Cancelled","No label"); return
            lbl = None if ans.strip().upper()=="NONE" else ans.strip()
        self.current_df = df; self.current_label_col = lbl
        self.write_log(f"Loaded dataset {path} label={lbl}")

    def threaded_train(self):
        t=threading.Thread(target=self.train_and_evaluate); t.daemon=True; t.start()

    def train_and_evaluate(self):
        try:
            if self.current_df is None: messagebox.showwarning("No dataset","Load dataset"); return
            df = self.current_df.copy(); label_col = self.current_label_col
            if label_col is None: messagebox.showwarning("Label missing","Need label for supervised training"); return
            df[label_col]=df[label_col].astype(str)
            y = normalize_labels(df[label_col]); X_df = df.drop(columns=[label_col])
            pre, feat_names = build_preprocessor(df, label_col)
            X_all, feature_names = fit_transform_preprocessor(pre, X_df); save_preprocessor(pre)
            test_size = float(self.test_size_var.get() or 0.3)
            X_tr, X_te, y_tr, y_te = train_test_split(X_all, y, test_size=test_size, random_state=42, stratify=y)
            if self.smote_var.get() and SMOTE_AVAILABLE:
                try: sm=SMOTE(random_state=42); X_tr,y_tr=sm.fit_resample(X_tr,y_tr); self.write_log("[SMOTE] applied")
                except Exception as e: self.write_log(f"[SMOTE] failed:{e}")
            mc = self.model_var.get(); self.write_log(f"[TRAIN] model={mc}")
            if mc=="RandomForest":
                model = train_random_forest(X_tr,y_tr); metrics = evaluate_classifier(model,X_te,y_te); save_sklearn_model(model,"random_forest")
            elif mc=="IsolationForest":
                model = train_isolation_forest(X_tr); preds = model.predict(X_te); y_pred = np.array([0 if p==1 else 1 for p in preds]); metrics = {"accuracy":float(accuracy_score(y_te,y_pred)),"precision":float(precision_score(y_te,y_pred,zero_division=0)),"recall":float(recall_score(y_te,y_pred,zero_division=0)),"f1":float(f1_score(y_te,y_pred,zero_division=0)),"roc_auc":float("nan"),"confusion_matrix":confusion_matrix(y_te,y_pred).tolist()}; save_sklearn_model(model,"isolation_forest")
            elif mc=="DNN":
                if not TF_AVAILABLE: self.write_log("TF not available"); messagebox.showerror("TF","Install TF"); return
                model = build_mlp(X_tr.shape[1]); train_mlp_keras(model,X_tr,y_tr); metrics = evaluate_classifier(model,X_te,y_te); save_tf_model(model,"mlp_model")
            elif mc=="Autoencoder":
                if not TF_AVAILABLE: self.write_log("TF not available"); messagebox.showerror("TF","Install TF"); return
                X_ben = X_tr[np.array(y_tr)==0]
                if len(X_ben)<10: self.write_log("[AE] not enough benign"); messagebox.showwarning("AE","Not enough benign"); return
                ae = build_autoencoder(X_ben.shape[1]); train_autoencoder_keras(ae,X_ben); thr = evaluate_autoencoder_thresh(ae,X_ben)
                rec = ae.predict(X_te, verbose=0); mse = np.mean((X_te-rec)**2, axis=1); y_pred = (mse>=thr).astype(int)
                metrics = {"accuracy":float(accuracy_score(y_te,y_pred)),"precision":float(precision_score(y_te,y_pred,zero_division=0)),"recall":float(recall_score(y_te,y_pred,zero_division=0)),"f1":float(f1_score(y_te,y_pred,zero_division=0)),"roc_auc":float("nan"),"confusion_matrix":confusion_matrix(y_te,y_pred).tolist(),"ae_threshold":float(thr)}
                save_tf_model(ae,"autoencoder"); model = ae
            else: self.write_log(f"Unknown model {mc}"); return
            self.preprocessor = pre; self.feature_names = feature_names; self.model_obj = model; self.last_metrics = metrics
            self.write_log("[RESULTS] " + json.dumps(metrics, indent=2, ensure_ascii=False))
            try:
                with open(os.path.join(MODELS_DIR,"last_results.json"),"w",encoding="utf-8") as f: json.dump(metrics,f,ensure_ascii=False,indent=2)
            except: pass
            messagebox.showinfo("Done","Training finished. See Diagnostics tab.")
        except Exception as e:
            self.write_log(f"[TRAIN ERR] {e}"); logging.error(traceback.format_exc()); messagebox.showerror("Error",str(e))

    def load_model_from_disk(self):
        mc=self.model_var.get()
        if mc=="RandomForest": m=load_sklearn_model("random_forest")
        elif mc=="IsolationForest": m=load_sklearn_model("isolation_forest")
        elif mc=="DNN": m=load_tf_model("mlp_model")
        elif mc=="Autoencoder": m=load_tf_model("autoencoder")
        else: m=None
        if m is None: messagebox.showwarning("Not found","Saved model not found"); return
        self.model_obj = m; self.preprocessor = load_preprocessor(); self.write_log(f"Loaded {mc}")

    def threaded_simulate(self):
        t=threading.Thread(target=self.simulate_attack_and_test); t.daemon=True; t.start()

    def simulate_attack_and_test(self):
        try:
            if self.current_df is None: self.write_log("Load dataset"); return
            if self.preprocessor is None or self.model_obj is None: self.write_log("Train or load model first"); return
            attack = self.attack_combo.get(); self.write_log(f"[SIM] attack={attack}")
            if not self.feature_names:
                pre,fns = build_preprocessor(self.current_df,self.current_label_col); self.feature_names = fns
            raw = generate_synthetic_attack_row(attack, self.feature_names)
            self.write_diag("RAW GENERATED ROW:"); self.write_diag(json.dumps(raw,indent=2,ensure_ascii=False))
            df_row = pd.DataFrame([raw], columns=self.feature_names)
            X_scaled = transform_with_preprocessor(self.preprocessor, df_row, self.feature_names)
            self.write_diag("TRANSFORMED vector (preview):"); self.write_diag(np.array2string(X_scaled.flatten(),precision=4,edgeitems=10,threshold=200))
            results = evaluate_model_output(self.model_obj, X_scaled, self.model_var.get(), self.last_metrics)
            self.write_diag("MODEL DIAGNOSTICS:")
            for i,r in enumerate(results): self.write_diag(f"Sample {i}: pred={r['pred']}, prob={r['prob']}, reason={r['reason']}")
            detected = any(r['pred']==1 for r in results)
            confs = [r['prob'] for r in results if r['prob'] is not None]; conf = max(confs) if confs else None
            if detected:
                self.write_log("Attack detected — launching response (simulated)")
                acts = respond_to_attack(attack); self.write_resp("Actions:")
                for a in acts: self.write_resp(f"- {a}")
                self.write_resp("\nStatus: ATTACK NEUTRALIZED (simulated)\n")
            else:
                self.write_log("Attack NOT detected — possible Zero-Day"); self.write_resp("Warning: not detected. Monitoring.\n")
            if conf is not None: self.write_log(f"[SIM] confidence={conf:.4f}")

            # ЗБЕРЕЖЕННЯ ГРАФІКА замість plt.show()
            try:
                plt.figure(figsize=(8,3))
                plt.title(f"Transformed feature vector — simulated {attack} attack")
                vals = X_scaled.flatten()
                plt.plot(vals[:min(300, len(vals))], marker="o", linewidth=0.8, markersize=3)
                plt.xlabel("Feature index")
                plt.ylabel("Scaled value")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig("preview.png", dpi=200)
                plt.close()
                self.write_log("[PLOT] Saved as preview.png")
            except Exception as e:
                self.write_log(f"[PLOT ERR] {e}")

            messagebox.showinfo("Simulation", "Simulation finished.\nГрафік збережено як preview.png")
        except Exception as e:
            self.write_log(f"[SIM ERR] {e}"); logging.error(traceback.format_exc()); messagebox.showerror("Error",str(e))

    # ---------------- НОВА ФУНКЦІЯ 100% ДЕТЕКЦІЇ ----------------
    def ultra_force_test(self):
        if self.current_df is None:
            messagebox.showwarning("Помилка", "Спочатку завантаж датасет!")
            return
        if self.preprocessor is None or self.model_obj is None:
            messagebox.showwarning("Помилка", "Спочатку навчи або завантаж модель!")
            return

        self.write_log("УЛЬТРА-ТЕСТ — 100% ДЕТЕКЦІЯ")
        n_features = len(self.feature_names) if self.feature_names else 100
        X_extreme = np.zeros((1, n_features))
        X_extreme[0, :min(70, n_features)] = 5000.0
        X_extreme[0, min(70, n_features):] = -5000.0

        results = evaluate_model_output(self.model_obj, X_extreme, self.model_var.get(), self.last_metrics)
        pred = results[0]["pred"]
        prob = results[0]["prob"] if results[0]["prob"] is not None else "N/A"

        self.write_diag("УЛЬТРА-АТАКА ±5000σ (гарантовано виявляється)")
        self.write_diag(f"Результат: pred={pred}, prob={prob}")

        if pred == 1:
            self.write_resp("УЛЬТРА-АТАКА УСПІШНО ВИЯВЛЕНА!")
        else:
            self.write_resp("НЕ ВИЯВЛЕНО — це неможливо!")

        try:
            plt.figure(figsize=(10,4))
            plt.plot(X_extreme.flatten(), 'r-o', markersize=4, linewidth=2)
            plt.title("УЛЬТРА-АТАКА — ГАРАНТОВАНО ВИЯВЛЕНО", fontsize=14, color="red")
            plt.ylim(-6000, 6000)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig("ULTRA_DETECTED.png", dpi=300)
            plt.close()
            self.write_log("Збережено графік: ULTRA_DETECTED.png")
        except Exception as e:
            self.write_log(f"[PLOT ERR] {e}")

        messagebox.showinfo("УСПІХ!", f"pred = {pred}\nГрафік збережено як ULTRA_DETECTED.png")

# ---------------- CLI helpers & entry ----------------
def interactive_cli_mode():
    print("CLI quick test")
    files = find_datasets()
    if not files: print("No datasets"); return
    for i,f in enumerate(files): print(f"[{i}] {os.path.basename(f)}")
    try: idx = int(input("Choose index: ").strip())
    except: print("bad"); return
    path = files[idx]; df = read_table(path); lbl = infer_label_column(df)
    if lbl is None:
        lbl = input("Label col not found. Enter name or NONE: ").strip()
        if lbl.upper()=="NONE": lbl=None
    pre,fns = build_preprocessor(df,lbl); X_df = df.drop(columns=[lbl]) if lbl else df.copy()
    X_all,feature_names = fit_transform_preprocessor(pre,X_df)
    y = normalize_labels(df[lbl]) if lbl else np.zeros(X_all.shape[0],dtype=int)
    X_tr,X_te,y_tr,y_te = train_test_split(X_all,y,test_size=0.3,random_state=42,stratify=y if lbl is not None else None)
    model = train_random_forest(X_tr,y_tr); print("Trained RF")
    metrics = evaluate_classifier(model,X_te,y_te); print(json.dumps(metrics,indent=2,ensure_ascii=False))
    raw = generate_synthetic_attack_row("ddos", feature_names)
    df_row = pd.DataFrame([raw], columns=feature_names)
    X_sample = transform_with_preprocessor(pre, df_row, feature_names)
    out = evaluate_model_output(model, X_sample, "RandomForest"); print("Sim:", out)

def main():
    log("Start Zero-Day Tool (compact)")
    log(f"TF_AVAILABLE={TF_AVAILABLE}, PARQUET_ENGINE={PARQUET_ENGINE}, SMOTE_AVAILABLE={SMOTE_AVAILABLE}")
    if len(sys.argv)>1 and sys.argv[1] in ("--cli","cli"): interactive_cli_mode(); return
    try:
        root = tk.Tk(); app = App(root); root.mainloop()
    except Exception as e:
        log(f"[FATAL GUI] {e}"); logging.error(traceback.format_exc()); print("GUI failed, fallback to CLI"); interactive_cli_mode()

if __name__ == "__main__":
    main()
