# train_mieo_classifier.py
import os, json, time, random
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score, f1_score, average_precision_score, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score


# ====== CONFIG ======
DATA_DIR = "../preprocessed_10y0.1pct"
ART_DIR  = "/home/sbertucci/NEWres_mieo/mieo_pretrain0.1pct"                           
RUNS_DIR = "../NEWres_mieo&head"
RUN_NAME = "experiment21"
SEED     = 42

BATCH    = 1024
EPOCHS   = 100
LR       = 2e-3
WD       = 1e-4
PATIENCE = 12
# =====================

BINARY_COLS = []
bin_path = Path(DATA_DIR) / "binary_cols.txt"
if bin_path.exists():
    with open(bin_path, "r", encoding="utf-8") as f:
        BINARY_COLS = [ln.strip() for ln in f if ln.strip()]
else:
    BINARY_COLS = []


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#Funzione per la lettura delle risorse

def _read(base):
    p = Path(DATA_DIR) / f"{base}.parquet"
    if p.exists(): return pd.read_parquet(p)
    return pd.read_csv(Path(DATA_DIR) / f"{base}.csv")


#Funzione per preparare l'input

def make_xin(X, M):
    X2 = X.copy()

    if BINARY_COLS:
        present_bin_cols = [c for c in BINARY_COLS if c in X2.columns]
        if present_bin_cols:
            X2.loc[:, present_bin_cols] = X2.loc[:, present_bin_cols].fillna(2.0)

    
    Xz = X2.fillna(0.0).astype("float32").values

    Mm = M.astype("float32").values
    return torch.from_numpy(np.concatenate([Xz, Mm], axis=1))




class FrozenEncoder(nn.Module):
    def __init__(self, encoder_ckpt):
        super().__init__()
        obj = torch.load(encoder_ckpt, map_location="cpu")

        # --- meta ---
        in_dim  = int(obj.get("in_dim", 0) or obj.get("_meta_in_dim", [0])[0])
        latent  = int(obj.get("latent", 0) or obj.get("_meta_latent", [0])[0])

        #fallback
        state_raw = obj.get("encoder_state", obj.get("state_dict", obj))
        if (not in_dim or not latent) and isinstance(state_raw, dict):
            #normalizzo un minimo per leggere le shape
            example_key = next(iter(state_raw))
            #prendo tutte le chiavi con pesi 2D (Linear)
            linear_keys = [k for k,v in state_raw.items() if isinstance(v, torch.Tensor) and v.ndim == 2]
            if linear_keys:
                #prima e ultima Linear (per indice numerico se presente)
                def idx_of(k):
                    #estrae il primo intero che appare nella chiave
                    import re
                    m = re.search(r'(\d+)\.weight$', k)
                    return int(m.group(1)) if m else -1
                linear_keys_sorted = sorted(linear_keys, key=idx_of)
                first_w = state_raw[linear_keys_sorted[0]]
                last_w  = state_raw[linear_keys_sorted[-1]]
                if not in_dim:  in_dim  = first_w.shape[1] * 2  
                if not latent:  latent  = last_w.shape[0]

        if not in_dim or not latent:
            raise RuntimeError("Checkpoint privo di meta (in_dim/latent) e non inferibile dalle shape.")

        self.in_dim, self.latent = in_dim, latent

        
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.1), nn.Dropout(0.1),
            nn.Linear(1024, 512),    nn.BatchNorm1d(512),  nn.LeakyReLU(0.1), nn.Dropout(0.1),
            nn.Linear(512, 256),     nn.BatchNorm1d(256),  nn.LeakyReLU(0.1), nn.Dropout(0.1),
            nn.Linear(256, latent)
        )

        # --- state dict & strip prefissi ---
        state = state_raw
        if not isinstance(state, dict):
            raise RuntimeError("Formato checkpoint inatteso: manca encoder_state/state_dict")

        def strip_all_prefixes(d, prefixes=("encoder.", "enc.", "net.", "module.", "model.")):
            out = {}
            for k, v in d.items():
                kk = k
                changed = True
                #rimuoviamo prefissi noti finché ce ne sono
                while changed:
                    changed = False
                    for pref in prefixes:
                        if kk.startswith(pref):
                            kk = kk[len(pref):]
                            changed = True
                out[kk] = v
            return out

        state = strip_all_prefixes(state)

        # ora dovremmo avere chiavi tipo "0.weight", "1.weight", ...
        self.net.load_state_dict(state, strict=True)

        # freeze
        for p in self.parameters():
            p.requires_grad = False

        # full unfreeze: tutti i parametri sono addestrabili
        #for p in self.parameters():
        #   p.requires_grad = True


    def forward(self, xin): return self.net(xin)



#Classificatore

class Head(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(512, 256),    nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, z): return self.net(z).squeeze(1)


#Valutazione

@torch.no_grad()
def evaluate(model_e, model_h, loader):
    model_e.eval() 
    model_h.eval()
    logits, labels = [], []
    for xin, y in loader:
        xin = xin.to(DEVICE, non_blocking=True)
        y   = y.to(DEVICE, non_blocking=True)
        z = model_e(xin)
        l = model_h(z)
        logits.append(l.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
    logits = np.concatenate(logits); labels = np.concatenate(labels)
    probs = 1/(1+np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return {
        "bal_acc": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "auprc": average_precision_score(labels, probs),
        "auroc": roc_auc_score(labels, probs),
    }, probs, labels



def find_best_threshold(probs, labels, optimize="ba"):
    ths = np.linspace(0.0, 1.0, 1001)  
    best_t, best_score = 0.5, -1.0
    for t in ths:
        preds = (probs >= t).astype(int)
        if optimize == "macro_f1":
            score = f1_score(labels, preds, average="macro", zero_division=0)
        elif optimize == "ba":
            score = balanced_accuracy_score(labels, preds)
        else:
            raise ValueError("optimize must be 'macro_f1' or 'ba'")
        if score > best_score:
            best_score, best_t = score, t
    return best_t, best_score



def class_pos_weight(y):
    pos = (y==1).sum(); neg = (y==0).sum()
    return torch.tensor(neg/max(pos,1), dtype=torch.float32)




def main():
    seed_everything()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(RUNS_DIR) / f"{RUN_NAME}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # carica split
    X_tr, M_tr, y_tr = _read("X_train"), _read("M_train"), _read("y_event_train")["y_event"].astype("int8")
    X_va, M_va, y_va = _read("X_val"),   _read("M_val"),   _read("y_event_val")["y_event"].astype("int8")
    X_te, M_te, y_te = _read("X_test"),  _read("M_test"),  _read("y_event_test")["y_event"].astype("int8")

    # input per encoder
    Xin_tr, Xin_va, Xin_te = make_xin(X_tr, M_tr), make_xin(X_va, M_va), make_xin(X_te, M_te)

    # loader
    tr_ds = TensorDataset(Xin_tr, torch.from_numpy(y_tr.values.astype("float32")))
    va_ds = TensorDataset(Xin_va, torch.from_numpy(y_va.values.astype("float32")))
    te_ds = TensorDataset(Xin_te, torch.from_numpy(y_te.values.astype("float32")))
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,  pin_memory=torch.cuda.is_available())
    va_ld = DataLoader(va_ds, batch_size=BATCH, shuffle=False, pin_memory=torch.cuda.is_available())
    te_ld = DataLoader(te_ds, batch_size=BATCH, shuffle=False, pin_memory=torch.cuda.is_available())

    # encoder congelato
    enc = FrozenEncoder(Path(ART_DIR)/"encoder.pt").to(DEVICE)
    enc.eval()
    # --- check dimensioni encoder vs input ---
    print(f"[DEBUG] Xin_tr shape[1] = {Xin_tr.shape[1]}")
    print(f"[DEBUG] Xin_va shape[1] = {Xin_va.shape[1]}")
    print(f"[DEBUG] Xin_te shape[1] = {Xin_te.shape[1]}")
    print(f"[DEBUG] Encoder expects in_dim = {enc.in_dim}")

    if Xin_tr.shape[1] != enc.in_dim:
        raise RuntimeError(
            f"Mismatch dimensioni! Xin_tr ha {Xin_tr.shape[1]} colonne "
            f"ma l'encoder è stato pretrainato con in_dim={enc.in_dim}.\n"
        )
    head = Head(enc.latent).to(DEVICE)

    pos_w = class_pos_weight(y_tr).to(DEVICE)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)

    #opt   = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)

    best_val, best_state, patience = -1, None, PATIENCE
    best_t, best_val_metrics, best_epoch = 0.5, None, 0

    for epoch in range(1, EPOCHS+1):
        head.train()
        for xin, y in tr_ld:
            xin = xin.to(DEVICE, non_blocking=True)
            y   = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            #with torch.no_grad():
            z = enc(xin)
            logits = head(z)
            loss = crit(logits, y)
            loss.backward(); opt.step()

        val_metrics, val_probs, val_labels = evaluate(enc, head, va_ld)
        
        t_star, val_score = find_best_threshold(val_probs, val_labels, optimize="ba")

        # usa la metrica ottimizzata alla sua soglia
        if val_score > best_val:
            best_val = val_score
            best_state = {
                "encoder": {k: v.detach().cpu().clone() for k,v in enc.state_dict().items()},
                "head": {k: v.detach().cpu().clone() for k,v in head.state_dict().items()}
            }

            best_val_metrics = {
                **val_metrics,              
                "best_metric_on_val": float(val_score),
                "best_metric_name": "ba",
                "best_threshold_on_val": float(t_star)
            }
            best_epoch = epoch
            patience = PATIENCE
            best_t = float(t_star)
        else:
            patience -= 1
            if patience == 0:
                break

        print(f"Epoch {epoch:03d} | val@0.5 BA={val_metrics['bal_acc']:.3f} "
            f"val@0.5 F1={val_metrics['macro_f1']:.3f} | "
            f"best_ba@t*={val_score:.3f} (t*={t_star:.2f})")



    # TEST (soglia da validation)
    enc.load_state_dict(best_state["encoder"])
    head.load_state_dict(best_state["head"])
    test_metrics_raw, test_probs, test_labels = evaluate(enc, head, te_ld)
    thr = best_t
    preds = (test_probs>=thr).astype(int)
    test_bal_acc  = balanced_accuracy_score(test_labels, preds)
    test_macro_f1 = f1_score(test_labels, preds, average="macro", zero_division=0)
    test_precision = precision_score(test_labels, preds, average="macro", zero_division=0)
    test_recall    = recall_score(test_labels, preds, average="macro", zero_division=0)
    cm = confusion_matrix(test_labels, preds).tolist()

    print("\n=== TEST (MIEO+head, encoder frozen) ===")
    print(f"Balanced Acc: {test_bal_acc:.3f}  (soglia={thr:.2f})")
    print(f"Macro-F1:     {test_macro_f1:.3f}")
    print(f"AUPRC:        {test_metrics_raw['auprc']:.3f}")
    print(f"AUROC:        {test_metrics_raw['auroc']:.3f}")

    # Salvataggi
    torch.save({
        "encoder_ckpt": str(Path(ART_DIR)/"encoder.pt"),
        "head_state": best_state,
        "latent": enc.latent,
        "threshold_val": float(thr),
        "epoch_best": int(best_epoch)
    }, run_dir / "model_head.pt")

    with open(run_dir / "metrics.json", "w") as f:
        json.dump({
            "run_name": Path(run_dir).name,
            "data_dir": str(Path(DATA_DIR).resolve()),
            "encoder_ckpt": str(Path(ART_DIR)/"encoder.pt"),   # riferimento all’encoder usato
            "hparams": {
                "batch": BATCH,
                "epochs": EPOCHS,
                "lr": LR,
                "weight_decay": WD,
                "patience": PATIENCE,
                "seed": SEED
            },
            "val_metrics_best": {
                **best_val_metrics,
                "threshold_val": float(thr),
                "epoch_best": int(best_epoch)
            },
            "test_metrics": {
                "bal_acc_at_thr": float(test_bal_acc),
                "macro_f1_at_thr": float(test_macro_f1),
                "precision_at_thr": float(test_precision),
                "recall_at_thr": float(test_recall),
                "auprc": float(test_metrics_raw["auprc"]),
                "auroc": float(test_metrics_raw["auroc"]),
                "threshold_used": float(thr),
                "confusion_matrix": cm
            }
        }, f, indent=2)


if __name__ == "__main__":
    main()
