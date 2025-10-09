import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import os, json, time, platform, random, hashlib
from pathlib import Path
from sklearn.metrics import precision_score, recall_score



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 1024
EPOCHS = 100
LR = 1e-3                                                   # learning rate
WD = 1e-4                                                   # weight decay
PATIENCE = 12                                               # early stopping
DATA_DIR = "../preprocessed_10y75pct"
RUNS_DIR = "../NEWres_ann"                         # dove salvare i risultati
RUN_NAME = "annOnly75pct"                          # nome run
SEED = 42                                          # seed di riproducibilità




def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def sha256sum(path, bufsize=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def manifest_dataset(data_dir):
    """Piccolo manifest dei file usati (hash/size) per un po' di tracciabilità."""
    p = Path(data_dir)
    names = [
        "X_train.parquet","X_val.parquet","X_test.parquet",
        "M_train.parquet","M_val.parquet","M_test.parquet",
        "y_event_train.parquet","y_event_val.parquet","y_event_test.parquet",
        "feature_cols.txt","binary_cols.txt","continuous_cols.txt","dropped_cols.txt",
    ]
    out = []
    for name in names:
        fp = p / name
        if fp.exists():
            out.append({"name": name, "size": fp.stat().st_size, "sha256": sha256sum(fp)})
    return out



#Funzione per caricare split degli input del modello (dati, maschera e label)

def load_split(split):

    if split not in {"train", "test", "val"}:
        raise ValueError(f"Split non valido: {split}. Ammessi: train, val, test.")

    X = pd.read_parquet(f"{DATA_DIR}/X_{split}.parquet")
    M = pd.read_parquet(f"{DATA_DIR}/M_{split}.parquet")
    y = pd.read_parquet(f"{DATA_DIR}/y_event_{split}.parquet")["y_event"].astype("int8")
    return X, M, y


#Funzioni per preparare gli input per pytorch

def make_tensor_inputs(X, M):
    
    Xz = X.fillna(0.0).astype("float32").values         # settiamo i null a 0.0
    
    Mm = M.astype("float32").values
    X_in = np.concatenate([Xz, Mm], axis=1)             # Concatenazione dati e maschera: [X_zero || Mask]
    return torch.from_numpy(X_in)                       # Tensore pytorch



def make_loader(X, M, y=None, shuffle=False):
    X_in = make_tensor_inputs(X, M)

    if y is None:
        ds = TensorDataset(X_in)
    else:
        y_t = torch.from_numpy(y.values.astype("float32"))
        ds = TensorDataset(X_in, y_t)
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, pin_memory=True)





#Multilayer Perceptron (ANN)

class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),                             #Layer fortemente connesso
            nn.BatchNorm1d(512),                                #Normalizzazione 
            nn.LeakyReLU(0.1),                                  #Funzione di attivazione LeakyReLU
            nn.Dropout(0.2),                                    #Disattivazione 20% dei neuroni (random) per evitare overfitting
            #Secondo layer
            nn.Linear(512, 256),                    
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(256, 1)  # logits (layer finale)
        )
    def forward(self, x): return self.net(x).squeeze(1)                 #Output da (batch_size, 1) --> (batch_size)






#Gestione sbilanciamento delle classi (ci serve per la BCEWithLogitsLoss)

def class_pos_weight(y):
    pos = (y==1).sum()
    neg = (y==0).sum()
    w = (neg / max(pos,1))
    return torch.tensor(w, dtype=torch.float32)





#Valutazione

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    logits, labels = [], []
    for batch in loader:
        x, y = (t.to(DEVICE, non_blocking=True) for t in batch)
        z = model(x)                                                        #Forward pass
        logits.append(z.cpu().numpy())
        labels.append(y.cpu().numpy())
    logits = np.concatenate(logits); labels = np.concatenate(labels)
    probs = 1/(1+np.exp(-logits))                                           # sigmoide (converte logits in probabilità) --> [0,1] rappresenta la confidenza per la classe positiva
    # soglia 0.5 (per stampare metriche grezze)
    preds = (probs >= 0.5).astype(int)
    return {                                                                # return di: 1. dizionario di metriche, le probabilità, le etichette vere
        "bal_acc": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "auprc": average_precision_score(labels, probs),
        "auroc": roc_auc_score(labels, probs),
    }, probs, labels





#Ricerca soglia migliore

def find_best_threshold(probs, labels, optimize="ba"):
    # candidati densi, includi estremi
    ths = np.linspace(0.0, 1.0, 1001)
    best_t, best_score = 0.5, -1.0
    labels = labels.astype(int)  # sicurezza
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
    return float(best_t), float(best_score)







#Main: addestramento del modello e valutazione
def main():

    seed_everything()

    #cartella run
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(RUNS_DIR) / f"{RUN_NAME}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)


    #Load dei dati
    X_tr, M_tr, y_tr = load_split("train")
    X_va, M_va, y_va = load_split("val")
    X_te, M_te, y_te = load_split("test")

    in_dim = X_tr.shape[1]*2    #[X || Mask]
    model = MLP(in_dim).to(DEVICE)

    train_loader = make_loader(X_tr, M_tr, y_tr, shuffle=True)
    val_loader   = make_loader(X_va, M_va, y_va, shuffle=False)
    test_loader  = make_loader(X_te, M_te, y_te, shuffle=False)



    #Loss con class imbalance
    pos_w = class_pos_weight(y_tr).to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)                                   # loss
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)             # ottimizzatore



    #Training + early stopping (val)
    best_val, best_state, patience = -1, None, PATIENCE
    best_t, best_val_metrics, best_epoch = 0.5, None, 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()


        #Validation
        val_metrics, val_probs, val_labels = evaluate(model, val_loader)

        #Scegliamo soglia e punteggio su validation ottimizzando la metrica target
        t_star, val_score = find_best_threshold(val_probs, val_labels, optimize="ba")  # oppure "macro-f1"

        #early stopping sulla metrica ottimizzata @t*
        if val_score > best_val:
            best_val = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = {
                **val_metrics,                     
                "best_metric_name": "ba",    
                "best_metric_on_val": float(val_score),
                "best_threshold_on_val": float(t_star)
            }
            best_epoch = epoch
            patience = PATIENCE
            best_t = float(t_star)                 
        else:
            patience -= 1
            if patience == 0:
                break

        print(f"Epoch {epoch:03d} | "
        f"val@0.5 BA={val_metrics['bal_acc']:.3f} "
        f"val@0.5 F1={val_metrics['macro_f1']:.3f} | "
        f"best_ba@t*={val_score:.3f} (t*={t_star:.2f})")





    #Test con best state & best threshold
    model.load_state_dict(best_state)
    test_metrics_raw, test_probs, test_labels = evaluate(model, test_loader)

    thr = best_t  #Usiamo la soglia trovata su validation
    test_preds = (test_probs >= thr).astype(int)
    test_bal_acc  = balanced_accuracy_score(test_labels, test_preds)
    test_macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    test_precision = precision_score(test_labels, test_preds, average="macro", zero_division=0)
    test_recall    = recall_score(test_labels, test_preds, average="macro", zero_division=0)
    test_cm = confusion_matrix(test_labels, test_preds).tolist()

    print("\n=== TEST ===")
    print(f"Balanced Acc: {test_bal_acc:.3f}  (soglia={thr:.2f})")
    print(f"Macro-F1:     {test_macro_f1:.3f}")
    print(f"AUPRC:        {test_metrics_raw['auprc']:.3f}")
    print(f"AUROC:        {test_metrics_raw['auroc']:.3f}")



    #Salvataggi

    #Pesi modello
    torch.save({
        "state_dict": best_state,
        "in_dim": in_dim,
        "architecture": [in_dim, 512, 256, 1],
        "threshold_val": float(thr),
        "epoch_best": int(best_epoch),
        "pos_weight": float(pos_w.detach().cpu().item()),
        "config": {"batch": BATCH, "epochs": EPOCHS, "lr": LR, "weight_decay": WD, "patience": PATIENCE, "seed": SEED}
    }, run_dir / "model.pt")

    #Report metrics.json
    report = {
        "run_name": run_dir.name,
        "data_dir": str(Path(DATA_DIR).resolve()),
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "device": DEVICE
        },
        "hparams": {"batch": BATCH, "epochs": EPOCHS, "lr": LR, "weight_decay": WD, "patience": PATIENCE},
        "val_metrics_best": {**best_val_metrics, "threshold_val": float(thr), "epoch_best": int(best_epoch)},
        "test_metrics": {
            "bal_acc_at_thr": float(test_bal_acc),
            "macro_f1_at_thr": float(test_macro_f1),
            "precision_at_thr": float(test_precision),
            "recall_at_thr": float(test_recall),
            "auprc": float(test_metrics_raw["auprc"]),
            "auroc": float(test_metrics_raw["auroc"]),
            "threshold_used": float(thr),
            "confusion_matrix": test_cm
        },
        "dataset_manifest": manifest_dataset(DATA_DIR)
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSalvato in: {run_dir.resolve()}")



if __name__ == "__main__":
    main()