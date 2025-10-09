# pretrain_mieo.py
import os, json, time, random
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ====== CONFIG ======
DATA_DIR   = "../preprocessed_10y75pct"
RUNS_DIR   = "../NEWres_mieo"
RUN_NAME   = "mieo_pretrain75pct"
SEED       = 42

LATENT     = 128         # dimensione embedding
BATCH      = 1024
EPOCHS     = 100
LR         = 1e-3
WD         = 1e-4
PATIENCE   = 10
MASK_PROB  = 0.20        # masking artificiale
ALPHA_BIN  = 1.0         # peso BCE     (features binarie)
BETA_CONT  = 5.0         # peso MSE     (features continue)
# =====================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#Funzioni I/O Helper

def _read(base):
    p = Path(DATA_DIR) / f"{base}.parquet"
    if p.exists(): return pd.read_parquet(p)
    return pd.read_csv(Path(DATA_DIR) / f"{base}.csv")

def load_lists():
    bins = [l.strip() for l in (Path(DATA_DIR)/"binary_cols.txt").read_text().splitlines() if l.strip()]
    cont = [l.strip() for l in (Path(DATA_DIR)/"continuous_cols.txt").read_text().splitlines() if l.strip()]
    return bins, cont


#Classe per l'applicazione del masking artificiale al dataset

class TabAE(Dataset):
    """Ritorna (x_zero||m, m, idx_bin, idx_cont) e applica masking artificiale a runtime."""
    def __init__(self, X: pd.DataFrame, M: pd.DataFrame, idx_bin, idx_cont, mask_prob=0.2):
        assert (X.columns == M.columns).all()
        self.X = X.reset_index(drop=True)
        self.M = M.reset_index(drop=True)
        self.idx_bin = np.array(idx_bin, dtype=np.int64)
        self.idx_cont = np.array(idx_cont, dtype=np.int64)
        self.mask_prob = mask_prob

    def __len__(self): return len(self.X)           #Numero di righe di X

    def __getitem__(self, i):
        #Estraggo la riga i
        x = self.X.iloc[i].to_numpy(dtype=np.float32)       
        m = self.M.iloc[i].to_numpy(dtype=np.float32)

        #Imputazione zero per input
        x_in = x.copy()
        x_in[np.isnan(x_in)] = 0.0          #NaN --> 0.0

        # --- masking artificiale: spengo una quota di observed (=1) ---
        if self.mask_prob > 0:
            #Numero di righe
            num_righe = m.shape[0]

            #Genero vettore di valori casuali uniformi [0, 1)
            random_values = np.random.rand(num_righe)

            #Creo maschera booleana con probabilità self.mask_prob di essere True
            mask_bool = random_values < self.mask_prob

            #Converto in float32 (1.0 per True, 0.0 per False)
            mask_art = mask_bool.astype(np.float32)

            #Solo dove m==1 (osservato) si può mascherare artificialmente
            flip = (mask_art * m) > 0       #equivalente di un AND
            m = m.copy()
            m[flip] = 0.0
            #Azzero anche il valore corrispondente (visto che l'AE non deve "vederlo")
            x_in[flip] = 0.0

        xin = np.concatenate([x_in, m], axis=0)  # [X_zero || M]
        return torch.from_numpy(xin), torch.from_numpy(m), torch.from_numpy(x)



#Encoder

class Encoder(nn.Module):
    def __init__(self, in_dim, latent):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            nn.Linear(256, latent)   # embedding finale
        )
    def forward(self, x): return self.net(x)


#Decoder

class Decoder(nn.Module):
    def __init__(self, latent, out_bins, out_conts):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(latent, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),

            nn.Linear(1024, 512),  # ultimo step di ricostruzione
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
        )
        self.head_bin  = nn.Linear(512, out_bins)   # logits
        self.head_cont = nn.Linear(512, out_conts)  # valori

    def forward(self, z):
        h = self.shared(z)
        return self.head_bin(h), self.head_cont(h)



#Pipeline MIEO

class MIEO(nn.Module):
    def __init__(self, in_dim, n_bins, n_conts, latent=128):
        super().__init__()
        self.enc = Encoder(in_dim, latent)
        self.dec = Decoder(latent, n_bins, n_conts)
        self.n_bins, self.n_conts = n_bins, n_conts

    def forward(self, xin):
        z = self.enc(xin)
        logits_bin, cont = self.dec(z)
        return z, logits_bin, cont



#Valutazione

@torch.no_grad()
def eval_recon(model, loader, idx_bin, idx_cont):
    model.eval()
    bce_sum = 0.0; bce_den = 0.0
    mse_sum = 0.0; mse_den = 0.0
    loss_sum = 0.0; n_batches = 0
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for xin, m, x in loader:
        xin = xin.to(DEVICE); m = m.to(DEVICE); x = x.to(DEVICE)
        z, logits_bin, cont = model(xin)
        # split target e mask
        x_bin  = x[:, idx_bin]
        x_cont = x[:, idx_cont]
        m_bin  = m[:, idx_bin]
        m_cont = m[:, idx_cont]

        x_bin  = torch.nan_to_num(x_bin,  nan=0.0)
        x_cont = torch.nan_to_num(x_cont, nan=0.0)

        # losses mascherate
        lb = bce(logits_bin, x_bin) * m_bin
        lc = (cont - x_cont)**2 * m_cont
        bce_sum += lb.sum().item();  bce_den += m_bin.sum().item() + 1e-8
        mse_sum += lc.sum().item();  mse_den += m_cont.sum().item() + 1e-8
        loss_sum += (ALPHA_BIN * lb.sum().item() + BETA_CONT * lc.sum().item()) / (m_bin.sum().item()+m_cont.sum().item()+1e-8)
        n_batches += 1
    bce_avg = bce_sum / bce_den
    mse_avg = mse_sum / mse_den
    return bce_avg, mse_avg, loss_sum / n_batches





def main():
    seed_everything()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(RUNS_DIR) / f"{RUN_NAME}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    #Colonne bin/cont in ordine
    bin_cols, cont_cols = load_lists()
    n_bins, n_conts = len(bin_cols), len(cont_cols)

    #X/M supervised
    X_tr = _read("X_train");  M_tr = _read("M_train")
    X_va = _read("X_val");    M_va = _read("M_val")

    #Unlabelled se esistono (fallback a vuoto)
    try:
        X_un = _read("X_unlabelled"); M_un = _read("M_unlabelled")
        have_unlab = True
    except Exception:
        have_unlab = False

    #Concat train + unlabelled per pretraining
    if have_unlab:
        X_pt = pd.concat([X_tr, X_un], axis=0, ignore_index=True)
        M_pt = pd.concat([M_tr, M_un], axis=0, ignore_index=True)
    else:
        X_pt, M_pt = X_tr, M_tr

    #Index mapping: [bin || cont]
    

    # 1) pulisco le liste: mantengo solo colonne che esistono davvero in X
    all_cols = list(X_tr.columns)
    bin_cols  = [c for c in bin_cols  if c in all_cols]
    cont_cols = [c for c in cont_cols if c in all_cols and c not in bin_cols]

    # 2) dimensioni reali
    d_x    = X_tr.shape[1]
    n_bins = len(bin_cols)
    n_conts = d_x - n_bins  # tutto il resto sono continue

    # 3) indici posizionali
    idx_bin  = list(range(n_bins))
    idx_cont = list(range(n_bins, d_x))  # massimo = d_x-1

    # 4) crea dataset/loader
    train_ds = TabAE(X_pt, M_pt, idx_bin, idx_cont, mask_prob=MASK_PROB)
    val_ds   = TabAE(X_va, M_va, idx_bin, idx_cont, mask_prob=0.0)
    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  pin_memory=torch.cuda.is_available())
    val_ld   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, pin_memory=torch.cuda.is_available())

    # 5) dimensione input reale per il modello
    _sample_xin, _, _ = next(iter(train_ld))
    in_dim = _sample_xin.shape[1]  # deve essere 2 * d_x

    # 6) modello + opt
    model = MIEO(in_dim, n_bins, n_conts, latent=LATENT).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    bce   = nn.BCEWithLogitsLoss(reduction="none")

    #Check veloci
    assert d_x == (n_bins + n_conts), f"Incoerenza: d_x={d_x}, n_bins+n_conts={n_bins+n_conts}"
    assert max(idx_cont) == d_x-1,    f"max(idx_cont)={max(idx_cont)}, d_x={d_x}"
    assert in_dim == 2 * d_x,         f"in_dim={in_dim}, atteso {2*d_x}"


    train_ds = TabAE(X_pt, M_pt, idx_bin, idx_cont, mask_prob=MASK_PROB)
    val_ds   = TabAE(X_va, M_va, idx_bin, idx_cont, mask_prob=0.0)  # niente masking artificiale in val
    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  pin_memory=torch.cuda.is_available())
    val_ld   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, pin_memory=torch.cuda.is_available())

    # prendo un batch per inferire la vera dimensione di xin
    _sample_xin, _, _ = next(iter(train_ld))
    in_dim = _sample_xin.shape[1]  # dimensione reale di [X||M]
    model = MIEO(in_dim, n_bins, n_conts, latent=LATENT).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    best_val, best_state, patience = float("inf"), None, PATIENCE
    history = []

    for epoch in range(1, EPOCHS+1):
        model.train()
        run_loss = 0.0; nb = 0
        for xin, m, x in train_ld:
            xin = xin.to(DEVICE); m = m.to(DEVICE); x = x.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            z, logits_bin, cont = model(xin)
            # split target e mask
            x_bin  = x[:, idx_bin]
            x_cont = x[:, idx_cont]
            m_bin  = m[:, idx_bin]
            m_cont = m[:, idx_cont]

            x_bin  = torch.nan_to_num(x_bin,  nan=0.0)
            x_cont = torch.nan_to_num(x_cont, nan=0.0)

            # losses mascherate
            lb = bce(logits_bin, x_bin) * m_bin
            lc = (cont - x_cont)**2 * m_cont
            # normalizzo per #elementi validi
            lb_mean = lb.sum() / (m_bin.sum()  + 1e-8)
            lc_mean = lc.sum() / (m_cont.sum() + 1e-8)
            loss = ALPHA_BIN * lb_mean + BETA_CONT * lc_mean
            loss.backward()
            opt.step()
            run_loss += loss.item(); nb += 1

        # validation (ricostruzione)
        bce_v, mse_v, loss_v = eval_recon(model, val_ld, idx_bin, idx_cont)
        history.append({"epoch": epoch, "train_loss": run_loss/max(nb,1), "val_bce": bce_v, "val_mse": mse_v, "val_loss": loss_v})
        print(f"Epoch {epoch:03d} | train={run_loss/max(nb,1):.4f} | val_loss={loss_v:.4f} (bce={bce_v:.4f}, mse={mse_v:.4f})")

        # early stopping su val_loss
        if loss_v < best_val - 1e-6:
            best_val = loss_v
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = PATIENCE
        else:
            patience -= 1
            if patience == 0:
                break

    # salvo l'encoder
    (Path(RUNS_DIR)/"mieo_pretrain75pct").mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_state": {k.replace("enc.", "", 1): v for k, v in best_state.items() if k.startswith("enc.")},
        "in_dim": in_dim, "latent": LATENT,
        "n_bins": n_bins, "n_conts": n_conts,
        "bin_cols": bin_cols, "cont_cols": cont_cols
    }, Path(RUNS_DIR)/"mieo_pretrain75pct"/"encoder.pt")

    with open(Path(RUNS_DIR)/"mieo_pretrain75pct"/"pretrain_metrics.json", "w") as f:
        json.dump({
            "run_dir": str(Path(RUNS_DIR)/f"{RUN_NAME}_{time.strftime('%Y%m%d-%H%M%S')}"),
            "hparams": {
                "latent": LATENT,
                "mask_prob": MASK_PROB,
                "alpha_bin": ALPHA_BIN,
                "beta_cont": BETA_CONT,
                "batch": BATCH,
                "epochs": EPOCHS,
                "lr": LR,
                "weight_decay": WD,
                "patience": PATIENCE
            },
            "history": history,
            "best_val_loss": best_val
        }, f, indent=2)


if __name__ == "__main__":
    main()
