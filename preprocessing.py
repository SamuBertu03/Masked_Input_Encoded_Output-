#!/usr/bin/env python3
"""
Preprocessing MIEO:
- Drops dei leakage/outcome/date features
- Drops delle features con missingness > threshold (default 0.70)
- Conversione oggetti simil-numerici in floats
- Costruzione null-mask (1=presente, 0=NaN)
- Splits supervised (label_10 in {0,1}) vs unlabelled (NaN)
- Train/Val/Test split: 60/20/20 
- Scale delle features continue
- Salvataggio parquet files + metadata lists + scaler
"""

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample



FORCE_DROP = {"Vmonth_0"}       #PATCH: Vmonth_0 è una feature baseline, dunque la rimuoviamo


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess dataset for 10y classification.")
    p.add_argument("--input", type=str, required=True, help="Path to input CSV (raw).")
    p.add_argument("--out-dir", type=str, default="./preprocessed_10y", help="Output directory.")
    p.add_argument("--label", type=str, default="label_10", help="Name of the label column (0/1/NaN).")
    p.add_argument("--missing-thr", type=float, default=0.70, help="Drop features with missing fraction > this threshold.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--test-size", type=float, default=0.20, help="Test size fraction (default 0.20).")
    p.add_argument("--val-size", type=float, default=0.25, help="Validation fraction *relative to remaining* after test (default 0.25 => 20%% of total).")
    p.add_argument("--labeled-fraction", type=float, default=1.0, help="Frazione di pazienti labeled da mantenere supervisionati (bilanciati). Default=1.0 = tutti.")
    return p.parse_args()


#Funzione per controllare se sono presenti altre colonne spurie

def detect_spurious_index_cols(df: pd.DataFrame):
    """
    Identifica colonne indice salvate nel CSV che vanno droppate:
    '0', 'Unnamed: 0', 'index', 'level_0' quando sono contatori 0..N-1 o 1..N (o tutti NaN).
    """
    candidates = [c for c in df.columns if str(c) in {"0", "Unnamed: 0", "index", "level_0"}]
    to_drop = []
    for c in candidates:
        s = pd.to_numeric(df[c], errors="coerce")
        n = len(s)
        if s.isna().all() or s.equals(pd.Series(range(n))) or s.equals(pd.Series(range(1, n + 1))):
            to_drop.append(c)
    return to_drop



#Funzione per la conversione numerica di alcune features Obj

def numeric_like_conversion(df, label_col, drop_cols):
    """
    Converte in float le colonne object "numeric-like".
    Euris.: sostituisce ',' con '.' e fa to_numeric; accetta la conversione
    se >=95%% dei valori non-null diventano numerici.
    """
    for c in df.columns:
        if c == label_col or c in drop_cols:
            continue
        if df[c].dtype == "object":
            s = df[c].astype(str)
            s2 = s.str.replace(",", ".", regex=False)
            num = pd.to_numeric(s2, errors="coerce")
            before = df[c].notna().sum()
            after = num.notna().sum()
            if before > 0 and (after / before) >= 0.95:
                df[c] = num
    return df


#Funzione per separare features binare e continue

def identify_binary_continuous(df, feature_cols):
    bin_cols, cont_cols = [], []
    for c in feature_cols:
        s = df[c].dropna()
        if s.empty:
            cont_cols.append(c)
            continue
        uniq = pd.unique(s)
        if set(np.unique(uniq)).issubset({0, 1, 0.0, 1.0, True, False}):
            bin_cols.append(c)
        else:
            cont_cols.append(c)
    return bin_cols, cont_cols




def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #Load del CSV

    df = pd.read_csv(args.input, low_memory=False)
    if args.label not in df.columns:
        raise ValueError(f"Label column '{args.label}' not found in the dataset.")

    
    #Drop colonne indice spurie + drop forzato
    
    spurious = detect_spurious_index_cols(df)
    if spurious:
        print(f"[INFO] Droppo colonne indice spurie: {spurious}")
        df = df.drop(columns=spurious)

    force_present = sorted([c for c in FORCE_DROP if c in df.columns])
    if force_present:
        print(f"[INFO] Droppo colonne forzate: {force_present}")
        df = df.drop(columns=force_present)


    #Drop per categoria
    LEAKAGE_COLS = ["Follow_up_time", "Time_to_death", "Lost_FU", "Death_date", "Cause_of_death", "Vdate_0"]
    FLAG_SPARSE = ["K_urine_flag_0", "Na_urine_flag_0", "Creat_urine_flag_0"]
    TEXT_SERIES = ["PW_time_0"]

    #Drop delle features con missingness > threshold (except the label)
    hi_missing = [c for c in df.columns if c != args.label and df[c].isna().mean() > args.missing_thr]

    drop_cols = sorted(set([c for c in (LEAKAGE_COLS + FLAG_SPARSE + TEXT_SERIES + hi_missing) if c in df.columns]))

    #Conversione features object in float
    df = numeric_like_conversion(df, args.label, drop_cols)

    #Rimozione di ogni feature non-numerica rimasta (safety)
    non_numeric = [c for c in df.columns if c not in drop_cols + [args.label] and not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        print(f"[WARN] Dropping leftover non-numeric columns (not convertible): {len(non_numeric)}")
        drop_cols = sorted(set(drop_cols + non_numeric))

    #Lista di features finale
    drop_cols.append("Elapsed_time_visit0_death0")
    feature_cols = [c for c in df.columns if c not in drop_cols + [args.label]]
    if not feature_cols:
        raise RuntimeError("No features left after dropping. Check thresholds and lists.")

    #Separazione e riordinamento Binarie | Continue
    bin_cols, cont_cols = identify_binary_continuous(df, feature_cols)
    feature_cols_ordered = bin_cols + cont_cols
    df_feat = df[feature_cols_ordered].copy()

    #Costruzione null-mask (1=presente, 0=NaN)
    mask = df_feat.notna().astype("int8")

    #Supervised vs Unlabelled split
    sup_idx = df[args.label].isin([0, 1])
    X_sup = df_feat.loc[sup_idx].copy()
    M_sup = mask.loc[sup_idx].copy()
    y_sup = df.loc[sup_idx, args.label].astype("int8")
    #Event=1 (deceduto) se 0=deceased, 1=survived
    y_event_sup = (1 - y_sup).astype("int8")

    X_unlab = df_feat.loc[~sup_idx].copy()
    M_unlab = mask.loc[~sup_idx].copy()


    if args.labeled_fraction < 1.0:
        n_target = int(len(X_sup) * args.labeled_fraction)
        n_per_class = n_target // 2

        idx_0 = y_sup[y_sup == 0].index
        idx_1 = y_sup[y_sup == 1].index

        idx_sup_0 = resample(idx_0, n_samples=min(n_per_class, len(idx_0)), random_state=args.seed, replace=False)
        idx_sup_1 = resample(idx_1, n_samples=min(n_per_class, len(idx_1)), random_state=args.seed, replace=False)
        idx_sup = idx_sup_0.append(idx_sup_1)

        # differenza rispetto a tutti i supervised originali
        idx_to_unlab = y_sup.index.difference(idx_sup)

        # supervised ridotti
        X_sup_red = X_sup.loc[idx_sup]
        M_sup_red = M_sup.loc[idx_sup]
        y_sup_red = y_sup.loc[idx_sup]
        y_event_sup_red = y_event_sup.loc[idx_sup]

        # quelli da spostare in unlabeled
        X_unlab_extra = X_sup.loc[idx_to_unlab]
        M_unlab_extra = M_sup.loc[idx_to_unlab]

        # aggiorna i dataset
        X_sup, M_sup, y_sup, y_event_sup = X_sup_red, M_sup_red, y_sup_red, y_event_sup_red
        X_unlab = pd.concat([X_unlab, X_unlab_extra], axis=0)
        M_unlab = pd.concat([M_unlab, M_unlab_extra], axis=0)




    #Train/Val/Test split (60/20/20)
    X_tmp, X_test, M_tmp, M_test, y_tmp, y_test, y_ev_tmp, y_ev_test = train_test_split(
        X_sup, M_sup, y_sup, y_event_sup, test_size=args.test_size, random_state=args.seed, stratify=y_sup
    )
    val_rel = args.val_size
    X_train, X_val, M_train, M_val, y_train, y_val, y_ev_train, y_ev_val = train_test_split(
        X_tmp, M_tmp, y_tmp, y_ev_tmp, test_size=val_rel, random_state=args.seed, stratify=y_tmp
    )

    #Scaliamo le continue
    scaler = None
    if len(cont_cols) > 0:
        scaler = StandardScaler()
        scaler.fit(X_train[cont_cols])
        for X in (X_train, X_val, X_test, X_unlab):
            X.loc[:, cont_cols] = scaler.transform(X[cont_cols])

    #Salviamo gli output
    def save_parquet(d, name):
        d.to_parquet(out_dir / f"{name}.parquet")

    save_parquet(X_train, "X_train")
    save_parquet(X_val, "X_val")
    save_parquet(X_test, "X_test")
    save_parquet(M_train, "M_train")
    save_parquet(M_val, "M_val")
    save_parquet(M_test, "M_test")
    pd.DataFrame({"y": y_train}).to_parquet(out_dir / "y_train.parquet")
    pd.DataFrame({"y": y_val}).to_parquet(out_dir / "y_val.parquet")
    pd.DataFrame({"y": y_test}).to_parquet(out_dir / "y_test.parquet")

    #event=1 labels
    pd.DataFrame({"y_event": y_ev_train}).to_parquet(out_dir / "y_event_train.parquet")
    pd.DataFrame({"y_event": y_ev_val}).to_parquet(out_dir / "y_event_val.parquet")
    pd.DataFrame({"y_event": y_ev_test}).to_parquet(out_dir / "y_event_test.parquet")

    #Unlabelled per MIEO
    if len(X_unlab) > 0:
        save_parquet(X_unlab, "X_unlabelled")
        save_parquet(M_unlab, "M_unlabelled")

    #Metadata
    pd.Series(feature_cols_ordered).to_csv(out_dir / "feature_cols.txt", index=False, header=False)
    pd.Series(bin_cols).to_csv(out_dir / "binary_cols.txt", index=False, header=False)
    pd.Series(cont_cols).to_csv(out_dir / "continuous_cols.txt", index=False, header=False)
    pd.Series(drop_cols).to_csv(out_dir / "dropped_cols.txt", index=False, header=False)

    #Salviamo lo scaler
    if scaler is not None:
        joblib.dump({"scaler": scaler, "continuous_cols": cont_cols}, out_dir / "scaler_std.pkl")

    #Riassunto JSON (includendo i drop manuali)
    summary = {
        "input_csv": str(Path(args.input).resolve()),
        "out_dir": str(out_dir.resolve()),
        "label_col": args.label,
        "missing_drop_threshold": args.missing_thr,
        "n_rows_total": int(len(df)),
        "n_features_final": int(len(feature_cols_ordered)),
        "n_binary": int(len(bin_cols)),
        "n_continuous": int(len(cont_cols)),
        "n_supervised": int(len(X_sup)),
        "n_unlabelled": int(len(X_unlab)),
        "split_sizes": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test))
        },
        "dropped": {
            "manual": spurious + force_present,  
            "leakage_cols_found": [c for c in LEAKAGE_COLS if c in df.columns],
            "sparse_flags_found": [c for c in FLAG_SPARSE if c in df.columns],
            "text_series_found": [c for c in TEXT_SERIES if c in df.columns],
            "hi_missing": hi_missing,
            "non_numeric": non_numeric if 'non_numeric' in locals() else []
        }
    }
    with open(out_dir / "preprocess_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    #CLI riassunto
    print("=== Preprocessing done ===")
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Features: total={len(feature_cols_ordered)} | binary={len(bin_cols)} | continuous={len(cont_cols)}")
    print(f"Supervised: train={len(X_train)} | val={len(X_val)} | test={len(X_test)} | Unlabelled={len(X_unlab)}")
    print(f"Dropped columns written to: {out_dir/'dropped_cols.txt'}")

    #Controllo finale: nessuna colonna vietata nelle feature
    forbidden = {"0", "Vmonth_0", "Unnamed: 0", "index", "level_0"}
    assert not (set(feature_cols_ordered) & forbidden), "Colonne vietate ancora presenti nelle feature!"


if __name__ == "__main__":
    main()
