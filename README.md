# MIEO — Masked Input Encoded Output

Questo repository contiene il codice per **MIEO (Masked Input Encoded Output)**,  
un'architettura per la **rappresentazione e classificazione di dati clinici** con valori mancanti,  
basato su un **autoencoder** pre-addestrato e un classificatore supervisionato.

L’obiettivo è fornire una pipeline modulare per:
1. **Preprocessare** i dati clinici eterogenei in formato uniforme,
2. **Pre-addestrare** un encoder MIEO capace di gestire il missingness strutturale,
3. **Allenare** un classificatore supervisionato (ANN o MIEO+Head) per la predizione di eventi clinici.

---

## Struttura del progetto

### `preprocessing.py`
Script che esegue il **preprocessing completo del dataset**:
- Rimozione di colonne spurie, feature di leakage e con troppi NaN  
- Conversione automatica di colonne object in numeriche  
- Identificazione e separazione tra feature **binarie** e **continue**  
- Costruzione della **null-mask** (1 = valore presente, 0 = NaN)  
- Train/validation/test split stratificato (60/20/20)  
- Scaling delle feature continue  
- Esportazione di file `.parquet` e metadati (`binary_cols.txt`, `continuous_cols.txt`, ecc.)


---

### `pretraining_mieo.py`
Script per il **pretraining non supervisionato** del modello MIEO:
- Applica **masking artificiale** alle feature osservate per insegnare al modello la ricostruzione dei dati mancanti  
- L’architettura è composta da un **Encoder** (fully connected) e da un **Decoder** con due teste:
  - `BCE` per le feature binarie
  - `MSE` per le feature continue  
- La perdita complessiva è una combinazione pesata di entrambe (`α·BCE + β·MSE`)  
- Al termine salva solo l’**encoder** congelato (`encoder.pt`), utilizzabile per downstream tasks.


---

### `train_mieo_classifier.py`
Script per il **training supervisionato** del classificatore MIEO+Head:
- Carica l’encoder pre-addestrato (frozen)
- Costruisce gli input come `[X || Mask]`
- Addestra una **testa di classificazione (Head)** multilayer (512–256–1)  
- Ottimizza la `Balanced Accuracy` su validation con early stopping  
- Valuta su test e salva i risultati (`metrics.json` + confusion matrix)

---

### `Classifier.py`
Implementa il **classificatore baseline** (ANN standalone):
- Input diretto `[X || Mask]` senza encoder MIEO  
- Architettura MLP con due hidden layers (512, 256)  
- Training supervisionato con `BCEWithLogitsLoss` e pesi bilanciati  
- Valutazione su test set con `Balanced Accuracy`, `F1`, `AUPRC`, `AUROC`  
- Genera un report `metrics.json` e salva i pesi (`model.pt`)

Utilizzato per confrontare le prestazioni con la versione MIEO+Head.

---

### `datasetExploration.ipynb`
Notebook di **analisi esplorativa** del dataset grezzo:
- Classifica automaticamente le feature per tipo (binarie, continue, categoriche, temporali, testuali)  
- Analizza la percentuale di valori nulli e la cardinalità delle variabili  
- Evidenzia correlazioni forti tra variabili binarie  

È pensato come strumento esplorativo preliminare per la fase di preprocessing.


