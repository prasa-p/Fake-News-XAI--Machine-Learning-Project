# Data Download Instructions for Person 2

## Overview
You need to download 2 datasets and place them in the correct folders before you can start training DistilBERT.

---

## 📥 Dataset 1: Kaggle Fake and Real News

### Download Link
**Kaggle Dataset**: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

### Steps:
1. Go to the Kaggle link above
2. Click **"Download"** (you may need to sign in to Kaggle)
3. You'll get a ZIP file named something like `fake-and-real-news-dataset.zip`
4. Extract the ZIP file
5. You should see two CSV files:
   - `Fake.csv` (~23,000 fake news articles)
   - `True.csv` (~21,000 real news articles)

### Where to Place Them:
```
data/raw/kaggle/
├── Fake.csv    ← Put this file here
└── True.csv    ← Put this file here
```

### Verify:
```powershell
# Run this to check files are in the right place
ls data/raw/kaggle/
# Should show: Fake.csv and True.csv
```

---

## 📥 Dataset 2: LIAR Dataset

### Download Link
**LIAR Dataset**: https://github.com/thiagocastroferreira/LIAR-PLUS

### Alternative Links:
- Original paper: https://aclanthology.org/P17-2067/
- Direct download: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

### Steps:
1. Download the LIAR dataset ZIP file
2. Extract it
3. You should find 3 TSV files:
   - `train.tsv` (10,240 statements)
   - `valid.tsv` (1,284 statements)
   - `test.tsv` (1,266 statements)

### Where to Place Them:
```
data/raw/liar/
├── train.tsv    ← Put this file here
├── valid.tsv    ← Put this file here
└── test.tsv     ← Put this file here
```

### Verify:
```powershell
# Run this to check files are in the right place
ls data/raw/liar/
# Should show: train.tsv, valid.tsv, test.tsv
```

---

## ✅ Final Verification

Run this PowerShell command to verify all files are in place:

```powershell
# Check all required files exist
$kaggleFiles = @("data/raw/kaggle/Fake.csv", "data/raw/kaggle/True.csv")
$liarFiles = @("data/raw/liar/train.tsv", "data/raw/liar/valid.tsv", "data/raw/liar/test.tsv")
$allFiles = $kaggleFiles + $liarFiles

Write-Host "`nChecking raw data files..." -ForegroundColor Yellow
foreach ($file in $allFiles) {
    if (Test-Path $file) {
        Write-Host "✓ $file exists" -ForegroundColor Green
    } else {
        Write-Host "✗ $file MISSING" -ForegroundColor Red
    }
}

Write-Host "`nIf all files show ✓, you're ready to run preprocessing!" -ForegroundColor Cyan
```

---

## 🔄 Next Step: Preprocess the Data

Once all files are downloaded and in the correct folders, run:

```powershell
python src/data.py
```

This will:
- Clean the text (remove URLs, special characters, etc.)
- Split into train/val/test sets
- Save processed CSVs to `data/processed/`:
  - `kaggle_train.csv`, `kaggle_val.csv`, `kaggle_test.csv`
  - `liar_train.csv`, `liar_val.csv`, `liar_test.csv`

**Expected output:**
```
======================================================================
               DATA PREPROCESSING PIPELINE
          Person 1 (Prasa) - CP322 Group Project
======================================================================

### PROCESSING KAGGLE DATASET ###

Before cleaning: 44919 samples
After cleaning: ~43500 samples
✓ Saved kaggle splits to data/processed/

### PROCESSING LIAR DATASET ###

Before cleaning: 12791 samples  
After cleaning: ~12500 samples
✓ Saved liar splits to data/processed/

======================================================================
                    ✓ ALL DATASETS PROCESSED
======================================================================
```

---

## 🚨 Troubleshooting

### Error: "No such file or directory: data/raw/kaggle/Fake.csv"
**Solution**: You haven't downloaded the Kaggle dataset yet. Follow steps above.

### Error: "No such file or directory: data/raw/liar/train.tsv"
**Solution**: You haven't downloaded the LIAR dataset yet. Follow steps above.

### Files are there but preprocessing fails
**Solution**: Make sure the files are CSV/TSV format (not ZIP). Extract them first!

---

## 📊 Expected Dataset Sizes

After preprocessing, you should have approximately:

### Kaggle Dataset
- Train: ~35,000 samples
- Validation: ~4,000 samples
- Test: ~4,500 samples
- **Classes**: 0 = Fake, 1 = Real (balanced)

### LIAR Dataset
- Train: ~8,000 samples
- Validation: ~2,000 samples  
- Test: ~2,500 samples
- **Classes**: 0 = Fake, 1 = Real (converted from 6-class)

---

## ✨ Once Data is Ready

You can proceed to:
1. Install Python dependencies (`pip install -r requirements.txt`)
2. Run DistilBERT training (Person 2's main task)
3. Start tuning hyperparameters

See the main guide for Person 2's next steps!
