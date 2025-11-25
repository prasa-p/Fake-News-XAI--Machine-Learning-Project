# Person 2 (Zaid) - DistilBERT Training Log

**Project**: CP322 Fake News Detection with Explainable AI  
**Role**: DistilBERT Training & Hyperparameter Tuning  
**Date Started**: November 24, 2025

---

## Objectives

- [ ] Train DistilBERT on Kaggle Fake News dataset
- [ ] Train DistilBERT on LIAR dataset
- [ ] Beat baseline performance from Person 1
- [ ] Tune hyperparameters for optimal F1 score
- [ ] Save trained models for Person 3-5

---

## Baseline Performance (Person 1)

### Kaggle Dataset
- **Best Baseline**: TF-IDF + Logistic Regression
- **F1 Score**: (Check `artifacts/metrics/baseline_results.csv`)
- **Target**: Beat this with DistilBERT

### LIAR Dataset
- **Best Baseline**: TF-IDF + Logistic Regression
- **F1 Score**: (Check `artifacts/metrics/baseline_results.csv`)
- **Target**: Beat this with DistilBERT

---

## Kaggle Dataset Training Experiments

### Experiment 1: Baseline Configuration
**Date**: YYYY-MM-DD  
**Config**: `config/kaggle.yaml` (initial)
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Epochs**: 3
- **Max Length**: 512
- **Weight Decay**: 0.01
- **Warmup Steps**: 500

**Results**:
- Train Accuracy: _____
- Val Accuracy: _____
- **Test Accuracy**: _____
- **Test F1 Score**: _____
- Test Precision: _____
- Test Recall: _____

**Notes**:
- Training time: _____ hours
- GPU/CPU used: _____
- Observations: _____

**Next Steps**:
- [ ] Try higher learning rate?
- [ ] Increase epochs?
- [ ] Adjust batch size?

---

### Experiment 2: [Describe change]
**Date**: YYYY-MM-DD  
**Config**: Modified learning rate
- **Learning Rate**: 3e-5 ← CHANGED
- **Batch Size**: 16
- **Epochs**: 3
- **Max Length**: 512

**Results**:
- **Test Accuracy**: _____
- **Test F1 Score**: _____

**Comparison to Experiment 1**:
- F1 improvement: +____%

**Notes**:
- _____

---

### Experiment 3: [Describe change]
**Date**: YYYY-MM-DD  
**Changes**: _____

**Results**:
- **Test F1 Score**: _____

**Notes**:
- _____

---

### Best Kaggle Configuration
**Final Choice**: Experiment #___  
**Config File**: `config/kaggle_final.yaml` (if created)

**Hyperparameters**:
- Learning Rate: _____
- Batch Size: _____
- Epochs: _____
- Max Length: _____
- Warmup Steps: _____
- Weight Decay: _____

**Final Results**:
- **Test Accuracy**: _____
- **Test F1 Score**: _____
- **Test Precision**: _____
- **Test Recall**: _____

**Improvement over Baseline**:
- Baseline F1: _____
- DistilBERT F1: _____
- **Improvement**: +_____%

**Model Location**: `artifacts/distilbert/kaggle/final_model/`

---

## LIAR Dataset Training Experiments

### Experiment 1: Baseline Configuration
**Date**: YYYY-MM-DD  
**Config**: `config/liar.yaml` (initial)
- **Learning Rate**: 3e-5
- **Batch Size**: 32
- **Epochs**: 5
- **Max Length**: 128
- **Weight Decay**: 0.01
- **Warmup Steps**: 200

**Results**:
- **Test Accuracy**: _____
- **Test F1 Score**: _____
- **Test Precision**: _____
- **Test Recall**: _____

**Notes**:
- Training time: _____ minutes
- Observations: _____

---

### Experiment 2: [Describe change]
**Date**: YYYY-MM-DD  
**Changes**: _____

**Results**:
- **Test F1 Score**: _____

**Notes**:
- _____

---

### Best LIAR Configuration
**Final Choice**: Experiment #___

**Hyperparameters**:
- Learning Rate: _____
- Batch Size: _____
- Epochs: _____

**Final Results**:
- **Test F1 Score**: _____

**Improvement over Baseline**: +_____%

**Model Location**: `artifacts/distilbert/liar/final_model/`

---

## Challenges Encountered

### Challenge 1: [Description]
**Problem**: _____  
**Solution**: _____  
**Outcome**: _____

### Challenge 2: [Description]
**Problem**: _____  
**Solution**: _____

---

## Key Learnings

1. _____
2. _____
3. _____

---

## Deliverables Checklist

- [ ] Kaggle model trained and saved to `artifacts/distilbert/kaggle/final_model/`
- [ ] LIAR model trained and saved to `artifacts/distilbert/liar/final_model/`
- [ ] Both models beat baseline F1 scores
- [ ] Metrics saved to `artifacts/metrics/distilbert_*_results.json`
- [ ] Model loading tested with `scripts/test_model_loading.py`
- [ ] Training configurations documented in `config/kaggle.yaml` and `config/liar.yaml`
- [ ] This training log completed with all experiments
- [ ] Code committed to branch `zaid-distilbert-training`
- [ ] Pull Request created on GitHub

---

## Final Summary

**Kaggle Dataset**:
- Baseline F1: _____
- DistilBERT F1: _____
- **Improvement**: +_____%
- Total training time: _____ hours

**LIAR Dataset**:
- Baseline F1: _____
- DistilBERT F1: _____
- **Improvement**: +_____%
- Total training time: _____ hours

**Total Experiments Run**: _____ (Kaggle) + _____ (LIAR) = _____

**Models Ready for Team**:
-  Kaggle model ready for Person 3-5
-  LIAR model ready for Person 3-5

**Next Steps for Team**:
- Person 3 (Ryan) can start LIME/SHAP explainability
- Person 4 (Vidya) can start explanation evaluation
- Person 5 (Anton) can start Integrated Gradients and demo

---

## Acknowledgments

- Person 1 (Prasa) for data preprocessing and baselines
- Anton for initial DistilBERT pipeline
- Team for support and collaboration

---

**Completed by**: Zaid (Person 2)  
**Date Completed**: YYYY-MM-DD
