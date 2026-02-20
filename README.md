# High Voltage Induction Motor Fault Diagnosis: ML vs Fuzzy Logic

Comparative study of Random Forest (ML) and Mamdani Fuzzy Logic for fault classification in High Voltage Induction Motors using MATLAB Simulink simulation data.

**Results:** ML — 100% accuracy | Fuzzy Logic — 99.88% accuracy

---

## Fault Classes

| Class | Key Indicator |
|-------|--------------|
| Healthy | Balanced currents, rated speed |
| Voltage Unbalance | Current imbalance (~17), slight speed drop |
| Broken Rotor Bar | Reduced mean torque (5236 vs 5267 Nm) |
| Stator Fault | Phase B dead (ib=0), speed drops to 141 rad/s |

---

## Project Structure
├── inductionmain.slx # Simulink motor model
├── extract_features.m # MATLAB feature extraction (19 features)
├── retrain_from_csvs.py # Train Random Forest from dataset
├── fuzzy_diagnosis.py # Fuzzy Logic system (4 inputs, 11 rules)
├── dashboard.py # Streamlit web app (dual prediction)
├── dataset/ # 800 CSVs (200 per fault class)
│ ├── healthy/
│ ├── voltage_unbalance/
│ ├── broken_rotor_bar/
│ └── stator_fault/
├── ml_model.pkl # Trained RF model
├── features.pkl # Feature names list
├── extract_features.csv # Training dataset (800 × 20)
├── ml_results.png # ML confusion matrix
├── fuzzy_results.png # Fuzzy confusion matrix
└── fuzzy_membership_functions.png

text


---

## Quick Start

### Install Dependencies
```bash
pip install pandas numpy scikit-learn scikit-fuzzy joblib matplotlib seaborn plotly streamlit
Run Pipeline
Bash

# Step 1: Train ML model
python retrain_from_csvs.py

# Step 2: Validate fuzzy system
python fuzzy_diagnosis.py

# Step 3: Launch dashboard
streamlit run dashboard.py
Dashboard opens at http://localhost:8501 — upload any motor CSV to get dual ML + Fuzzy prediction.

Data Format
Each CSV: 10,000 samples at 10 kHz, 1 second duration

text

t,ia,ib,ic,Te,wm
0.0000,-0.034,0.000,0.063,0.001,147.788
0.0001,-0.045,0.000,-0.059,-0.021,147.729
ML vs Fuzzy Comparison
Aspect	Random Forest	Fuzzy Logic
Accuracy	100.00%	99.88%
Features	19	4
Training data needed	Yes (800 samples)	No
Interpretable	No (black-box)	Yes (11 rules)
Misclassified	0/240	1/800
Conclusion: ML wins on accuracy. Fuzzy wins on interpretability and simplicity. Running both simultaneously provides highest confidence — agreement between models confirms diagnosis.

Feature Extraction (19 Features)
Category	Features
Phase A Current	RMS, std, kurtosis, crest factor, dominant FFT bin
Cross-Phase	ib_rms, ic_rms, current_unbalance
Torque	mean, std, peak-to-peak
Speed	mean, std, peak-to-peak
Global	energy, FFT energy, torque ripple, speed ripple, THD-like
Fuzzy Logic Rules (11 Total)
text

Stator:    IF unbalance=High AND ib=Zero → Stator Fault
Unbalance: IF unbalance=Medium AND ib=Low → Voltage Unbalance
Rotor:     IF unbalance=Low AND ib=Normal AND torque=Low → Rotor Fault
Healthy:   IF unbalance=Low AND ib=Normal AND torque=Normal → Healthy
Limitations
Trained on simulation data only — real motor validation pending
Rotor fault has weakest separation from healthy (only torque differs by 31 Nm)
No incipient/gradual fault severity detection
No windowing applied to FFT analysis
Tech Stack
MATLAB/Simulink | Python | scikit-learn | scikit-fuzzy | Streamlit | Plotly

text


---

This is **concise, scannable, and covers everything** someone needs to understand and run your project. Want me to start on the report and PPT now?
