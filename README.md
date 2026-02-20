# Motor Fault Diagnosis System: Machine Learning vs Fuzzy Logic

A comprehensive comparative study implementing dual AI approaches (Random Forest ML and Mamdani Fuzzy Logic) for fault detection and classification in High Voltage Induction Motors (HVIM).

**Project Status:** ✅ Complete — 100% ML accuracy, 99.88% Fuzzy accuracy on 800 simulated motor recordings

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [System Architecture](#system-architecture)
4. [Installation & Setup](#installation--setup)
5. [Usage Guide](#usage-guide)
6. [Results & Performance](#results--performance)
7. [Methodology](#methodology)
8. [Comparison: ML vs Fuzzy Logic](#comparison-ml-vs-fuzzy-logic)
9. [File Descriptions](#file-descriptions)
10. [Future Improvements](#future-improvements)
11. [References](#references)

---

## Overview

This project implements an intelligent fault diagnosis system for induction motors using two complementary AI techniques:

- **Machine Learning (Random Forest)**: 150-tree ensemble achieving 100% classification accuracy on 240 test samples
- **Fuzzy Logic (Mamdani FIS)**: 11-rule expert system achieving 99.88% accuracy on 800 samples with interpretable rules

### Faults Detected

| Fault Class | Signature | Impact |
|------------|-----------|--------|
| **Healthy** | Balanced 3-phase, rated speed | Normal operation |
| **Voltage Unbalance** | Phase voltage asymmetry | Current imbalance, speed drop, temperature rise |
| **Broken Rotor Bar** | Rotor cage damage | Torque ripple, slight current variation, reduced efficiency |
| **Stator Winding Fault** | Open circuit in stator phase | Extreme current unbalance, phase loss, rapid deterioration |

### Key Results
