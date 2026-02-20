"""
FUZZY LOGIC MOTOR FAULT DIAGNOSIS
==================================
Mamdani FIS built from actual data distributions:

  Healthy:    unbal=0.002,  ib_rms=217.3,  wm_mean=148.5,  Te_mean=5267
  Unbalance:  unbal=17.2,   ib_rms=174.2,  wm_mean=147.8,  Te_mean=5267
  Rotor:      unbal=0.001,  ib_rms=218.0,  wm_mean=148.3,  Te_mean=5236
  Stator:     unbal=97.3,   ib_rms=0.0,    wm_mean=141.1,  Te_mean=5267

4 inputs, 1 output, 11 rules
"""

import numpy as np
import pandas as pd
import os
import glob

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
except ImportError:
    print("ERROR: Run 'pip install scikit-fuzzy' first!")
    exit(1)

CLASS_NAMES = ['Healthy', 'Voltage Unbalance', 'Rotor Fault', 'Stator Fault']


# ============================================================
# FUZZY SYSTEM DEFINITION
# ============================================================
def create_fuzzy_system():
    """
    Create Mamdani FIS with 4 inputs, 1 output, 11 rules.
    All membership function ranges derived from actual simulation data.
    """

    # === INPUT 1: Current Unbalance [0-100] ===
    # Healthy=0.002, Rotor=0.001, Unbalance=17.2, Stator=97.3
    unbal = ctrl.Antecedent(np.arange(0, 101, 0.1), 'current_unbalance')
    unbal['low']    = fuzz.trimf(unbal.universe, [0, 0, 8])
    unbal['medium'] = fuzz.trimf(unbal.universe, [5, 17, 35])
    unbal['high']   = fuzz.trimf(unbal.universe, [30, 100, 100])

    # === INPUT 2: Phase B RMS Current [0-230] ===
    # Stator=0, Unbalance=174, Healthy=217, Rotor=218
    ib = ctrl.Antecedent(np.arange(0, 231, 0.1), 'ib_rms')
    ib['zero']   = fuzz.trimf(ib.universe, [0, 0, 20])
    ib['low']    = fuzz.trimf(ib.universe, [100, 170, 200])
    ib['normal'] = fuzz.trimf(ib.universe, [195, 218, 230])

    # === INPUT 3: Mean Speed [138-150 rad/s] ===
    # Stator=141.1, Unbalance=147.8, Rotor=148.3, Healthy=148.5
    speed = ctrl.Antecedent(np.arange(138, 151, 0.01), 'wm_mean')
    speed['very_low'] = fuzz.trimf(speed.universe, [138, 141, 145])
    speed['low']      = fuzz.trimf(speed.universe, [143, 147, 148.5])
    speed['normal']   = fuzz.trimf(speed.universe, [147.5, 148.5, 150])

    # === INPUT 4: Mean Torque [5150-5350 Nm] ===
    # Rotor=5236, Healthy/Unbalance/Stator=5267
    # This is the KEY discriminator between Healthy and Rotor
    torque = ctrl.Antecedent(np.arange(5150, 5350, 0.1), 'Te_mean')
    torque['low']    = fuzz.trimf(torque.universe, [5150, 5230, 5252])
    torque['normal'] = fuzz.trimf(torque.universe, [5248, 5267, 5350])

    # === OUTPUT: Fault Type [0-3] ===
    fault = ctrl.Consequent(np.arange(0, 3.01, 0.01), 'fault_type')
    fault['healthy']   = fuzz.trimf(fault.universe, [0, 0, 0.75])
    fault['unbalance'] = fuzz.trimf(fault.universe, [0.5, 1, 1.5])
    fault['rotor']     = fuzz.trimf(fault.universe, [1.5, 2, 2.5])
    fault['stator']    = fuzz.trimf(fault.universe, [2.25, 3, 3])

    # === 11 FUZZY RULES ===
    rules = [
        # --- STATOR FAULT (Rules 1-4) ---
        # Phase B dead + extreme unbalance + speed drop
        ctrl.Rule(unbal['high'] & ib['zero'], fault['stator']),
        ctrl.Rule(unbal['high'] & speed['very_low'], fault['stator']),
        ctrl.Rule(ib['zero'] & speed['very_low'], fault['stator']),
        ctrl.Rule(unbal['high'], fault['stator']),

        # --- VOLTAGE UNBALANCE (Rules 5-7) ---
        # Medium unbalance + reduced Phase B + slight speed drop
        ctrl.Rule(unbal['medium'] & ib['low'], fault['unbalance']),
        ctrl.Rule(unbal['medium'] & speed['low'], fault['unbalance']),
        ctrl.Rule(unbal['medium'], fault['unbalance']),

        # --- BROKEN ROTOR BAR (Rules 8-9) ---
        # Normal currents but LOW torque (key physical indicator)
        ctrl.Rule(unbal['low'] & ib['normal'] & torque['low'], fault['rotor']),
        ctrl.Rule(unbal['low'] & speed['normal'] & torque['low'], fault['rotor']),

        # --- HEALTHY (Rules 10-11) ---
        # Everything normal
        ctrl.Rule(unbal['low'] & ib['normal'] & torque['normal'], fault['healthy']),
        ctrl.Rule(unbal['low'] & speed['normal'] & torque['normal'], fault['healthy']),
    ]

    system = ctrl.ControlSystem(rules)
    return system, unbal, ib, speed, torque, fault


def predict_fuzzy(system, current_unbalance, ib_rms, wm_mean, Te_mean):
    """
    Single prediction using fuzzy inference.
    Returns: (class_index, class_name, defuzzified_value)
    """
    sim = ctrl.ControlSystemSimulation(system)

    # Clip inputs to valid universe ranges
    sim.input['current_unbalance'] = np.clip(float(current_unbalance), 0.01, 99.99)
    sim.input['ib_rms']            = np.clip(float(ib_rms), 0.01, 229.99)
    sim.input['wm_mean']           = np.clip(float(wm_mean), 138.01, 149.99)
    sim.input['Te_mean']           = np.clip(float(Te_mean), 5150.1, 5349.9)

    try:
        sim.compute()
        value = sim.output['fault_type']

        if value < 0.5:
            cls = 0
        elif value < 1.5:
            cls = 1
        elif value < 2.5:
            cls = 2
        else:
            cls = 3

        return cls, CLASS_NAMES[cls], value

    except Exception as e:
        # If no rules fire, classify based on strongest single indicator
        if current_unbalance > 50:
            return 3, 'Stator Fault', 3.0
        elif current_unbalance > 8:
            return 1, 'Voltage Unbalance', 1.0
        elif Te_mean < 5250:
            return 2, 'Rotor Fault', 2.0
        else:
            return 0, 'Healthy', 0.0


def extract_fuzzy_inputs(filepath):
    """Extract the 4 fuzzy inputs from a raw CSV file."""
    df = pd.read_csv(filepath)
    if 't' in df.columns:
        df = df.drop(columns=['t'])

    ib = df['ib'].values.astype(float)
    ic = df['ic'].values.astype(float)
    Te = df['Te'].values.astype(float)
    wm = df['wm'].values.astype(float)

    ib_rms = np.sqrt(np.mean(ib**2))
    ic_rms = np.sqrt(np.mean(ic**2))
    current_unbalance = np.std([ib_rms, ic_rms])
    wm_mean = np.mean(wm)
    Te_mean = np.mean(Te)

    return current_unbalance, ib_rms, wm_mean, Te_mean


# ============================================================
# MAIN — Run validation when executed directly
# ============================================================
if __name__ == '__main__':
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    DATA_FOLDER = 'dataset'
    CLASS_MAP = {
        'healthy': 0,
        'voltage_unbalance': 1,
        'broken_rotor_bar': 2,
        'stator_fault': 3
    }

    # ========================================
    # BUILD THE FUZZY SYSTEM
    # ========================================
    print("=" * 60)
    print("FUZZY LOGIC MOTOR FAULT DIAGNOSIS")
    print("=" * 60)

    system, unbal, ib, speed, torque, fault = create_fuzzy_system()

    print(f"  Type:    Mamdani FIS")
    print(f"  Inputs:  4 (current_unbalance, ib_rms, wm_mean, Te_mean)")
    print(f"  Output:  1 (fault_type → 0=Healthy, 1=Unbalance, 2=Rotor, 3=Stator)")
    print(f"  Rules:   11")
    print(f"  Defuzz:  Centroid")

    # ========================================
    # PLOT MEMBERSHIP FUNCTIONS
    # ========================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    plot_data = [
        (unbal,  axes[0, 0], 'Input 1: Current Unbalance',
         {'Healthy': 0.002, 'Unbal': 17.2, 'Stator': 97.3}),
        (ib,     axes[0, 1], 'Input 2: Phase B RMS (A)',
         {'Stator': 0, 'Unbal': 174, 'Healthy': 217}),
        (speed,  axes[0, 2], 'Input 3: Mean Speed (rad/s)',
         {'Stator': 141.1, 'Unbal': 147.8, 'Healthy': 148.5}),
        (torque, axes[1, 0], 'Input 4: Mean Torque (Nm)',
         {'Rotor': 5236, 'Healthy': 5267}),
        (fault,  axes[1, 1], 'Output: Fault Type', {}),
    ]

    colors = {'low': 'green', 'medium': 'orange', 'high': 'red',
              'zero': 'red', 'normal': 'green',
              'very_low': 'darkred',
              'healthy': 'green', 'unbalance': 'orange',
              'rotor': 'purple', 'stator': 'red'}

    for var, ax, title, markers in plot_data:
        for term in var.terms:
            c = colors.get(term, 'blue')
            ax.plot(var.universe, var[term].mf, label=term.capitalize(),
                    linewidth=2, color=c)
        # Add data point markers
        for name, val in markers.items():
            ax.axvline(x=val, color='gray', linestyle='--', alpha=0.5)
            ax.text(val, 1.05, name, ha='center', fontsize=7, color='gray')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.3)

    # Rule summary in the empty subplot
    axes[1, 2].axis('off')
    rule_text = (
        "FUZZY RULES (11 total)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "STATOR FAULT (4 rules):\n"
        "  R1: IF unbal=High AND ib=Zero → Stator\n"
        "  R2: IF unbal=High AND speed=VeryLow → Stator\n"
        "  R3: IF ib=Zero AND speed=VeryLow → Stator\n"
        "  R4: IF unbal=High → Stator\n\n"
        "VOLTAGE UNBALANCE (3 rules):\n"
        "  R5: IF unbal=Medium AND ib=Low → Unbalance\n"
        "  R6: IF unbal=Medium AND speed=Low → Unbalance\n"
        "  R7: IF unbal=Medium → Unbalance\n\n"
        "BROKEN ROTOR BAR (2 rules):\n"
        "  R8: IF unbal=Low AND ib=Normal AND Te=Low → Rotor\n"
        "  R9: IF unbal=Low AND speed=Normal AND Te=Low → Rotor\n\n"
        "HEALTHY (2 rules):\n"
        "  R10: IF unbal=Low AND ib=Normal AND Te=Normal → Healthy\n"
        "  R11: IF unbal=Low AND speed=Normal AND Te=Normal → Healthy"
    )
    axes[1, 2].text(0.05, 0.95, rule_text, transform=axes[1, 2].transAxes,
                    fontsize=8, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle('Fuzzy Inference System — Membership Functions & Rules',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fuzzy_membership_functions.png', dpi=150)
    print("\n✅ Saved: fuzzy_membership_functions.png")

    # ========================================
    # QUICK TEST WITH CLASS CENTERS
    # ========================================
    print("\n" + "=" * 60)
    print("QUICK TEST — Known class center values")
    print("=" * 60)

    test_cases = [
        ("Healthy",           0.002, 217.3, 148.5, 5267),
        ("Voltage Unbalance", 17.2,  174.2, 147.8, 5267),
        ("Rotor Fault",       0.001, 218.0, 148.3, 5236),
        ("Stator Fault",      97.3,    0.0, 141.1, 5267),
    ]

    print(f"  {'Expected':<22} {'Predicted':<22} {'Defuzz':>8}")
    print(f"  {'─' * 55}")
    for expected, cu, ib_val, wm, te in test_cases:
        cls, name, val = predict_fuzzy(system, cu, ib_val, wm, te)
        match = "✅" if expected[:5] in name[:5] else "❌"
        print(f"  {match} {expected:<20} → {name:<20} {val:>8.3f}")

    # ========================================
    # FULL VALIDATION ON ALL 800 CSVs
    # ========================================
    print("\n" + "=" * 60)
    print("FULL VALIDATION — All 800 CSV files")
    print("=" * 60)

    all_true = []
    all_pred = []
    all_defuzz = []
    errors = []

    for folder_name, label in CLASS_MAP.items():
        folder_path = os.path.join(DATA_FOLDER, folder_name)
        if not os.path.exists(folder_path):
            print(f"  ⚠️  Folder not found: {folder_path}")
            continue

        csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))
        correct = 0
        total = len(csv_files)

        for f in csv_files:
            try:
                cu, ib_val, wm, te = extract_fuzzy_inputs(f)
                cls, name, val = predict_fuzzy(system, cu, ib_val, wm, te)
                all_true.append(label)
                all_pred.append(cls)
                all_defuzz.append(val)
                if cls == label:
                    correct += 1
            except Exception as e:
                errors.append((os.path.basename(f), str(e)))
                all_true.append(label)
                all_pred.append(-1)
                all_defuzz.append(-1)

        pct = correct / total * 100 if total > 0 else 0
        print(f"  {CLASS_NAMES[label]:25s}: {correct:3d}/{total:3d} = {pct:6.1f}%")

    # Overall results
    overall_acc = accuracy_score(all_true, all_pred) * 100

    print(f"\n  {'━' * 40}")
    print(f"  FUZZY LOGIC ACCURACY: {overall_acc:.2f}%")
    print(f"  {'━' * 40}")

    if errors:
        print(f"\n  ⚠️  Errors: {len(errors)}")
        for fname, err in errors[:5]:
            print(f"     {fname}: {err}")

    print(f"\n{classification_report(all_true, all_pred, target_names=CLASS_NAMES)}")

    # ========================================
    # CONFUSION MATRIX
    # ========================================
    cm = confusion_matrix(all_true, all_pred)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax2, annot_kws={'size': 14})
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('Actual', fontsize=12)
    ax2.set_title(f'Fuzzy Logic Confusion Matrix — Accuracy: {overall_acc:.2f}%',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fuzzy_results.png', dpi=150)
    print("✅ Saved: fuzzy_results.png")

    # ========================================
    # COMPARISON TABLE
    # ========================================
    print("\n" + "=" * 60)
    print("ML vs FUZZY LOGIC COMPARISON")
    print("=" * 60)
    print(f"  {'Method':<25} {'Accuracy':>10} {'Features':>10} {'Type':>15}")
    print(f"  {'─' * 62}")
    print(f"  {'Random Forest':<25} {'100.00%':>10} {'19':>10} {'Black-box':>15}")
    print(f"  {'Fuzzy Logic (Mamdani)':<25} {f'{overall_acc:.2f}%':>10} {'4':>10} {'Interpretable':>15}")
    print(f"\n  Fuzzy advantage: Human-readable rules, no training data needed")
    print(f"  ML advantage: Higher accuracy, handles complex patterns")
    print("=" * 60)

    plt.show()