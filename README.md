# Inlet Guide Vane (IGV) Performance Evaluation in Fluid Machinery

---

## 🌀 Overview

- This repository contains Python scripts for analyzing the performance variation of a fan
by adjusting the Inlet Guide Vane (IGV) angle during experiment.

- The project focuses on understanding how IGV angle affects fan efficiency,
total pressure rise, and air flow characteristics through data-driven analysis.

- The workflow allows calculation of efficiency curves and visualization of fan performance
based on experimental data collected at different vane angles.

---

## ⚙️ Features

- Fan performance computation based on experimental data
- Automatic efficiency calculation using total pressure and shaft power
- Visualization of efficiency vs. flowrate (Q) curves
- Comparison between multiple IGV angles
- Easy-to-extend structure for other experimental data (e.g., RPM variation, pressure, etc.)

---

## 🧮 Experimental Background

In fluid machinery laboratories, performance tests are conducted by adjusting the inlet guide vane (IGV)
to control the flow direction and air volume entering the impeller.
This directly influences the static pressure, dynamic pressure, and overall fan efficiency.

During each test:

The guide vane angle is changed incrementally (e.g., 19°)

Measurements are taken for:
- Air volume (m³/min)
- Total pressure (mmAq)
- Shaft power (kW)
- The fan efficiency is then computed using:

𝜂=((𝑄×Δ𝑃total)/Pshaft)x100

where
- Q = air flow rate (m³/s)
- Δ𝑃𝑡𝑜𝑡𝑎𝑙 = total pressure rise (Pa)
- Pshaft = shaft power input (W)

---

## 🧰 Code Description
Script: igv_efficiency_curve.py
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Experimental Data (Example: IGV = 30°)
Q_m3_min = np.array([0, 945.9, 1337.0, 1829.3, 2110.7, 2449.6, 2664.7, 2746.4])
P_total_mmAq = np.array([127.1, 103.4, 94.2, 81.6, 69.0, 48.6, 34.7, 33.4])
P_shaft_kW = np.array([36.3, 35.7, 35.9, 36.2, 33.9, 30.3, 27.9, 27.8])

# Unit Conversion
Q_m3_s = Q_m3_min / 60
P_total_Pa = P_total_mmAq * 9.80665
P_shaft_W = P_shaft_kW * 1000

# Efficiency Calculation
eta = (Q_m3_s * P_total_Pa / P_shaft_W) * 100

# Display Results
df = pd.DataFrame({
    "Air Volume (m³/min)": Q_m3_min,
    "Total Pressure (mmAq)": P_total_mmAq,
    "Shaft Power (kW)": P_shaft_kW,
    "Efficiency (%)": np.round(eta, 2)
})
print(df)

# Plot Efficiency Curve
plt.figure(figsize=(7,4))
plt.plot(Q_m3_min, eta, 'o-', color='darkorange', lw=2, label='Computed Efficiency')
plt.xlabel("Air Volume (m³/min)")
plt.ylabel("Efficiency (%)")
plt.title("Fan Total Efficiency Curve (IGV Angle = 30°)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

```

---

## 📂 Project Directory
```
data/
├─ igv_angle_19.csv


src/
└─ igv_efficiency_curve.py

output/
├─ plots/
│  ├─ efficiency_curve_igv30.png
│  ├─ efficiency_comparison.png
└─ logs/
   └─ igv_experiment_log.txt

```

---

## 👤 Author

**Yongbeen Kim**

Intelligent Mechatronics Research Center, KETI

📧 Email: ybin521@keti.re.kr

📅 Last updated: 2025.10.08
