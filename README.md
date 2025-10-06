# Inlet Guide Vane (IGV) Performance Evaluation in Fluid Machinery


## 🌀 Overview

This repository contains Python scripts for analyzing the performance variation of a centrifugal fan
by adjusting the Inlet Guide Vane (IGV) angle during experiments.

The project focuses on understanding how IGV angle affects fan efficiency,
total pressure rise, and air flow characteristics through data-driven analysis.

The workflow allows calculation of efficiency curves and visualization of fan performance
based on experimental data collected at different vane angles.


## ⚙️ Features

Fan performance computation based on experimental data
Automatic efficiency calculation using total pressure and shaft power
Visualization of efficiency vs. flowrate (Q) curves
Comparison between multiple IGV angles
Easy-to-extend structure for other experimental data (e.g., RPM variation, pressure, etc.)


🧮 Experimental Background

In fluid machinery laboratories, performance tests are conducted by adjusting the inlet guide vane (IGV)
to control the flow direction and air volume entering the impeller.
This directly influences the static pressure, dynamic pressure, and overall fan efficiency.

During each test:

The guide vane angle is changed incrementally (e.g., 19°)

Measurements are taken for:

Air volume (m³/min)

Total pressure (mmAq)

Shaft power (kW)

The fan efficiency is then computed using:

𝜂
=
𝑄
×
Δ
𝑃
𝑡
𝑜
𝑡
𝑎
𝑙
𝑃
𝑠
ℎ
𝑎
𝑓
𝑡
×
100
η=
P
shaft
	​

Q×ΔP
total
	​

	​

×100

where

𝑄
Q = air flow rate (m³/s)

Δ
𝑃
𝑡
𝑜
𝑡
𝑎
𝑙
ΔP
total
	​

 = total pressure rise (Pa)

𝑃
𝑠
ℎ
𝑎
𝑓
𝑡
P
shaft
	​

 = shaft power input (W)

🧰 Code Description
Script: igv_efficiency_curve.py
