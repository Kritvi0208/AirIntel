# Atmospheric Physics & Data Engineering Protocol

AirIntel models air quality by combining **thermodynamic boundary layer physics, meteorological washout dynamics, and advanced feature selection**.

---

## 1. Atmospheric Physics Principles

### A. Planetary Boundary Layer (PBL) & Winter Inversions
* **Summer Convective Mixing**:
    * Strong solar heating elevates the Planetary Boundary Layer (PBL) height to **2,000 m – 3,000 m**.
    * Turbulence rapidly dilutes and disperses ground-level emissions into a large atmospheric column.
* **Winter Ground Inversion (November – January)**:
    * Rapid radiational cooling of the earth creates a cold, dense air layer trapped under warmer air aloft.
    * The boundary layer collapses to as low as **150 m – 300 m** (a **~10× vertical volume compression**).
    * Trapped emissions cause an immediate **3.2× surge in ambient PM2.5 and PM10 concentrations** across the landlocked Indo-Gangetic Basin.

### B. Wet Deposition & Monsoon Scavenging
* **Rainfall Aerosol Scavenging**:
    * Exponential washout follows: <b><i>C</i>(<i>t</i>) = <i>C</i>₀ &times; <i>e</i><sup>&minus;&Lambda;<i>t</i></sup></b>, where &Lambda; is the scavenging coefficient scaling with precipitation intensity.
    * Coastal centers (Mumbai, Chennai) experience **~75% reduction in particulate mass** during peak precipitation spells.

### C. Solar Diurnal Cycles & Photochemical Ozone
* **Photochemical Formation**:
    * Ground-level Ozone (O₃) is generated when solar UV radiation splits Nitrogen Dioxide (NO₂) in the presence of VOCs:
        * Step 1: `NO₂ + Sunlight (UV) → NO + O`
        * Step 2: `O + O₂ → O₃ (Tropospheric Ozone)`
* **Diurnal Pattern**:
    * Peaks sharply between **12:00 PM and 4:00 PM** corresponding with peak solar irradiance.
    * Titrated at night by fresh vehicle NO emissions, dropping ground ozone concentrations back to baseline.

<div class="doc-image-card">
  <img src="../assets/images/seasonal_trends.png" alt="Seasonal Trends in Atmospheric Pollutants" style="max-height: 420px; width: auto;" />
  <div class="doc-image-caption">Figure 2: Empirical Seasonal Particulate & Gaseous Trends across Monitored Indian Climatic Corridors</div>
</div>

---

## 2. Telemetry Ingestion & Sensor Protocol

### Monitoring Infrastructure
* **Source**: Central Pollution Control Board (CPCB) Continuous Ambient Air Quality Monitoring Station (CAAQMS) network.
* **Volume**: **842,160+ continuous hourly records**.
* **Geographic Coverage**: **29 major Indian urban centers** covering Northern, Central, Coastal, and Northeastern corridors.

### Monitored Sensor Channels

| Category | Channel | Unit | Significance |
| :--- | :--- | :--- | :--- |
| **Fine Particulate** | PM2.5 | µg/m³ | Microscopic particles entering bloodstream; primary AQI driver |
| **Coarse Particulate** | PM10 | µg/m³ | Inhalable dust, construction debris, road particulate |
| **Gaseous Pollutants** | NO₂ | µg/m³ | Combustion byproduct from vehicle exhaust and thermal power |
| **Gaseous Pollutants** | SO₂ | µg/m³ | Industrial emissions and coal-fired generation |
| **Gaseous Pollutants** | CO | mg/m³ | Incomplete vehicular combustion and biomass burning |
| **Photochemical** | O₃ | µg/m³ | Ground-level secondary oxidant; daytime respiratory irritant |
| **Meteorology** | Temp | °C | Thermal inversion trigger; diurnal temperature amplitude |
| **Meteorology** | Humidity | % | Hygroscopic particulate growth; secondary sulfate formation |
| **Meteorology** | Pressure | hPa | High-pressure anticyclonic stagnant capping |
| **Meteorology** | Wind Speed | km/h | Atmospheric ventilation and mechanical dispersion |
| **Meteorology** | Rainfall | mm | Wet deposition aerosol scavenging |

---

## 3. The Missingness Protocol: City × Season Median Imputation

Raw sensor telemetry in developing nations exhibits **>35% missing values** due to power outages, optical sensor zero-point drift, and calibration blackouts.

* **Why Global Mean Imputation Was Rejected**:
    * Imputing nationwide averages destroys localized microclimates.
    * Example: Imputing a hot, dry Delhi summer temperature (42°C) into an unmonitored monsoon day in Thiruvananthapuram introduces severe artificial contamination.
* **The Solution: City × Season Median Imputation**:
    * Partitioned records into 29 city clusters across 4 meteorological seasons (Winter, Pre-Monsoon, Monsoon, Post-Monsoon).
    * Imputed missing values using the exact localized subset median:
      <div style="background: #F8FAFC; border: 1px solid #DBEAFE; border-left: 4px solid #1E3A8A; border-radius: 8px; padding: 14px 20px; margin: 12px 0; font-family: 'Cambria Math', 'Times New Roman', serif; text-align: center;">
        <div style="font-size: 17px; font-weight: 700; color: #1E3A8A; letter-spacing: 0.02em;">
          <i>x̂</i><sub><i>i, c, s</i></sub> = Median( { <i>x</i> &in; &#119967; | City = <i>c</i>, &nbsp; Season = <i>s</i> } )
        </div>
      </div>
    * Preserves local variance, inter-quartile ranges, and microclimate characteristics with **zero cross-city leakage**.

---

## 4. Feature Engineering Factory: 233 → 36 Features

AirIntel synthesized **233 candidate features** across multiple domains:

* **Cyclic Temporal Harmonics**:
    * Smooth circular encodings preventing artificial boundary discontinuities between December and January:
      <div style="background: #F8FAFC; border: 1px solid #DBEAFE; border-left: 4px solid #1E3A8A; border-radius: 8px; padding: 14px 20px; margin: 12px 0; font-family: 'Cambria Math', 'Times New Roman', serif; text-align: center;">
        <div style="font-size: 15.5px; font-weight: 700; color: #1E3A8A; letter-spacing: 0.02em;">
          Month<sub>sin</sub> = sin(2&pi; &times; Month / 12), &emsp; Month<sub>cos</sub> = cos(2&pi; &times; Month / 12)<br>
          Hour<sub>sin</sub> = sin(2&pi; &times; Hour / 24), &emsp; Hour<sub>cos</sub> = cos(2&pi; &times; Hour / 24)
        </div>
      </div>
* **Thermodynamic Couplings**:
    * `Temp × Humidity` (heat index interaction proxy).
    * `Surface Pressure / Temp` (atmospheric density and inversion capping).
    * `Wind Speed × PM2.5` (pollutant flux and dispersion index).
* **Spatio-Temporal Proxies**:
    * Northern India binary indicator (`Latitude > 20°N`).
    * Absolute Latitude and Latitude-Longitude interaction.

### The 8-Way Consensus Feature Selection
All 233 features were scored by 8 independent algorithmic selectors:
1. **Random Forest Gini Importance**
2. **LightGBM Split Gain**
3. **XGBoost Weight Importance**
4. **Permutation Importance**
5. **Mutual Information Score**
6. **Lasso (L1) Regularization Shrinkage**
7. **Spearman Non-Linear Rank Correlation**
8. **SHAP Mean Absolute Attribution**

Only features reaching top-tier consensus across multiple independent evaluators were selected, producing the lean, high-accuracy **36 production features** that enable **< 20 ms serving**.

<div class="doc-image-card">
  <img src="../assets/images/correlation_heatmap.png" alt="Atmospheric Feature Correlation Heatmap" style="max-height: 440px; width: auto;" />
  <div class="doc-image-caption">Figure 3: Feature Collinearity & Atmospheric Interaction Heatmap across Primary Sensor Channels</div>
</div>
