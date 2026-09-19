# TreeSHAP & Spatial Autocorrelation Analytics

Black-box machine learning models are unacceptable in environmental epidemiology and municipal policy planning. AirIntel integrates **TreeSHAP game-theoretic attributions** and **spatial econometric statistics** to provide full transparency into every single prediction.

---

## 1. Game-Theoretic TreeSHAP Foundations

SHAP (SHapley Additive exPlanations) computes the marginal contribution of each feature across all possible feature subsets:

<div style="background: #F8FAFC; border: 1px solid #DBEAFE; border-left: 4px solid #1E3A8A; border-radius: 8px; padding: 16px 20px; margin: 18px 0; font-family: 'Cambria Math', 'Times New Roman', serif; text-align: center;">
  <div style="font-size: 17px; font-weight: 700; color: #1E3A8A; letter-spacing: 0.02em;">
    &phi;<sub><i>i</i></sub>(<i>x</i>) = &sum;<sub><i>S</i> &sube; <i>F</i> &setminus; {<i>i</i>}</sub> 
    <sup>|<i>S</i>|! (|<i>F</i>| &minus; |<i>S</i>| &minus; 1)!</sup>&frasl;<sub>|<i>F</i>|!</sub>
    &times; [ <i>f</i><sub><i>S</i> &cup; {<i>i</i>}</sub>(<i>x</i><sub><i>S</i> &cup; {<i>i</i>}</sub>) &minus; <i>f<sub>S</sub></i>(<i>x<sub>S</sub></i>) ]
  </div>
</div>

TreeSHAP optimizes this calculation from exponential time **O(TL2<sup>|F|</sup>)** down to polynomial time **O(TLD²)**, making live, real-time attribution possible in under 20 milliseconds on CPU.

### Core Mathematical Axioms Satisfied:
1. **Efficiency**: The sum of all feature SHAP values plus the base expected value equals the model output:
   <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 6px; padding: 10px 16px; margin: 8px 0; font-size: 15px; font-weight: 700; color: #1E40AF; text-align: center; font-family: 'Cambria Math', 'Times New Roman', serif;">
     <i>f</i>(<i>x</i>) = <b>E</b>[<i>f</i>(<i>x</i>)] + &sum;<sub><i>i</i>=1</sub><sup><i>M</i></sup> &phi;<sub><i>i</i></sub>(<i>x</i>)
   </div>
   For AirIntel's continuous regressor, the empirical baseline expected value across the nationwide dataset is **E[f(x)] = 112.5 AQI**.
2. **Symmetry**: If two features contribute equally across all subsets, their attributions are identical.
3. **Dummy (Null Effect)**: Features with zero impact receive exactly **&phi;<sub><i>i</i></sub> = 0**.
4. **Additivity**: Attributions sum linearly across ensemble trees.

<div class="doc-image-card">
  <img src="../assets/images/shap_beeswarm.png" alt="TreeSHAP Global Summary Beeswarm" style="max-height: 480px; width: auto;" />
  <div class="doc-image-caption">Figure: TreeSHAP Global Feature Impact Beeswarm across Continuous US AQI Predictions</div>
</div>

---

## 2. Real-Time Attribution Decomposition (Waterfall Chart)

In the interactive prediction engine, every single inference is decomposed in real time into directional pushes:

```text
E[f(x)] Baseline = 112.5 AQI
    │
    ├─► PM2.5 Mass (+158.4) ────────────► [Elevates AQI]
    ├─► Thermal Inversion Temp (+11.5) ─► [Elevates AQI]
    ├─► Surface Pressure Capping (+2.3) ► [Elevates AQI]
    ├─► Wind Ventilation (-6.7) ────────► [Lowers AQI / Cleans Air]
    │
    ▼
Predicted AQI = 278.0 (Very Unhealthy)
```

### Interpretation Palette:
- **Rose Vermilion (`#E11D48`)**: Positive impact pushes that degrade air quality and elevate AQI.
- **Cerulean Ocean (`#0284C7`)**: Negative impact pushes (e.g. strong wind dispersion or heavy rainfall) that cleanse the atmosphere and lower AQI.

---

## 3. Spatial Autocorrelation & Moran's I Analysis

Air pollution respects physical geography rather than political boundaries. To measure the spatial dependency of atmospheric quality across Indian urban centers, we computed **Global Moran's I**:

<div style="background: #F8FAFC; border: 1px solid #DBEAFE; border-left: 4px solid #1E3A8A; border-radius: 8px; padding: 16px 20px; margin: 18px 0; font-family: 'Cambria Math', 'Times New Roman', serif; text-align: center;">
  <div style="font-size: 18px; font-weight: 700; color: #1E3A8A; letter-spacing: 0.02em;">
    <i>I</i> = <sup><i>N</i></sup>&frasl;<sub><i>S</i>₀</sub> &times; 
    [ &sum;<sub><i>i</i>=1</sub><sup><i>N</i></sup> &sum;<sub><i>j</i>=1</sub><sup><i>N</i></sup> <i>w<sub>ij</sub></i> (<i>x<sub>i</sub></i> &minus; <i>x̄</i>)(<i>x<sub>j</sub></i> &minus; <i>x̄</i>) ]
    &frasl;
    [ &sum;<sub><i>i</i>=1</sub><sup><i>N</i></sup> (<i>x<sub>i</sub></i> &minus; <i>x̄</i>)&sup2; ]
  </div>
</div>

* **<i>N</i> = 29**: Total national monitoring centers.
* **<i>w<sub>ij</sub></i> = 1 / <i>d<sub>ij</sub></i>**: Inverse-distance spatial weight matrix between city *i* and city *j*.
* **<i>S</i>₀ = &sum;<sub><i>i</i></sub> &sum;<sub><i>j</i></sub> <i>w<sub>ij</sub></i>**: Aggregate normalization sum of all spatial weights.

### Empirical Spatial Findings:
* **Global Moran's <i>I</i> = 0.412** (*z* = 4.82, *p* < 0.001):
  Confirms strong positive spatial autocorrelation — polluted cities cluster together geographically, rather than being randomly distributed.
* **High-High Cluster (Indo-Gangetic Basin)**:
  Delhi, Gurugram, Lucknow, Patna, and Kolkata form a contiguous spatial corridor of severe winter particulate stagnation.
* **Low-Low Cluster (Deccan Plateau & Southern Coast)**:
  Bengaluru, Hyderabad, Panaji, and Thiruvananthapuram benefit from higher wind ventilation and maritime air mass replenishment.
