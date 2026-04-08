### **1. Main Goal**
Transform preprocessed audio into meaningful numerical features that a model can use to classify machine faults. Focus on **capturing characteristics that differentiate machine types and states** (normal vs abnormal).

---
### **2. Tasks / What to Do**
1. **Extract Time-Domain Features**
    - **Zero-Crossing Rate (ZCR):** Frequency of signal crossings through zero — indicates noise and vibration patterns.
    - **Energy / RMS:** Loudness variations over time.
    - **Entropy of Energy:** Measures irregularity in energy patterns.
        
2. **Extract Frequency-Domain Features**
    - **MFCCs (Mel-Frequency Cepstral Coefficients):** Most commonly used audio features.
    - **Spectral Centroid:** Indicates the “brightness” of the sound.
    - **Spectral Contrast:** Measures difference between peaks and valleys of spectrum.
    - **Chroma Features:** Useful for periodic vibrations or tonal components.
        
3. **Deep Learning Features (Optional / Creative)**
    - Convert audio to **spectrograms** or **Mel-spectrograms** to feed CNNs.
    - Try **log-Mel spectrograms** or **MFCC + delta features** for richer representation.
        
4. **Feature Selection & Dimensionality Reduction**
    - Evaluate which features are most important for classification.
    - Apply:
        - **PCA (Principal Component Analysis)**
        - **LDA (Linear Discriminant Analysis)**
        - Or feature importance methods (e.g., Random Forest importance).
    - Optional: Combine multiple feature sets (time + frequency + deep features).

---
### **3. Research Focus**
For each extraction step, search for recent methods and best practices:
1. **Traditional Features**
    - “MFCC for industrial machine fault detection”
    - “Spectral features in audio classification”
    - “Chroma features for vibration sounds”

2. **Deep Learning Features**    
    - “CNN for audio classification using spectrograms”
    - “CRNN for sequential audio”
    - “Hybrid MFCC + CNN approaches”

3. **Feature Selection**    
    - “PCA vs LDA for audio classification”
    - “Automatic feature selection for sound classification”
    - “Ensemble-based feature importance methods”

---
### **4. Areas for Creativity**
- **Hybrid Feature Sets:** Combine time-domain, frequency-domain, and deep features to increase model robustness.
- **Feature Augmentation:** Create features representing **differences between consecutive frames** or **statistical summaries** (mean, variance, skewness).
- **Segment-Based Features:** Extract features from small windows of audio (1–2 seconds) and summarize them (mean/max/std).
- **Visual Features:** Use spectrograms as images for CNN, or apply **image augmentation techniques** (cropping, scaling) for audio features.