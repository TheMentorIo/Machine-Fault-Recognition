### **1. Main Goal**
Clean and standardize the audio data so that the features extracted in the next stage are meaningful and robust. Handle variations like **background noise, volume differences, silence, and microphone quality**.

---
### **2. Tasks / What to Do**
1. **Audio Inspection**
    - Listen to sample audio files.
    - Check for:
        - Noise (machinery hum, ambient noise)
        - Silence or pauses
        - Volume variations
        - Differences in microphone quality
    - Identify problematic audio patterns that could affect classification.
        
2. **Noise Reduction**
    - Remove background noise using:
        - Spectral gating
        - Bandpass filters
        - Adaptive noise cancellation
    - Compare different noise removal methods to see which works best.
        
3. **Silence Removal**
    - Trim leading and trailing silence.
    - Optionally segment audio into active sound windows.
    
4. **Volume Normalization**
    - Adjust audio amplitude to a standard level.
    - Ensure loud and soft recordings are comparable.
    
5. **Resampling / Mic Correction**
    - Resample audio to a common sampling rate if needed.
    - Optional equalization if microphone quality varies.

---

### **3. Research Focus**

For each preprocessing step, the team member should **search for papers and methods** like:
1. **Noise Reduction**
    - “Denoising industrial audio recordings”
    - “Spectral gating vs wavelet denoising in audio”
    - “Deep learning audio denoising techniques”
        
2. **Silence Removal**
    - “Voice activity detection”
    - “Silence trimming algorithms for audio classification”
        
3. **Volume Normalization**
    - “Audio normalization for machine learning”
    - “RMS-based amplitude scaling vs peak normalization”
        
4. **Data Augmentation
    - “Time stretch, pitch shift for industrial audio”
    - “Adding synthetic noise for robustness”
    - “Mixing sounds to simulate real-world conditions”    
---
### **4. Areas for Creativity**

- **Data Augmentation:**
    - Slightly speed up or slow down audio to simulate different machine speeds.
    - Add low-level noise to simulate factory conditions.
    - Pitch shift to simulate microphone variations.
        
- **Segmenting Audio:**
    - Split long recordings into multiple smaller windows to capture temporal patterns.
        
- **Adaptive Preprocessing:**
    - Automatically detect noisy files and apply stronger denoising only when needed.
        
- **Visual Feedback:**
    - Plot spectrograms before/after preprocessing to visually verify improvements.