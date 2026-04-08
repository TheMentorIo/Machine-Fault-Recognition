### **1. Main Goal**
Train one or more models that can accurately classify machine audio into the 6 classes. Explore **classical machine learning**, **deep learning**, and **ensemble/boosting techniques**, while using research to guide design choices.

---
### **2. Tasks / What to Do**

#### **Step 1: Baseline Models**
Start simple to get a reference performance.
- **Classical ML models**:
  - **Random Forest:** Handles high-dimensional features well; gives feature importance.
  - **SVM (Support Vector Machine):** Effective for small-medium datasets; good for structured features like MFCC.
  - **KNN:** Simple distance-based classifier; good as a sanity check.
- **Goal:** Compare baseline accuracy and runtime to know what improvements are needed.

---
#### **Step 2: Deep Learning Models**
- **CNN (Convolutional Neural Network):**
  - Best for spectrogram images (spatial patterns of sound).  
  - Architecture tips: 2–4 convolutional layers → pooling → fully connected → softmax.
- **LSTM / GRU (Recurrent Neural Network):**
  - Good for sequential audio features (time dependencies).  
  - Can take MFCC sequences or raw audio frames as input.
- **Hybrid CRNN:** Combine CNN for feature extraction and LSTM for sequence modeling.
- **Goal:** Capture both spatial and temporal characteristics of audio.

---
#### **Step 3: Ensemble & Boosting (Creativity + Research)**
- **Voting Classifier:** Combine predictions of multiple models (e.g., RF + SVM + CNN) via majority voting.  
- **AdaBoost / Gradient Boosting:** Boost weak learners to improve performance.  
- **Bagging:** Reduce variance for unstable models like decision trees.  
- **Goal:** Increase accuracy and robustness by leveraging multiple models.

---
#### **Step 4: Hyperparameter Tuning**
- **Classical ML:** Number of trees (RF), kernel type & C parameter (SVM), k (KNN).  
- **Deep Learning:** Learning rate, number of layers, dropout rate, batch size, sequence length.  
- **Method:** Use validation set to select the best hyperparameters.  

---
#### **Step 5: Model Validation & Selection**
- Split dataset: train / validation / test.  
- Evaluate:
  - Accuracy per class  
  - Confusion matrix  
  - Runtime per file  
- Select best-performing model or ensemble for the final system.

---
### **3. Research Focus**
At each sub-step, research should guide decisions:
- **Baseline Models:** Compare methods in literature for audio fault detection.  
- **Deep Learning Architectures:** Look for CNN/LSTM/CRNN architectures used in similar problems.  
- **Ensemble & Boosting:** Check which ensemble methods improved results in previous studies.  
- **Hyperparameter Tuning:** Explore grid search, random search, or Bayesian optimization strategies for audio classification.
---
### **4. Areas for Creativity**
- **Hybrid models:** Combine deep learning with classical ML (e.g., CNN features → SVM classifier).  
- **Ensemble innovation:** Weighted voting or stacking multiple models.  
- **Adaptive models:** Use different preprocessing pipelines for different machine types (if dataset allows).  
- **Runtime optimization:** Experiment with smaller models or pruning for faster predictions.
