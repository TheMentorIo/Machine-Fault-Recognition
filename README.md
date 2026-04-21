
# Cairo University
## Faculty of Engineering
### Computer Engineering Department

# Pattern Recognition and Neural Networks
## Project Document
### Machine Fault Recognition
**Spring 2026**

## Preprocessing-only (working)
- Install deps: `pip install -r requirements.txt`
- Run on a folder: `python src/main.py --input "Prototype 1/Student" --output preprocessed`
- Run on one file: `python src/main.py --input path\\to\\file.wav --output preprocessed`
- Supported inputs: `.wav` always; `.flac/.ogg` if `soundfile` is installed (included in requirements)

---

# Contents
1. [Objectives](#objectives)  
2. [Project: Machine Listener System](#project-machine-listener-system)  
3. [Team Formation](#team-formation)  
4. [Research Team Proposal](#research-team-proposal)  
5. [Final Deliverables](#final-deliverables)  
6. [Rules and Instructions](#rules-and-instructions)  
7. [Grading Criteria](#grading-criteria)  
8. [FAQ](#faq)  
9. [Delivery Instructions](#delivery-instructions)  

---

## Objectives
By the end of this project, you should be able to:  
1. Analyze the problem, extract features, and choose the most appropriate preprocessing method (if necessary).  
2. Assess performance of different machine learning techniques.  
3. Design and implement your own ML pipeline.  

---

## Project: Machine Listener System
In this project, you are required to implement a Mahine Fault Recognition System. Given an audio file, your system is supposed to classify the audio into one of 6 classes (from 0 to 5). Each audio file represents a recording of a machine in a factory. There are 3 types of machines and two states (normal/abnormal) for each type.

**Label Classes:**  
- 0: Machine 1, Normal  
- 1: Machine 1, Abnormal  
- 2: Machine 2, Normal  
- 3: Machine 2, Abnormal  
- 4: Machine 3, Normal  
- 5: Machine 3, Abnormal  

### System Specs
Your system should handle variations and implement a complete ML pipeline, including:  
- **Preprocessing Module**  
- **Feature Extraction/Selection Module**  
- **Model Selection and Training Module**  
- **Performance Analysis Module**  

Preprocessing should handle:  
- Background noise  
- Volume variations  
- Silence removal  
- Different microphone quality  

You will need to research related literature to identify the best approaches to improve accuracy.

### Dataset
The dataset can be downloaded from:  
[Dataset Link](https://drive.google.com/drive/folders/1pFPnn7lbpxWVfOmrDEyvHxXp0_6GFB90)

Divide the dataset into:  
- Training set  
- Validation set  
- Test set  

---

## Team Formation
- Teams of **3 to 4 members**. No team should have more than 4.  
- Register your team number in the provided spreadsheets.  
- All team members must attend the final project discussion. Non-attendance = zero contribution.  

---

## Research Team Proposal
Optionally, a team can work on an innovative research idea. The team will be supervised by the instructor and may publish a research paper. Research teams are **not part of the competition** and grades are based on the research work. Proposals must explain motivation and contribution.

---

## Final Deliverables
1. **PDF Report** including:  
   - Abstract  
   - Introduction  
   - System Overview  
   - Experimental Setup  
   - Results and Analysis  
   - Conclusion  
   - Workload distribution  

   Recommended template: [Overleaf Template](https://www.overleaf.com/read/wbsmvbqzkytx#c67e8b)  

2. **Code**: zipped folder `code[team number].zip` including `readme.md` and `requirements.txt`. Include either an executable or `infer.py` for test execution.  

3. **Model file** in correct path; should load and run for predictions (Pickle recommended).  

4. **Dockerfile** to replicate environment.  

5. **BONUS**: Deploy on cloud (Hugging Face recommended).  

---

## Rules and Instructions
1. All submissions on the drive; no printed copies.  
2. Late penalty applies.  
3. Cheating/plagiarism = ZERO for both teams + up to 50% penalty.  
4. Workload must be fair; non-contributors penalized.  
5. All members must attend final discussion; absentees get ZERO.  

---

## Grading Criteria
- **Report and Discussion**: 20%  
- **Project Performance**: 80% (Accuracy 65%, Runtime 15%)  

> Rankings are based on accuracy and time; non-linear weighting.

---

## FAQ
1. Any programming language is allowed.  
2. Free to use classical ML or deep learning (CNN, RNN, etc.) within 2B parameters.  
3. Research paper resources: EKB, Microsoft Academic, Google Scholar.  
4. Projects tested on unseen test set; divide your data into train/validation/test.  
5. Ranking: based on accuracy + runtime combination.  

---

## Delivery Instructions
### Input Format
Test set folder `data` contains files `1.wav, 2.wav, ...`  

Process files in increasing order. Output must match input order.

### Output Format
Generate **two text files**:  
1. `results.txt` → model predictions (0–5), one per line  
2. `time.txt` → runtime in seconds per test file, one per line  

**Important Notes**:  
- No extra info or test case numbers in files  
- Round runtime to three decimals  
- Do not print results to console  

---

**Example:**

**results.txt:**  
```

2
3
0
2
1
5
3
...

```

**time.txt:**  
```

30.000
35.205
32.320
40.150
35.121
38.982
26.233
...
20.654

```
```

