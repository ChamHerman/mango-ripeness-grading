# Texture & Surface Analysis — Plan and Full Explanation

**Developer:** Wong Kai Bin
**Notebook:** `texture_analysis_kb.ipynb` (in this folder)
**Task:** Texture and Surface Analysis for the automated mango ripeness grading project.

This file explains **what** the notebook does, **why** it does it that way, and **how to write it up**
in the assignment documentation. It was written so you understand every decision — not just the code.

---

## 1. What this task is about

The project classifies mango images into three ripeness stages: **unripe**, **fully_ripe** and **overripe**.
Your part is the **texture** of the fruit surface. As a mango ripens, its skin changes: it becomes
smoother, then wrinkled and blotchy, and overripe fruit develops dark spots and a dull, uneven surface.
These changes alter the *spatial arrangement of pixel intensities* on the surface — that is exactly what
**texture features** measure.

The notebook does two jobs:

1. **Feature extraction** — for every cleaned image, compute 7 numbers (features) that describe its
   surface texture, and save them to a CSV.
2. **Classification** — feed those features into three machine learning classifiers
   (Random Forest, SVM, KNN), compare them, and save the best one.

---

## 2. What is GLCM and why is it useful for ripeness?

A **Gray Level Co-occurrence Matrix (GLCM)** is a statistical table that answers a simple question:
*"How often does a pixel of grey level `i` sit next to a pixel of grey level `j`, at a given distance
and direction?"*

### How it is built
1. Convert the image to grayscale (each pixel has a grey level 0–255).
2. Pick an offset — here **distance = 1 pixel** in **four directions**: 0°, 45°, 90°, 135°.
3. For each direction, count how often each pair `(i, j)` occurs. The counts form a 256×256 matrix.
4. Average the four matrices so the result is **rotation-invariant** — a mango lying at any angle
   gives the same features.

### The 4 features derived from the matrix

| Feature | Formula (idea) | What it measures | Expected trend as mango ripens |
|---|---|---|---|
| **Contrast** | Σ (i−j)² · P(i,j) | how much grey levels differ between neighbours → **roughness** | increases (spots, wrinkles) |
| **Correlation** | linear dependence of pixel on its neighbour | how **consistent/regular** the texture is | changes with blotchiness |
| **Energy** | Σ P(i,j)² | **uniformity** of the texture | high for smooth, even surfaces |
| **Homogeneity** | Σ P(i,j) / (1 + |i−j|) | how close co-occurrences are to the diagonal → **smoothness** | decreases with irregularities |

### Why it helps for ripeness
Unripe mangoes have a smooth, fine-grained surface; overripe mangoes are rough and irregular.
These differences change how often certain grey-level pairs co-occur, so the four GLCM values
differ between stages and give the classifier useful evidence.

**Background handling (important detail):** the cleaned images have a *black background* (value 0).
If we kept it in the GLCM, the huge black area would dominate the counts and hide the fruit's texture.
So the code zeroes-out row/column 0 of the matrix (all pairs involving grey level 0) and re-normalises
the rest. This is why the extraction also builds a **fruit mask** (Step 2) first.

---

## 3. What is LBP and why is it useful for ripeness?

A **Local Binary Pattern (LBP)** describes the local *structure* around each pixel, not the colour.

### How it works
1. For each centre pixel, take its **8 neighbours** on a circle of **radius 1**.
2. Compare each neighbour with the centre: **neighbour ≥ centre → 1, otherwise → 0**.
3. The 8 bits form a binary code (0–255) that describes the local pattern (edge, corner, flat area, …).

We use the **uniform** variant: only patterns with at most two 0↔1 transitions are kept.
Uniform patterns are the most common and robust to noise, and for P=8 there are only **10** codes (0–9).

### The 3 statistical features extracted from the LBP image

| Feature | What it measures | Why it matters for ripeness |
|---|---|---|
| **Mean** | the average LBP code in the fruit | overall texture tendency of the surface |
| **Variance** | how spread out the local patterns are | roughness — more varied patterns → higher variance |
| **Entropy** | −Σ p·log₂(p) of the pattern histogram | **complexity/randomness** of the surface; a wrinkled, spotted skin has higher entropy than a smooth one |

Again, the features are computed **only on fruit pixels** (using the mask) so the black background
does not influence the statistics.

---

## 4. How does the feature extraction process work?

Step by step (this is what Steps 1–8 of the notebook do):

1. **Define paths and parameters** — locate the `cleaned_data` folder robustly (the kernel may start
   in different folders), define the three classes, GLCM parameters (distance 1, 4 directions) and
   LBP parameters (P=8, R=1).
2. **Fruit mask** — mark pixels that are not black as fruit (1) and the background as 0.
3. **Grayscale** — convert each image from BGR to a single intensity channel (texture does not need colour).
4. **GLCM** — build the co-occurrence matrix on the masked image, remove background pairs, re-normalise,
   and average Contrast, Correlation, Energy, Homogeneity over the 4 directions.
5. **LBP** — compute the uniform LBP image, take the fruit pixels, and calculate mean, variance, entropy.
6. **Loop over all images** — the same 7 features are extracted for every image in the six folders
   (`train|test` × `unripe|fully_ripe|overripe`). A `tqdm` progress bar shows progress; a `try/except`
   skips any file that cannot be read instead of crashing the run.
7. **Save** — one CSV (`output/texture_based/texture_features.csv`) with 10 columns:
   `filename`, `class`, `train_or_test`, and the 7 features. One row per image (715 rows in total:
   571 train + 144 test).

> **Why the mask matters twice:** without it, both GLCM and LBP would mostly describe the black
> background, not the mango. Masking is the single most important correctness decision in this notebook.

---

## 5. How does the classification process work?

This is Steps 9–13 of the notebook.

1. **Load the CSV** and separate features `X` (the 7 columns) from labels `y` (the class).
2. **Encode labels** — strings become integers so scikit-learn can use them:
   `unripe → 0`, `fully_ripe → 1`, `overripe → 2`.
3. **Split by `train_or_test`** — the train/test separation is **not** made randomly here; it reuses the
   80/20 split created during preprocessing. This avoids *data leakage* (e.g. near-duplicate photos
   appearing in both sets).
4. **Feature scaling** — `StandardScaler` gives every feature zero mean and unit variance, **fitted on
   the training set only**. SVM and KNN are distance-based: without scaling, the large-scale feature
   (e.g. GLCM contrast) would dominate the distance and the others would be ignored.
5. **Train 3 classifiers** on the same scaled features:
   - **Random Forest** — 100 decision trees, majority vote.
   - **SVM** (RBF kernel) — finds the boundary that best separates the classes.
   - **KNN** (k=5) — classifies by the 5 nearest neighbours.
6. **Evaluate on the unseen test set** with:
   - **Accuracy** — fraction of test images correct.
   - **Precision** — of the images predicted as class X, how many really are X.
   - **Recall** — of the images that really are X, how many were caught.
   - **F1-score** — the harmonic mean of precision and recall (balanced measure).
   - **Confusion matrix** — shows exactly which classes get confused.
7. **Pick the best** — highest test accuracy (ties broken by macro F1) — and save:
   - `output/texture_based/texture_model.joblib` (model + scaler + feature list, reusable later),
   - `output/texture_based/texture_classification_report.png` and `output/texture_based/texture_confusion_matrix.png` for the report.

### Verified results (run on all 715 images, 144-image test set)

| Classifier | Accuracy | Macro F1 |
|---|---|---|
| Random Forest | **96.53%** | 0.9653 |
| SVM (RBF) | 95.83% | 0.9583 |
| **KNN (k=5)** | **97.22%** | 0.9722 |

KNN was the best on this data — the texture features separate the three stages so cleanly that a
nearest-neighbour rule works very well. All three exceed the project's **85% accuracy objective**.
The KNN confusion matrix shows almost perfect separation: only 4 of 144 test images were misclassified,
and every fully_ripe mango was recognised correctly.

---

## 6. Parameters chosen and how to justify them in the report

The assignment documentation does **not** fix these values, so you should state and justify them:

| Parameter | Value | Justification (write this in the report) |
|---|---|---|
| GLCM distance | 1 pixel | captures fine, local surface texture — most relevant to skin changes |
| GLCM directions | 0°, 45°, 90°, 135° | averaged → rotation-invariant features |
| GLCM grey levels | 256 | full 8-bit range, no information lost |
| LBP radius R | 1 | smallest neighbourhood — captures fine detail |
| LBP neighbours P | 8 | standard for R=1 |
| LBP method | uniform | robust to noise; only 10 codes for P=8, so the histogram is stable |
| Classifier seeds | random_state=42 | reproducible results |

---

## 7. How to read the results (for the report)

- **The confusion matrices** (printouts in Step 12 and the saved PNG) tell you the failure modes:
  for KNN, unripe was occasionally confused with fully_ripe/overripe — expected, because a mango's
  skin texture changes gradually.
- **The feature-importance chart** (Step 14) shows which of the 7 features Random Forest leaned on.
  Use it in the Discussion to argue which texture cues matter most.
- **The boxplots** (Step 15) show whether features separate the classes. Cite them in the Results as
  evidence of *why* classification works.
- **Metrics to quote:** test accuracy of the best classifier, per-class precision/recall/F1, and the
  confusion matrix. All of these appear in the saved PNGs.

---

## 8. Report-writing tips (mapped to the assignment document)

- **Methodology (§3.3, "Applications of the algorithms"):** describe the pipeline in your own words
  (sections 4–5 above) and include **pseudocode** for the feature extraction loop.
- **Results (§4.1):** use the summary table (section 5 above) and the two saved PNGs; add the
  feature-importance bar chart and boxplots from the notebook.
- **Discussion (§4.2):** do not just quote the numbers — explain *why* KNN won (clean separation in
  feature space), *why* some images were confused (gradual texture change between stages), and compare
  with the colour/morphology results from your teammates if available.
- **Conclusion (§5.1):** state that the texture pipeline met the ≥85% accuracy objective.

---

## 9. What was fixed / enhanced in the rewrite (vs. the original notebook)

| Item | Original notebook | Now |
|---|---|---|
| Data path | relative `'../cleaned_data'` — silently broke if kernel started elsewhere | robust search over candidates + clear error message |
| CSV column | named `split` | `train_or_test` (matches the spec) |
| Progress bar | none during extraction | `tqdm` bar |
| Classifiers | Random Forest only | Random Forest + SVM + KNN with comparison |
| Feature scaling | none (harmless for RF, needed for SVM/KNN) | `StandardScaler` fitted on train only |
| Saved outputs | none | feature CSV, model `.joblib`, report + confusion-matrix PNGs |
| Class label bug | — | class now taken from the folder name (`path.parent.name`), so labels can never be mixed up |
| Empty-data guards | none | informative errors before training |
| Visualisations | original + LBP map + histogram | + feature importance + per-class boxplots |
| Inference demo | none | final cell reloads the saved model and shows a prediction picture (like the other pipelines) |
| Requirements | — | `tqdm` added to `requirements.txt` |

---

## 10. How to run it

1. Make sure your Python environment has the packages in `requirements.txt`
   (`pip install -r requirements.txt`).
2. Open the notebook: `jupyter notebook notebooks/texture_analysis_kb.ipynb`
3. **Run ▸ Run All Cells.** The extraction of all 715 images takes about a minute.
4. Check the `output/texture_based/` folder for the CSV, model and two PNGs.
5. The last cell (Step 16) runs the **inference test** — it shows pictures of sample mangoes
   with their predicted ripeness, confidence and processing latency.

---

## 11. What each output file is (for your documentation)

| File | What it is | What it's for |
|---|---|---|
| `texture_features.csv` | 715 rows × 10 columns; every image's 7 texture features plus filename, class and split | the reproducible feature set used for classification |
| `texture_model.joblib` | one packaged Python object: the best classifier **plus** the scaler, feature list and label map | reload it to predict new images without retraining (used by the Streamlit app later) |
| `texture_classification_report.png` | the best classifier's precision / recall / F1 table, rendered as an image | paste into the Results section of the report |
| `texture_confusion_matrix.png` | heatmap of true vs predicted class counts for the best classifier | paste into the report; shows which stages get confused |

### The CSV columns explained

| Column | Meaning |
|---|---|
| `filename` | the image file name |
| `class` | the **true** ripeness stage: unripe / fully_ripe / overripe |
| `train_or_test` | which of the two preprocessing splits the image belongs to |
| `glcm_contrast` / `glcm_correlation` / `glcm_energy` / `glcm_homogeneity` | the four GLCM texture features (see section 2) |
| `lbp_mean` / `lbp_variance` / `lbp_entropy` | the three LBP statistics (see section 3) |

---

## 12. Testing on a new image (Step 16)

The notebook ends with an **inference test**, the same idea as the other pipelines' demos:

1. it reloads `texture_model.joblib` (model + scaler),
2. runs the texture extraction on one image,
3. scales the features with the saved scaler,
4. predicts the ripeness with the best classifier,
5. draws the image with the **true class, predicted class, confidence and latency** in the title.

Verified on one sample per class: unripe → unripe (80% confidence), fully_ripe → fully_ripe
(100%), overripe → overripe (100%), at about **74 ms per image** — comfortably under the
project's 200 ms per-image latency objective.
