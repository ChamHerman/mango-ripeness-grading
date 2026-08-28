# Plan: Texture & Surface Analysis — Review, Rework & Document (Kai Bin)

## Goal
Rework `notebooks/texture_analysis_kb.ipynb` into a complete, correct, runnable pipeline (feature extraction → CSV → 3-way classifier comparison → saved outputs), add a teaching/plan markdown file, and verify everything runs end-to-end on the real `cleaned_data` (715 images).

## Part A — The 4 things you asked to understand (full detail goes in the .md file and notebook markdown cells)

**1. GLCM (Gray Level Co-occurrence Matrix) — what & why.** GLCM counts how often a pair of gray levels (i, j) appear as neighbours separated by a fixed offset (here 1 pixel, in 4 directions: 0°, 45°, 90°, 135°). From that matrix we derive 4 scalar features:
- *Contrast* — Σ(i−j)²P(i,j): how much gray levels differ between neighbours → surface roughness. Riper/overripe mangoes develop spots and wrinkling → higher contrast.
- *Correlation* — linear dependency of a pixel on its neighbour → how consistent/smooth the texture is.
- *Energy* — Σ P(i,j)²: textural uniformity; a smooth, uniform surface gives high energy.
- *Homogeneity* — Σ P(i,j)/(1+|i−j|): how close the co-occurrences are to the diagonal → smoothness; spoilage irregularities lower it.
Averaging the 4 directions makes the feature rotation-invariant, so the result is the same whichever way the mango sits in the image.

**2. LBP (Local Binary Pattern) — what & why.** For each pixel, LBP compares it to its 8 neighbours on a circle of radius 1. Each neighbour contributes a binary bit (1 if ≥ centre pixel, else 0); the bits form a code 0–255. We use the `uniform` variant (most common patterns only, 10 possible codes for P=8), then take statistics of the LBP-transformed image inside the fruit only:
- *Mean* — average LBP code → overall texture tendency.
- *Variance* — spread of local patterns → texture roughness.
- *Entropy* — −Σ p·log₂p of the pattern histogram → how complex/random the surface is. As the skin changes (smooth → wrinkled/spotted), entropy changes, which is what separates ripeness stages.

**3. Feature extraction process.** For every image in each of the 6 folders (train/test × 3 classes): load → convert to grayscale → build a fruit mask (non-black pixels; the preprocessed images have a black background) → compute GLCM only on masked pixels (zero-out the background level row/col so the black background doesn't skew features) → average the 4 directions → compute LBP (P=8, R=1) on masked fruit pixels → mean/variance/entropy. Each image becomes one row with 7 features + filename + class + train_or_test; all rows go into one CSV. Per-image try/except skips corrupted files; tqdm shows progress.

**4. Classification process.** Load the CSV → map labels (unripe=0, fully_ripe=1, overripe=2) → split X/y strictly by the `train_or_test` column (no random split, no leakage) → scale features with StandardScaler (fit on train only — essential for SVM and KNN whose distances are scale-sensitive) → train Random Forest, SVM (RBF), and KNN → evaluate each on the untouched test set with accuracy, per-class precision/recall/F1, and a confusion matrix → print a comparison summary table → pick the best model (highest test accuracy) → save its classification report and confusion matrix as PNG images + save the model with joblib (consistent with the color/morphology pipelines).

## Part B — Correctness review of your current notebook (what's right / what I'll fix)
Keep (correct and defensible): mask-based GLCM with background level zeroing + renormalisation; LBP on masked pixels; uniform LBP with P=8, R=1; Shannon entropy via normalised histogram; visual inspection grid; RF feature-importance plot.
Fix / enhance:
1. **Path robustness** — `ROOT = '../cleaned_data'` only works if the kernel cwd is `notebooks/`. Replace with pathlib resolution that tries candidates and raises a clear error if the data folder isn't found.
2. **Column naming** — rename `split` → `train_or_test` to match your spec exactly.
3. **Progress bar** — add tqdm to the feature-extraction loop (currently missing).
4. **Add SVM and KNN** — only Random Forest exists today; your spec requires all three with a comparison.
5. **Feature scaling** — StandardScaler fit on training data only, before SVM/KNN.
6. **Save outputs** — feature CSV (required), best model as `.joblib`, classification report + confusion matrix as PNGs (required). Currently none are saved.
7. **Guards** — assert non-empty DataFrames and that all 3 classes exist before training; informative errors instead of silent empty-frame crashes.
8. **Small** — add tqdm to `requirements.txt` (used by every other notebook, currently missing).

## Part C — Files I will change/create

**1. Rewrite `notebooks/texture_analysis_kb.ipynb`** — same overall story, restructured into ~15 labelled steps, each with a teaching markdown cell + code cell:
- Title & overview (your name/task)
- Step 1 Imports & Configuration (incl. robust path resolution, GLCM/LBP params documented)
- Step 2 Fruit Mask Extraction
- Step 3 GLCM Features (4 directions averaged, distance 1)
- Step 4 LBP Features (P=8, R=1, uniform)
- Step 5 Combined pipeline function
- Step 6 Visual inspection grid (original / grayscale / LBP map / LBP histogram)
- Step 7 Build feature dataset (tqdm + try/except) for train & test
- Step 8 Save features to `output/texture_features.csv`
- Step 9 Load CSV, encode labels (unripe=0, fully_ripe=1, overripe=2), split by `train_or_test`
- Step 10 Feature scaling (StandardScaler on train only)
- Step 11 Train Random Forest, SVM, KNN
- Step 12 Evaluate & compare (accuracy, classification_report, confusion matrices, summary table)
- Step 13 Save best model + report & confusion-matrix images
- Step 14 Feature importance (Random Forest)
- Step 15 Feature distribution across classes (boxplots — material for your Results/Discussion section)

**2. Create `notebooks/texture_analysis_plan_kb.md`** — your requested plan/explanation file, written to teach, containing:
- Overview of the Texture & Surface Analysis task
- Detailed explanations of GLCM and LBP and why each is useful for ripeness
- The full pipeline explained step by step (mapped to the notebook cells)
- The classification step explained (encoding, scaling, the 3 models, metrics, how to pick the best)
- The parameters chosen (GLCM distance=1 & 4 directions; LBP P=8, R=1) and how to justify them in the report
- How to read the results (what the metrics mean, tie-in to the ≥85% accuracy SMART objective)
- Report tips (pseudocode, visualisations for §4.1, discussion angles for §4.2)
- A checklist of what was wrong in the original notebook and what I changed

**3. `requirements.txt`** — add `tqdm>=4.66.0` (one line; already used by the preprocessing and morphology notebooks).

## Part D — Verification (after approval)
Run the notebook end-to-end with `jupyter nbconvert --to notebook --execute` (or `jupyter execute`) so the real pipeline runs on all 715 images. Expected outputs: `output/texture_features.csv` (715 rows, 10 columns), a trained `.joblib` model, and report/confusion-matrix PNGs. Fix any runtime issues until it executes cleanly with no errors, then report the actual test accuracies of RF / SVM / KNN to you.

Note: running the notebook will write `output/texture_features.csv`, `output/texture_model.joblib`, and two PNGs — these are the assignment deliverables, so that's expected (no existing file named `texture_features.csv` will be overwritten — it doesn't exist yet).