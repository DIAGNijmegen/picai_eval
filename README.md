# 3D Lesion Detection Evaluation
![Tests](https://github.com/DIAGNijmegen/picai_eval/actions/workflows/tests.yml/badge.svg)

This repository contains code to evaluate 3D detection performance, geared towards Prostate Cancer Detection in MRI. This repository contains the official evaluation pipeline of the [Prostate Imaging: Cancer AI (PI-CAI)](https://pi-cai.grand-challenge.org/) Grand Challenge.

![Detection pipeline overview](detection-pipeline.png)
_Figure: overview of a lesion detection and evaluation pipeline._

Supported evaluation metrics:
- Average Precision (AP)
- Area Under the Receiver Operating Characteristic curve (AUROC)
- PI-CAI ranking metric: `(AUROC + AP) / 2`
- Precision-Recall (PR) curve
- Receiver Operating Characteristic (ROC) curve
- Free-Response Receiver Operating Characteristic (FROC) curve

Supported evaluation options:
- Case-wise sample weight (also applied to lesion-level evaluation, with same weight for all lesion candidates of the same case).
- Subset analysis by providing a list of case identifiers.

See [**Accessing metrics after evaluation**](#accessing-metrics-after-evaluation) to learn how to access these metrics.

## Installation
`picai_eval` is pip-installable:

`pip install git+https://github.com/DIAGNijmegen/picai_eval`


## Usage
The evaluation pipeline expects **detection maps** and **annotations** in the following format:
- detection maps: 3D volumes with connected components (in 3D) of the same _confidence_. Each detection map may contain an arbitrary number of connected components.
- annotations: 3D volumes with connected components (in 3D) with `1` as the target class and `0` as background.

Note: we define a _connected component_ as all non-zero voxels with _squared connectivity_ equal to two. This means that in a 3×3×3 neighbourhood all voxels are connected to the centre voxel, except for the eight voxels at the corners of the "cube".

### Evaluate samples with Python
To run the evaluation from Python, import the `evaluate` function and provide detection maps and annotations:

```python
from picai_eval import evaluate

subject_list = [
    "case-0",
    "case-1",
    "case-2",
]

metrics = evaluate(
    y_det=y_det,
    y_true=y_true,
    subject_list=subject_list,  # may be omitted
)
```

- `y_det` iterable of all detection_map volumes to evaluate. Each detection map should a 3D volume containing connected components (in 3D) of the same confidence. Each detection map may contain an arbitrary number of connected components, with different or equal confidences.
Alternatively, y_det may contain filenames ending in .nii.gz/.mha/.mhd/.npy/.npz, which will be loaded on-the-fly.
- `y_true`: iterable of all ground truth labels. Each label should be a 3D volume of the same shape as the corresponding detection map. Alternatively, `y_true` may contain filenames ending in .nii.gz/.mha/.mhd/.npy/.npz, which should contain binary labels and be loaded on-the-fly. Use `1` to encode ground truth lesion, and `0` to encode background.

The default parameters will perform the evaluation as used in the PI-CAI challenge. Optionally, the evaluation can be adapted using the following parameters:

- `sample_weight`: case-level sample weight. These weights will also be applied to the lesion-level evaluation, with same weight for all lesion candidates of the same case.
- `subject_list`: list of sample identifiers, to give recognizable names to the evaluation results.
- `min_overlap`: defines the minimal required Intersection over Union (IoU) or Dice similarity coefficient (DSC) between a lesion candidate and ground truth lesion, to be counted as a true positive detection. Default: 0.1.
- `overlap_func`: function to calculate overlap between a lesion candidate and ground truth mask. May be 'IoU' for Intersection over Union, or 'DSC' for Dice similarity coefficient. Alternatively, provide a function with signature `func(detection_map, annotation) -> overlap [0, 1]`. Default: IoU.
- `case_confidence`: function to derive case-level confidence from lesion-level confidences. Default: max.
- `multiple_lesion_candidates_selection_criteria`: when multiple lesion candidates have sufficient overlap with the ground truth lesion mask, this determines which lesion candidate is selected as TP. Default: overlap.
- `allow_unmatched_candidates_with_minimal_overlap`: when multiple lesion candidates have sufficient overlap with the ground truth lesion mask, this determines whether the lesion that is not selected counts as a false positive. Default: not counted as false positive.
- `num_parallel_calls`: number of threads to use for evaluation. Default: 8.


### Evaluate samples stored in a folder
To evaluate detection maps stored on disk, prepare the input folders in the following format:

```
path/to/detection_maps/
├── [case-0]_detection_map.nii.gz
├── [case-1]_detection_map.nii.gz
├── [case-2]_detection_map.nii.gz
...

path/to/annotations/
├── [case-0]_label.nii.gz
├── [case-1]_label.nii.gz
├── [case-2]_label.nii.gz
```

See [here](https://github.com/DIAGNijmegen/picai_eval/tree/public-release-prep/tests/test-maps) for an example. If the folders containing the detection maps and annotations are different, then the `_detection_map` and `_label` postfixes are optional. The allowed file extensions are: `.npz` from nnUNet, `.npy`, `.nii.gz`, `.nii`, `.mha` and `.mhd`. The first file matching one of these extensions (in the order shown here) is selected.

Evaluation of samples stored in a folder can be performed from Python:

```python
from picai_eval import evaluate_folder

subject_list = [
    "case-0",
    "case-1",
    "case-2",
]

metrics = evaluate_folder(
    y_det_dir="path/to/detection_maps",
    y_true_dir="path/to/annotations",
    subject_list=subject_list,  # may be omitted
)
```

This will evaluate the cases specified in `subject_list`. The `evaluate_folder` function also accepts all parameters described [above](#python).

To run the evaluation from the command line:

```bash
python -m picai_eval --input path/to/detection_maps --labels path/to/annotations
```

This will evaluate all cases found in `path/to/detection_maps` against the annotations in `path/to/annotations`, and store the metrics in `path/to/detection_maps/metrics.json`. Optionally, the `--labels` parameter may be omitted, which will default to the `--input` folder. To specify the output location of the metrics, use `--output /path/to/metrics.json`.

### Evaluate softmax predictions
To evaluate softmax predictions (instead of detection maps), the function to extract lesion candidates from the softmax prediction volume must be provided. The dynamic lesion extraction procedure from the [Report-Guided Annotation module](https://github.com/DIAGNijmegen/Report-Guided-Annotation) can be used for this (see [mechanism](https://github.com/DIAGNijmegen/Report-Guided-Annotation#mechanism) for a depiction of the dynamic lesion extraction procedure).

```python
from picai_eval import evaluate
from report_guided_annotation import extract_lesion_candidates

metrics = evaluate(
    y_det=y_pred,
    y_true=y_true,
    subject_list=subject_list,  # may be omitted
    y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0],
)
```

To evaluate a folder containing softmax predictions:

```python
from picai_eval import evaluate_folder
from report_guided_annotation import extract_lesion_candidates

metrics = evaluate_folder(
    y_det_dir=in_dir_softmax,
    y_true_dir=in_dir_annot,
    y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0],
)
```

## Storing and reading Metrics
Metrics can be easily saved and loaded to/from disk, to facilitate evaluation of multiple models, and subsequent (statistical) analysis. To read metrics, provide the path:

```python
from picai_eval import Metrics

metrics = Metrics("path/to/metrics.json")
```

To save metrics, provide the path to save the metrics to:

```python
metrics.save("path/to/metrics.json")
# metrics.save_full("path/to/metrics.json")  # also store derived curves
# metrics.save_minimal("path/to/metrics.json")  # only store minimal information to re-load Metrics
```

The command line interface described in [Evaluate samples stored in a folder](#evaluate-samples-stored-in-a-folder) will automatically save the metrics to disk. It's output path can be controlled with the `--output` parameter.

## Accessing metrics after evaluation
To access metrics after evaluation, we recommend using the `Metrics` class:

```python
metrics = ...  # from evaluate, evaluate_folder, or Metrics("/path/to/metrics.json")

# aggregate metrics
AP = metrics.AP
auroc = metrics.auroc
picai_score = metrics.score

# Precision-Recall (PR) curve
precision = metrics.precision
recall = metrics.recall

# Receiver Operating Characteristic (ROC) curve
tpr = metrics.case_TPR
fpr = metrics.case_FPR

# Free-Response Receiver Operating Characteristic (FROC) curve
sensitivity = metrics.lesion_TPR
fp_per_case = metrics.lesion_FPR
```

These can for example be used to plot the performance curves:

```python
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

# plot recision-Recall (PR) curve
disp = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=AP)
disp.plot()
plt.show()

# plot Receiver Operating Characteristic (ROC) curve
disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auroc)
disp.plot()
plt.show()

# plot Free-Response Receiver Operating Characteristic (FROC) curve
f, ax = plt.subplots()
disp = RocCurveDisplay(fpr=fp_per_case, tpr=sensitivity)
disp.plot(ax=ax)
ax.set_xlim(0.001, 5.0); ax.set_xscale('log')
ax.set_xlabel("False positives per case"); ax.set_ylabel("Sensitivity")
plt.show()
```

## Statistical tests
The PI-CAI challenge features AI vs AI, AI vs Radiologists from Clinical Routine and AI vs Panel of Readers. Each of these comparisons come with a statistical test. For AI vs AI, a permuations test with the ranking metric is performend. Readers cannot be assigned a ranking metric without introducing bias, so for AI vs Panel of Readers and AI vs Single Reader we compare performance at matched operating points. See each section below for more details.

**Note**: extended tests to verify if the statistical tests are well-calibrated (i.e., don't over- or underestimate the p-value), will be performed in the future.

### AI vs AI
**Comparison**: Between pairs of AI algorithms, with multiple restarts per AI algorithm.

**Statistical question**: What is the probability that one AI algorithm outperforms another, while accounting for the performance variance stemming from each AI algorithm’s training method?

**Statistical test**: Permutation tests (as applied in Ruamviboonsuk et al., 2022, Bulten et al., 2022, McKinney et al., 2020). In each replication, performance metrics (ranking score, AP or AUROC) are shuffled across methods (different AI algorithms) and their instances (independently trained samples of each method).

The permutation test using the performance metrics can be used as follows:

```python
from picai_eval.statistical_helper import perform_permutation_test

scores_algorithm_a = [0.92, 0.94, 0.95, 0.81, 0.82, 0.86]
scores_algorithm_b = [0.96, 0.91, 0.90, 0.85, 0.81, 0.80]

# perform multiple permutation tests
p = perform_permutation_test(
    scores_alternative=metrics_algorithm_a,
    scores_baseline=scores_algorithm_b,
)

# p-value should be 0.7218614718614719
```

This will calculate the p-value for the null hypothesis _Performance(baseline algorithm) > Performance(alternative algorithm)_ (given the evidence of the provided performance metrics). The scores shown above (0.92, 0.94, etc.) are **performance metrics**, not model predictions. These could for example be obtained from the evaluation pipeline:

```python
from picai_eval import Metrics

scores_algorithm = [
    Metrics(path).score
    for path in [
        "/path/to/algorithm/metrics-restart-1.json",
        "/path/to/algorithm/metrics-restart-2.json",
        "/path/to/algorithm/metrics-restart-3.json",
        ...
    ]
]
```

### AI vs Radiologists from Clinical Routine
**Comparison**: Between AI algorithm, and the historical reads made by radiologists during clinical routine.

**Statistical question**: What is the probability that a given trained AI algorithm outperforms radiologists from clinical routine, while accounting for the performance variance stemming from different cases and the AI algorithm’s training method? 

**Statistical test**: Paired bootstrapping (as applied in Ruamviboonsuk et al., 2022, McKinney et al., 2020, Rodriguez-Ruiz et al., 2019), using predictions from a given operating point. Here, the operating point is that of radiologists (PI-RADS ≥ 3 or PI-RADS ≥ 4) from clinical routine. Trained AI algorithms are thresholded to match the radiologist's sensitivity/specificity (for patient diagnosis) or recall/precision (for lesion detection). In each of 1M replications, ∼U(0,N) cases are sampled with replacement, and used to calculate the _test statistic_. Iterations that sample only one class are rejected. The test statistic is the rank of historical reads made by radiologists, with respect to the predictions made by trained AI algorithms, where the rank is determined by the conjugate performance metric.

**Note**: In contrast to the [permutation test](#ai-vs-ai), bootstrapping approximates the statistical question. As a result, the p-value from bootstrapping can be mis-calibrated, giving p-values that are higher or lower than they should be. The permutation test does not have this issue, but cannot be applied in this scenario, because we have only a single radiological prediction per case.

The matched bootstrapping test can be used as follows:

```python
import numpy as np
from picai_eval.statistical_helper import perform_matched_boostrapping

# predictions: 3 restarts (rows) of 4 cases (columns)
y_pred_ai = [
    [0.92, 0.23, 0.12, 0.95],
    [0.42, 0.81, 0.13, 0.86],
    [0.26, 0.15, 0.14, 0.67]
]
y_pred_reader = np.array([5, 4, 2, 3]) >= 3
y_true = [1, 1, 0, 0]

# perform matched bootstrapping
p = perform_matched_boostrapping(
    y_true=y_true,
    y_pred_ai=y_pred_ai,
    y_pred_reader=y_pred_reader,
    match='sensitivity',
    iterations=int(1e4),
)

# Probability for Performance(AI) > Performance(Reader): p = 0.3 (approximately)
```

The predictions shown above (0.92, 0.23, ..., 0.95 and 5, 4, ..., 2 etc.) are **predictions**, not performance metrics. The predictions for the historical read should be thresholded (e.g. PI-RADS ≥ 3 or PI-RADS ≥ 4), while the predictions for the algorithm should be confidences between 0 and 1. These could for example be obtained from the evaluation pipeline:

```python
from picai_eval import Metrics

y_pred_ai = [
    Metrics(path).case_pred
    for path in [
        "/path/to/algorithm/metrics-restart-1.json",
        "/path/to/algorithm/metrics-restart-2.json",
        "/path/to/algorithm/metrics-restart-3.json",
        ...
    ]
]
```

### AI vs Radiologists from Reader Study
**Comparison**: Between AI algorithm, and a given panel of readers.

**Statistical question**: What is the probability that a given AI algorithm outperforms the typical reader from a given panel of radiologists, while accounting for the performance variance stemming from different readers, and the AI algorithm’s training method?

**Statistical test**: Permutation test (as applied in Ruamviboonsuk et al., 2022, Bulten et al., 2022, McKinney et al., 2020). Permutation tests are used to statistically compare lesion-level detection and patient-level diagnosis performance at PI-RADS operating points. Here, in each of the replications, performance metrics (reader performance w.r.t. AI performance at reader’s operating point) are shuffled across methods (AI, radiologists) and their instances (independently trained samples of AI algorithm, different readers).

```python
import numpy as np
from picai_eval.statistical_helper import perform_matched_permutation_test

# predictions: 3 restarts (rows) of 4 cases (columns)
y_pred_ai = [
    [0.92, 0.23, 0.12, 0.95],
    [0.82, 0.81, 0.13, 0.42],
    [0.26, 0.90, 0.14, 0.67]
]
y_pred_readers = np.array([
    [5, 4, 2, 2],
    [4, 5, 1, 2],
    [5, 2, 3, 2]
]) >= 3
y_true = [1, 1, 0, 0]

p = perform_matched_permutation_test(
    y_true=y_true,
    y_pred_ai=y_pred_ai,
    y_pred_readers=y_pred_readers,
    match="sensitivity",
    iterations=int(1e4),
)

# Probability for Performance(Panel of readers) > Performance(AI): p = 0.8 (approximately)
```

The predictions shown above (0.92, 0.23, ..., 0.95 and 5, 4, ..., 2 etc.) are **predictions**, not performance metrics. The predictions for the readers should be thresholded (e.g. PI-RADS ≥ 3 or PI-RADS ≥ 4), while the predictions for the algorithm should be confidences between 0 and 1. These could for example be obtained from the evaluation pipeline:

```python
from picai_eval import Metrics

y_pred_ai = [
    Metrics(path).case_pred
    for path in [
        "/path/to/algorithm/metrics-restart-1.json",
        "/path/to/algorithm/metrics-restart-2.json",
        "/path/to/algorithm/metrics-restart-3.json",
        ...
    ]
]
```
