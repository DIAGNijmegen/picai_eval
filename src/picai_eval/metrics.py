#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import division, print_function

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Tuple, Union

import numpy as np
import sklearn
from packaging import version
from sklearn.metrics import auc, precision_recall_curve, roc_curve

try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass

from picai_eval.data_utils import PathLike, load_metrics, save_metrics


@dataclass
class Metrics:
    lesion_results: Union[Dict[Hashable, List[Tuple[int, float, float]]], PathLike]
    case_target: Optional[Dict[Hashable, int]] = None
    case_pred: Optional[Dict[Hashable, float]] = None
    case_weight: Optional[Union[Dict[Hashable, float], List[float]]] = None
    lesion_weight: Optional[Dict[Hashable, List[float]]] = None
    thresholds: "Optional[npt.NDArray[np.float64]]" = None
    subject_list: Optional[List[str]] = None
    sort: bool = True

    def __post_init__(self):
        if isinstance(self.lesion_results, (str, Path)):
            # load metrics from file
            self.load(self.lesion_results)

        if self.subject_list is None:
            self.subject_list = sorted(list(self.lesion_results))

        if self.case_target is None:
            # derive case-level targets as the maximum lesion-level target
            self.case_target = {
                idx: max([is_lesion for is_lesion, _, _ in case_y_list]) if len(case_y_list) else 0
                for idx, case_y_list in self.lesion_results.items()
            }

        if self.case_pred is None:
            # derive case-level predictions as the maximum lesion-level prediction
            self.case_pred = {
                idx: max([confidence for _, confidence, _ in case_y_list]) if len(case_y_list) else 0
                for idx, case_y_list in self.lesion_results.items()
            }

        if not isinstance(self.case_weight, dict):
            subject_list = list(self.case_target)
            if self.case_weight is None:
                self.case_weight = {idx: 1 for idx in subject_list}
            else:
                self.case_weight = {idx: weight for idx, weight in zip(subject_list, self.case_weight)}

        if self.lesion_weight is None:
            subject_list = sorted(list(self.lesion_results))
            self.lesion_weight = {idx: [1]*len(case_y_list) for idx, case_y_list in self.lesion_results.items()}

        if self.sort:
            # sort dictionaries
            subject_list = sorted(list(self.lesion_results))
            self.lesion_results = {idx: self.lesion_results[idx] for idx in subject_list}
            self.lesion_weight = {idx: self.lesion_weight[idx] for idx in subject_list}
            self.case_target = {idx: self.case_target[idx] for idx in subject_list}
            self.case_pred = {idx: self.case_pred[idx] for idx in subject_list}
            self.case_weight = {idx: self.case_weight[idx] for idx in subject_list}

    # aggregates
    def calc_auroc(self, subject_list: Optional[List[str]] = None) -> float:
        """Calculate case-level Area Under the Receiver Operating Characteristic curve (AUROC)"""
        return self.calculate_ROC(subject_list=subject_list)['AUROC']

    @property
    def auroc(self) -> float:
        """Calculate case-level Area Under the Receiver Operating Characteristic curve (AUROC)"""
        return self.calc_auroc()

    def calc_AP(self, subject_list: Optional[List[str]] = None) -> float:
        """Calculate Average Precision"""
        return self.calculate_precision_recall(subject_list=subject_list)['AP']

    @property
    def AP(self) -> float:
        """Calculate Average Precision"""
        return self.calc_AP()

    @property
    def num_cases(self) -> int:
        """Calculate the number of cases"""
        return len(self.subject_list)

    @property
    def num_lesions(self) -> int:
        """Calculate the number of ground truth lesions"""
        return sum([is_lesion for is_lesion, *_ in self.lesion_results_flat])

    @property
    def score(self):
        """Calculate the ranking score, as used in the PI-CAI 22 Grand Challenge"""
        return (self.auroc + self.AP) / 2

    def lesion_TPR_at_FPR(self, FPR: float) -> float:
        """Calculate the lesion-level true positive rate (sensitivity) at a given
        false positive rate (average number of false positives per examimation)"""
        if np.min(self.lesion_FPR) > FPR:
            return 0
        return self.lesion_TPR[self.lesion_FPR <= FPR][-1]

    # lesion-level results
    def get_lesion_results_flat(self, subject_list: Optional[List[str]] = None):
        """Flatten the per-case lesion evaluation results into a single list"""
        if subject_list is None:
            subject_list = self.subject_list

        return [
            (is_lesion, confidence, overlap)
            for subject_id in subject_list
            for is_lesion, confidence, overlap in self.lesion_results[subject_id]
        ]

    @property
    def lesion_results_flat(self) -> List[Tuple[int, float, float]]:
        """Flatten the per-case y_list"""
        return self.get_lesion_results_flat()

    def get_lesion_weight_flat(self, subject_list: Optional[List[str]] = None) -> List[float]:
        """Retrieve lesion-wise sample weights (for a given subset of cases)"""
        if subject_list is None:
            subject_list = self.subject_list

        # collect lesion weights (and flatten)
        return [weight for subject_id in subject_list for weight in self.lesion_weight[subject_id]]

    @property
    def lesion_weight_flat(self) -> List[float]:
        """Retrieve lesion-wise sample weights (for a given subset of cases)"""
        return self.get_lesion_weight_flat()

    @property
    def precision(self) -> "npt.NDArray[np.float64]":
        """Calculate lesion-level precision at each threshold"""
        return self.calculate_precision_recall()['precision']

    @property
    def recall(self) -> "npt.NDArray[np.float64]":
        """Calculate lesion-level recall at each threshold"""
        return self.calculate_precision_recall()['recall']

    @property
    def lesion_TP(self) -> "npt.NDArray[np.float64]":
        """Calculate number of true positive lesion detections at each threshold"""
        return self.calculate_counts()['TP']

    @property
    def lesion_FP(self) -> "npt.NDArray[np.float64]":
        """Calculate number of false positive lesion detections at each threshold"""
        return self.calculate_counts()['FP']

    @property
    def lesion_TPR(self) -> "npt.NDArray[np.float64]":
        """Calculate lesion-level true positive rate (sensitivity) at each threshold"""
        if self.num_lesions > 0:
            return self.lesion_TP / self.num_lesions
        else:
            return np.array([np.nan] * len(self.lesion_TP))

    @property
    def lesion_FPR(self) -> "npt.NDArray[np.float64]":
        """Calculate lesion-level false positive rate (number of false positives per case) at each threshold"""
        return self.lesion_FP / self.num_cases

    # case-level results
    def calc_case_TPR(self, subject_list: Optional[List[str]] = None) -> "npt.NDArray[np.float64]":
        """Calculate case-level true positive rate (sensitivity) at each threshold"""
        return self.calculate_ROC(subject_list=subject_list)['TPR']

    @property
    def case_TPR(self) -> "npt.NDArray[np.float64]":
        """Calculate case-level true positive rate (sensitivity) at each threshold"""
        return self.calc_case_TPR()

    def calc_case_FPR(self, subject_list: Optional[List[str]] = None) -> "npt.NDArray[np.float64]":
        """Calculate case-level false positive rate (1 - specificity) at each threshold"""
        return self.calculate_ROC(subject_list=subject_list)['FPR']

    @property
    def case_FPR(self) -> "npt.NDArray[np.float64]":
        """Calculate case-level false positive rate (1 - specificity) at each threshold"""
        return self.calc_case_FPR()

    # supporting functions
    def calculate_counts(self, subject_list: Optional[List[str]] = None) -> "Dict[str, npt.NDArray[np.float32]]":
        """
        Calculate lesion-level true positive (TP) detections and false positive (FP) detections as each threshold.
        """
        # flatten y_list (and select cases in subject_list)
        lesion_y_list = self.get_lesion_results_flat(subject_list=subject_list)

        # collect targets and predictions
        y_true: "npt.NDArray[np.float64]" = np.array([target for target, *_ in lesion_y_list])
        y_pred: "npt.NDArray[np.float64]" = np.array([pred for _, pred, *_ in lesion_y_list])

        if self.thresholds is None:
            # collect thresholds for lesion-based analysis
            self.thresholds = np.unique(y_pred)
            self.thresholds[::-1].sort()  # sort thresholds in descending order (inplace)

            # for >10,000 thresholds: resample to 10,000 unique thresholds, while also
            # keeping all thresholds higher than 0.8 and the first 20 thresholds
            if len(self.thresholds) > 10_000:
                rng = np.arange(1, len(self.thresholds), len(self.thresholds)/10_000, dtype=np.int32)
                st = [self.thresholds[i] for i in rng]
                low_thresholds = self.thresholds[-20:]
                self.thresholds = np.array([t for t in self.thresholds if t > 0.8 or t in st or t in low_thresholds])

        # define placeholders
        FP: "npt.NDArray[np.float32]" = np.zeros_like(self.thresholds, dtype=np.float32)
        TP: "npt.NDArray[np.float32]" = np.zeros_like(self.thresholds, dtype=np.float32)

        # for each threshold: count FPs and TPs
        for i, th in enumerate(self.thresholds):
            y_pred_thresholded = (y_pred >= th).astype(int)
            tp = np.sum(y_true*y_pred_thresholded)
            fp = np.sum(y_pred_thresholded - y_true*y_pred_thresholded)

            # update with new point
            FP[i] = fp
            TP[i] = tp

        # extend curve to infinity
        TP[-1] = TP[-2]
        FP[-1] = np.inf

        return {
            'TP': TP,
            'FP': FP,
        }

    def calculate_precision_recall(self, subject_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate Precision-Recall curve and calculate average precision (AP).
        """
        # flatten y_list (and select cases in subject_list)
        lesion_y_list = self.get_lesion_results_flat(subject_list=subject_list)

        # collect targets and predictions
        y_true: "npt.NDArray[np.float64]" = np.array([target for target, *_ in lesion_y_list])
        y_pred: "npt.NDArray[np.float64]" = np.array([pred for _, pred, *_ in lesion_y_list])

        # calculate precision-recall curve
        if version.parse(sklearn.__version__) >= version.parse("1.5"):
            # in the future this if/else block can be removed, then set 1.5 as minimum in requirements.txt
            precision, recall, thresholds = precision_recall_curve(
                y_true=y_true,
                y_score=y_pred,
                sample_weight=self.get_lesion_weight_flat(subject_list=subject_list)
            )
        else:
            precision, recall, thresholds = precision_recall_curve(
                y_true=y_true,
                probas_pred=y_pred,
                sample_weight=self.get_lesion_weight_flat(subject_list=subject_list)
            )

        # set precision to zero at a threshold of "zero", as those lesion
        # candidates are included just to convey the number of lesions to
        # the precision_recall_curve function, and not actual candidates
        precision[:-1][thresholds == 0] = 0

        # calculate average precision using the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve
        # Taken from https://github.com/scikit-learn/scikit-learn/blob/
        # 32f9deaaf27c7ae56898222be9d820ba0fd1054f/sklearn/metrics/_ranking.py#L212
        AP = -np.sum(np.diff(recall) * np.array(precision)[:-1])

        return {
            'AP': AP,
            'precision': precision,
            'recall': recall,
        }

    def calculate_ROC(self, subject_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate Receiver Operating Characteristic curve for case-level risk stratification.
        """
        if subject_list is None:
            subject_list = self.subject_list

        fpr, tpr, _ = roc_curve(
            y_true=[self.case_target[s] for s in subject_list],
            y_score=[self.case_pred[s] for s in subject_list],
            sample_weight=[self.case_weight[s] for s in subject_list],
        )

        auroc = auc(fpr, tpr)

        return {
            'FPR': fpr,
            'TPR': tpr,
            'AUROC': auroc,
        }

    @property
    def version(self):
        return "1.4.x"

    def as_dict(self):
        return {
            # aggregates
            "auroc": self.auroc,
            "AP": self.AP,
            "num_cases": self.num_cases,
            "num_lesions": self.num_lesions,
            "picai_eval_version": self.version,

            # lesion-level results
            "lesion_results": self.lesion_results,
            "lesion_weight": self.lesion_weight,

            # case-level results
            "case_pred": self.case_pred,
            "case_target": self.case_target,
            "case_weight": self.case_weight,
        }

    def full_dict(self):
        return {
            # aggregates
            "auroc": self.auroc,
            "AP": self.AP,
            "num_cases": self.num_cases,
            "num_lesions": self.num_lesions,
            "picai_eval_version": self.version,

            # lesion-level results
            "lesion_results": self.lesion_results,
            "lesion_weight": self.lesion_weight,
            "precision": self.precision,
            "recall": self.recall,
            'lesion_TPR': self.lesion_TPR,
            'lesion_FPR': self.lesion_FPR,
            "thresholds": self.thresholds,

            # case-level results
            "case_pred": self.case_pred,
            "case_target": self.case_target,
            "case_weight": self.case_weight,
        }

    def minimal_dict(self):
        return {
            # lesion-level results
            "lesion_results": self.lesion_results,
            "lesion_weight": self.lesion_weight,

            # case-level results
            "case_pred": self.case_pred,
            "case_target": self.case_target,
            "case_weight": self.case_weight,
        }

    def save(self, path: PathLike):
        """Save metrics to file (including aggregates)"""
        save_metrics(metrics=self.as_dict(), file_path=path)

    def save_full(self, path: PathLike):
        """Save metrics to file (including derived metrics)"""
        save_metrics(metrics=self.full_dict(), file_path=path)

    def save_minimal(self, path: PathLike):
        """Save metrics to file (minimal required metrics)"""
        save_metrics(metrics=self.minimal_dict(), file_path=path)

    def load(self, path: PathLike):
        """Load metrics from file"""
        metrics = load_metrics(path)

        # parse metrics
        self.case_target = {idx: int(float(val)) for idx, val in metrics['case_target'].items()}
        self.case_pred = {idx: float(val) for idx, val in metrics['case_pred'].items()}
        self.case_weight = {idx: float(val) for idx, val in metrics['case_weight'].items()}
        self.lesion_weight = {idx: [float(val) for val in weights] for idx, weights in metrics['lesion_weight'].items()}
        self.lesion_results = {
            idx: [
                (int(float(is_lesion)), float(confidence), float(overlap))
                for (is_lesion, confidence, overlap) in lesion_results_case
            ]
            for idx, lesion_results_case in metrics['lesion_results'].items()
        }

    def __str__(self) -> str:
        return f"Metrics(auroc={self.auroc:.2%}, AP={self.AP:.2%}, {self.num_cases} cases, {self.num_lesions} lesions)"

    def __repr__(self) -> str:
        return self.__str__()
