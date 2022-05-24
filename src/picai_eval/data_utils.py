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

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar

import numpy as np

PathLike = TypeVar("PathLike", str, Path)


def save_metrics(metrics, file_path: PathLike):
    """Save metrics to disk"""
    # convert dtypes to stock Python
    save_metrics = sterilize(metrics)

    # save metrics using safe file write
    file_path_tmp = str(file_path) + '.tmp'
    with open(file_path_tmp, 'w') as fp:
        json.dump(save_metrics, fp, indent=4)
    os.rename(file_path_tmp, file_path)


def load_metrics(file_path: PathLike):
    """Read metrics from disk"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metrics not found at {file_path}!")

    with open(file_path) as fp:
        metrics = json.load(fp)

    return metrics


def sterilize(obj):
    """Prepare object for conversion to json"""
    if isinstance(obj, dict):
        return {k: sterilize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, np.ndarray)):
        return [sterilize(v) for v in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (str, int, bool, float)):
        return obj
    else:
        return obj.__repr__()


def get_dataset_config(
    name: PathLike,
    split: Optional[str] = None,
    fold: Optional[int] = None,
    config_root: PathLike = "/input/dataset-configs",
) -> Dict[str, Any]:
    """
    Read dataset configuration

    Returns:
    - dataset_config: {
        'subject_list': [subject_id1, subject_id2, ...],
        (optional) 'labels': {
            (optional) 'label_name1': {
                subject_id1: 0/1,
                ...
            },
            ...
        }
    }
    """
    config_root = Path(config_root)
    if split is None:
        split = "all"

    if split in ['test', 'all']:
        postfix = ""
    else:
        postfix = f"-fold-{fold}" if fold is not None else ""

    path = config_root / name / f"ds-config-{split}{postfix}.json"
    with open(path) as fp:
        dataset_config = json.load(fp)

    return dataset_config


def get_subject_list(
    name: str,
    split: Optional[str] = None,
    fold: Optional[int] = None,
    config_root: str = "/input/dataset-configs",
    construct_all_from_train=True
) -> Dict[str, Any]:
    """Get subject list from a dataset configuration"""
    try:
        dataset_config = get_dataset_config(name=name, split=split, fold=fold, config_root=config_root)
        return dataset_config['subject_list']
    except FileNotFoundError:
        # if trying to read 'all', construct from train & val splits (assuming 5-fold cross-validation)
        if split == 'all' and construct_all_from_train:
            subject_list_all = []
            for fold in range(5):
                subject_list_all += get_subject_list(name=name, split='train', fold=fold, config_root=config_root)
            subject_list_all = sorted(list(set(subject_list_all)))
            return subject_list_all
        else:
            # could not resolve error, re-raise it
            raise
