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

from pathlib import Path

import numpy as np
import SimpleITK as sitk

try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass

from picai_eval.data_utils import PathLike


def resize_image_with_crop_or_pad(image, img_size=(64, 64, 64), **kwargs):
    """
    Resize images (scans/predictions/labels) by cropping and/or padding
    Adapted from: https://github.com/DLTK/DLTK]
    """
    assert isinstance(image, np.ndarray)
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        "Target size doesn't fit image size"

    rank = len(img_size)  # image dimensions

    # placeholders for new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for _ in range(rank)]
    slicer = [slice(None)] * rank

    # for each dimension, determine process (cropping or padding)
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # create slicer object to crop/leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # pad cropped image to extend missing dimension
    return np.pad(image[tuple(slicer)], to_padding, **kwargs)


def read_image(path: PathLike):
    """Read image, given a filepath"""
    if isinstance(path, Path):
        path = path.as_posix()
    else:
        assert isinstance(path, str), f"Unexpected path type: {type(path)}. Please provide a Path or str."

    if '.npy' in path:
        return np.load(path)
    elif '.nii' in path or '.mha' in path or 'mhd' in path:
        return sitk.GetArrayFromImage(sitk.ReadImage(path))
    elif '.npz' in path:
        # read the nnU-Net format
        data = np.load(path)
        data = data["softmax"] if "softmax" in data else data["probabilities"]
        return data.astype("float32")[1]
    else:
        raise ValueError(f"Unexpected file path. Supported file formats: .nii(.gz), .mha, .npy and .npz. Got: {path}.")


def read_prediction(path: PathLike) -> "npt.NDArray[np.float32]":
    """Read prediction, given a filepath"""
    # read prediction and ensure correct dtype
    pred: "npt.NDArray[np.float32]" = np.array(read_image(path), dtype=np.float32)
    return pred


def read_label(path: PathLike) -> "npt.NDArray[np.int32]":
    """Read label, given a filepath"""
    # read label and ensure correct dtype
    lbl: "npt.NDArray[np.int32]" = np.array(read_image(path), dtype=np.int32)
    return lbl
