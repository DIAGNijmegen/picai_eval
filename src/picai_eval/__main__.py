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

import argparse
import json
import os

from report_guided_annotation import extract_lesion_candidates

from picai_eval import evaluate_folder

# acquire and parse input and output paths
parser = argparse.ArgumentParser(description='Command Line Arguments')
parser.add_argument("-i", "--input", type=str, required=True,
                    help="Path to folder with model predicitons (detection maps)")
parser.add_argument("-l", "--labels", type=str, required=False,
                    help="Path to folder with labels (defaults to input folder if unspecified)")
parser.add_argument("-o", "--output", type=str, default="metrics.json",
                    help="Path to store metrics file, relative to the input folder.")
parser.add_argument("-s", "--subject_list", type=str, required=False,
                    help="Path to subject list, relative to the input folder. The subject list " +
                         "may be stored as json list, or json dictionary with 'subject_list' as parameter.")
parser.add_argument("--pred_extensions", type=str, nargs="+", required=False,
                    help="List of allowed file formats for detection maps." +
                         "Default: .npz, .npy, .nii.gz, .nii, .mha and .mhd")
parser.add_argument("--label_extensions", type=str, nargs="+", required=False,
                    help="List of allowed file formats for annotations." +
                         "Default: .nii.gz, .nii, .mha, .mhd, .npz and .npy")
parser.add_argument("--y_det_postprocess_func", type=str, required=False,
                    help="Post-processing function for detection maps. Available: `extract_lesion_candidates`")
parser.add_argument("--y_det_postprocess_kwargs", type=str, required=False,
                    help='Post-processing arguments for detection maps. E.g.: `{"threshold": "dynamic"}`')
args = parser.parse_args()

if args.labels is None:
    args.labels = args.input
args.output = os.path.join(args.input, args.output)
if args.subject_list is not None:
    args.subject_list = os.path.join(args.input, args.subject_list)
    with open(args.subject_list) as fp:
        args.subject_list = json.load(fp)
    if isinstance(args.subject_list, dict):
        args.subject_list = args.subject_list['subject_list']

if args.y_det_postprocess_func is not None:
    if args.y_det_postprocess_func == "extract_lesion_candidates":
        if args.y_det_postprocess_kwargs is None:
            args.y_det_postprocess_kwargs = {}
        else:
            args.y_det_postprocess_kwargs = json.loads(args.y_det_postprocess_kwargs)
        args.y_det_postprocess_func = lambda pred: extract_lesion_candidates(pred, **args.y_det_postprocess_kwargs)[0]
    else:
        raise ValueError(f"Received unsupported post-processing function: {args.y_det_postprocess_func}")

print(f"""
    PICAI Evaluation
    Model predictions path: {args.input}
    Labels path: {args.labels}
    Output Metrics Path: {args.output}
""")

assert os.path.exists(args.input), f"Input folder does not exist at {args.input}!"
assert os.path.exists(os.path.dirname(args.output)), f"Output folder does not exist at {args.output}!"

# calculate metrics
metrics = evaluate_folder(
    y_det_dir=args.input,
    y_true_dir=args.labels,
    subject_list=args.subject_list,
    pred_extensions=args.pred_extensions,
    label_extensions=args.label_extensions,
    y_det_postprocess_func=args.y_det_postprocess_func,
)

# show metrics
print("Metrics:")
print(metrics)

# save metrics
metrics.save(path=args.output)
