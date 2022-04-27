# Test detection maps and labels

- Case 0: no ground truth lesions, no detection maps.
- Case 1: single ground truth lesion, no detection maps. 1 FN.
- Case 2: no ground truth lesion, single ground truth lesion. 1 FP.
- Case 3: single ground truth lesion, single detection map. Candidate 1: IoU = 0. 1 FN + 1 FP.
- Case 4: single ground truth lesion, single detection map. Candidate 1: IoU = 1/3. 1 TP.
- Case 5: single ground truth lesion, two detection maps. Candidate 1: IoU = 9/41, confidence = 1. Candidate 2: IoU = 3/47, confidence = 2. 1 TP + 1 FP.
- Case 6: single ground truth lesion, two detection maps. Candidate 1: IoU = 1/3, confidence = 1. Candidate 2: IoU = 2/7, confidence = 2. 1 TP.
- Case 7: two ground truth lesion, single detection map. Candidate 1: IoU = 1/6 with either ground truth lesion. 1 TP + 1 FN.
- Case 8: two ground truth lesion, single detection map. Candidate 1: IoU = 5/11 with one ground truth lesion, 1/11 with other ground truth lesion. 1 TP + 1 FN.
- Case 9: label with two detection maps. Candidate 1: IoU = 1/3, confidence = 1. Candidate 2: IoU = 1/3, confidence = 2. Full empty slice between candidates.