import os
import pandas as pd

from model_utils import predict_xray

CSV_PATH = "data/Data_Entry_2017.csv"
IMAGE_FOLDER = "data/images"
OUTPUT_CSV = "evaluation_results.csv"
SUMMARY_PATH = "evaluation_report.txt"

LABEL_MAPPING = {
    "Lung Opacity": [
        "Lung Opacity",
        "Infiltration",
        "Pneumonia",
        "Consolidation"
    ],
    "Lung Lesion": [
        "Lung Lesion",
        "Mass",
        "Nodule"
    ],
    "Enlarged Cardiomediastinum": [
        "Enlarged Cardiomediastinum",
        "Cardiomegaly"
    ],
    "Cardiomegaly": [
        "Cardiomegaly",
        "Enlarged Cardiomediastinum"
    ],
    "Mass": [
        "Mass",
        "Lung Lesion",
        "Nodule"
    ],
    "Nodule": [
        "Nodule",
        "Mass",
        "Lung Lesion"
    ],
    "Infiltration": [
        "Infiltration",
        "Lung Opacity",
        "Pneumonia",
        "Consolidation"
    ],
    "Pneumonia": [
        "Pneumonia",
        "Infiltration",
        "Lung Opacity",
        "Consolidation"
    ],
    "Consolidation": [
        "Consolidation",
        "Infiltration",
        "Lung Opacity",
        "Pneumonia"
    ],
    "Effusion": [
        "Effusion"
    ],
    "Atelectasis": [
        "Atelectasis"
    ],
    "Pneumothorax": [
        "Pneumothorax"
    ],
    "Emphysema": [
        "Emphysema"
    ],
    "Pleural Thickening": [
        "Pleural Thickening"
    ],
    "Fracture": [
        "Fracture"
    ],
    "Edema": [
        "Edema"
    ],
    "Fibrosis": [
        "Fibrosis",
        "Interstitial Lung Disease"
    ],
    "Interstitial Lung Disease": [
        "Interstitial Lung Disease",
        "Fibrosis"
    ]
}

NO_FINDING_GROUND_TRUTH = {"No Finding", "No Findings"}

print("Loading labels...")

df = pd.read_csv(CSV_PATH)

evaluation_rows = []

images_evaluated = 0
correct_top1 = 0
correct_top5 = 0
correct_thresholded = 0
incorrect_matches = 0
mapped_matches = 0
direct_matches = 0

thresholded_tp = 0
thresholded_fp = 0
thresholded_fn = 0


def get_equivalent_labels(prediction):
    equivalent = LABEL_MAPPING.get(prediction, [prediction])
    return set(label.strip() for label in equivalent if label and isinstance(label, str))


def is_direct_match(prediction, truth_set):
    return prediction in truth_set


def is_match(prediction, truth_set):
    if is_direct_match(prediction, truth_set):
        return True
    equivalents = get_equivalent_labels(prediction)
    return bool(equivalents.intersection(truth_set))


def is_truth_matched(truth_label, predicted_labels):
    if truth_label in NO_FINDING_GROUND_TRUTH:
        return not predicted_labels
    for pred in predicted_labels:
        if truth_label == pred or truth_label in get_equivalent_labels(pred):
            return True
    return False


def thresholded_match_type(predicted_labels, truth_set):
    if not predicted_labels and truth_set & NO_FINDING_GROUND_TRUTH:
        return "No Finding"

    direct = any(pred in truth_set for pred in predicted_labels)
    mapped = any(
        not is_direct_match(pred, truth_set)
        and is_match(pred, truth_set)
        for pred in predicted_labels
    )

    if direct:
        return "Direct"
    if mapped:
        return "Mapped"
    return "No Match"

for filename in sorted(os.listdir(IMAGE_FOLDER)):

    if not filename.lower().endswith(".png"):
        continue

    if "(1)" in filename:
        continue

    row = df[df["Image Index"] == filename]
    if row.empty:
        continue

    actual_labels = row.iloc[0]["Finding Labels"]
    if pd.isna(actual_labels):
        continue

    ground_truth = [label.strip() for label in actual_labels.split("|") if label.strip()]
    if not ground_truth:
        continue

    truth_set = set(ground_truth)
    image_path = os.path.join(IMAGE_FOLDER, filename)

    try:
        prediction = predict_xray(image_path)
        top_prediction = prediction["predictions"][0][0]
        top5_predictions = [label for label, score in prediction["predictions"]]
        thresholded_predictions = prediction["selected_findings"]

        top1_match = is_match(top_prediction, truth_set)
        top1_direct = is_direct_match(top_prediction, truth_set)

        top5_direct = False
        top5_mapped = False
        for pred in top5_predictions:
            if is_direct_match(pred, truth_set):
                top5_direct = True
                break
            if is_match(pred, truth_set):
                top5_mapped = True

        top5_match = top5_direct or top5_mapped

        thresholded_match = False
        if not thresholded_predictions and truth_set & NO_FINDING_GROUND_TRUTH:
            thresholded_match = True
        elif any(is_match(pred, truth_set) for pred in thresholded_predictions):
            thresholded_match = True

        if top1_match:
            correct_top1 += 1

        if top5_match:
            correct_top5 += 1

        if thresholded_match:
            correct_thresholded += 1
            match_label = "Yes"
            match_type = thresholded_match_type(thresholded_predictions, truth_set)
            if match_type == "Direct":
                direct_matches += 1
            elif match_type == "Mapped":
                mapped_matches += 1
        else:
            incorrect_matches += 1
            match_label = "No"
            match_type = "No Match"

        images_evaluated += 1

        for pred in thresholded_predictions:
            if is_match(pred, truth_set):
                thresholded_tp += 1
            else:
                thresholded_fp += 1

        positive_truth_labels = [
            label for label in truth_set
            if label not in NO_FINDING_GROUND_TRUTH
        ]
        for truth_label in positive_truth_labels:
            if not is_truth_matched(truth_label, thresholded_predictions):
                thresholded_fn += 1

        evaluation_rows.append({
            "Image Name": filename,
            "Ground Truth Labels": actual_labels,
            "Top Prediction": top_prediction,
            "Top-5 Predictions": ", ".join(top5_predictions),
            "Thresholded Predictions": ", ".join(thresholded_predictions) if thresholded_predictions else "None",
            "Top-1 Match": "Yes" if top1_match else "No",
            "Top-5 Match": "Yes" if top5_match else "No",
            "Thresholded Match": match_label,
            "Thresholded Match Type": match_type
        })

        print(f"\n{filename}")
        print(f"Ground Truth: {actual_labels}")
        print(f"Top Prediction: {top_prediction}")
        print(f"Top-5 Predictions: {', '.join(top5_predictions)}")
        print(f"Thresholded Predictions: {', '.join(thresholded_predictions) if thresholded_predictions else 'None'}")
        print(f"Top-1 Match: {'Yes' if top1_match else 'No'}")
        print(f"Top-5 Match: {'Yes' if top5_match else 'No'}")
        print(f"Thresholded Match: {match_label} ({match_type})")

    except Exception as e:
        print(f"Error processing {filename}")
        print(e)

# Save results CSV
if evaluation_rows:
    pd.DataFrame(evaluation_rows).to_csv(OUTPUT_CSV, index=False)

# Build summary report
with open(SUMMARY_PATH, "w", encoding="utf-8") as summary_file:
    summary_file.write("MODEL EVALUATION REPORT\n")
    summary_file.write("=======================\n")
    summary_file.write(f"Images Evaluated: {images_evaluated}\n")
    summary_file.write(f"Top-1 Correct Matches: {correct_top1}\n")
    summary_file.write(f"Top-5 Correct Matches: {correct_top5}\n")
    summary_file.write(f"Thresholded Correct Matches: {correct_thresholded}\n")
    summary_file.write(f"Direct Matches: {direct_matches}\n")
    summary_file.write(f"Mapped Matches: {mapped_matches}\n")
    summary_file.write(f"Incorrect Matches: {incorrect_matches}\n")

    if images_evaluated > 0:
        top1_rate = (correct_top1 / images_evaluated) * 100
        top5_rate = (correct_top5 / images_evaluated) * 100
        thresholded_rate = (correct_thresholded / images_evaluated) * 100
        precision = thresholded_tp / (thresholded_tp + thresholded_fp) if (thresholded_tp + thresholded_fp) > 0 else 0.0
        recall = thresholded_tp / (thresholded_tp + thresholded_fn) if (thresholded_tp + thresholded_fn) > 0 else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        summary_file.write(f"Top-1 Match Rate: {top1_rate:.2f}%\n")
        summary_file.write(f"Top-5 Match Rate: {top5_rate:.2f}%\n")
        summary_file.write(f"Thresholded Match Rate: {thresholded_rate:.2f}%\n")
        summary_file.write(f"Precision: {precision:.3f}\n")
        summary_file.write(f"Recall: {recall:.3f}\n")
        summary_file.write(f"F1 Score: {f1_score:.3f}\n")

print("\n====================")
print("MODEL EVALUATION")
print("====================")
print(f"Images Evaluated: {images_evaluated}")
print(f"Top-1 Correct Matches: {correct_top1}")
print(f"Top-5 Correct Matches: {correct_top5}")
print(f"Thresholded Correct Matches: {correct_thresholded}")
print(f"Direct Matches: {direct_matches}")
print(f"Mapped Matches: {mapped_matches}")
print(f"Incorrect Matches: {incorrect_matches}")
if images_evaluated > 0:
    print(f"Top-1 Match Rate: {(correct_top1 / images_evaluated) * 100:.2f}%")
    print(f"Top-5 Match Rate: {(correct_top5 / images_evaluated) * 100:.2f}%")
    print(f"Thresholded Match Rate: {(correct_thresholded / images_evaluated) * 100:.2f}%")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1_score:.3f}")
print(f"Saved evaluation CSV to: {OUTPUT_CSV}")
print(f"Saved summary report to: {SUMMARY_PATH}")