from dataset.load_data import load_json
import argparse


def evaluate(predict_result, ground_truth, print_names=False):
    if isinstance(predict_result, str):
        predict_result = load_json(predict_result)
    if isinstance(ground_truth, str):
        ground_truth = load_json(ground_truth)

    name_nums = 0
    result_list = []
    for name in predict_result:
        # Get clustering labels in predict_result
        predicted_pubs = dict()
        for idx, pids in enumerate(predict_result[name]):
            for pid in pids:
                predicted_pubs[pid] = idx
        # Get paper labels in ground_truth
        pubs = []
        ilabel = 0
        true_labels = []
        for aid in ground_truth[name]:
            pubs.extend(ground_truth[name][aid])
            true_labels.extend([ilabel] * len(ground_truth[name][aid]))
            ilabel += 1

        predict_labels = []
        for pid in pubs:
            predict_labels.append(predicted_pubs[pid])

        pairwise_precision, pairwise_recall, pairwise_f1 = pairwise_evaluate(
            true_labels, predict_labels
        )
        result_list.append((pairwise_precision, pairwise_recall, pairwise_f1))
        name_nums += 1

        if print_names:
            print(
                "name:",
                name,
                "pre:",
                "%.4f" % pairwise_precision,
                "rec:",
                "%.4f" % pairwise_recall,
                "f1:",
                "%.4f" % pairwise_f1,
            )

    avg_pairwise_pre = sum([result[0] for result in result_list]) / name_nums
    avg_pairwise_rec = sum([result[1] for result in result_list]) / name_nums
    avg_pairwise_f1 = sum([result[2] for result in result_list]) / name_nums
    # print(f'avg_pairwise_pre:{avg_pairwise_pre:.3f}, avg_pairwise_rec:{avg_pairwise_rec:.3f}, Average Pairwise F1: {avg_pairwise_f1:.3f}')

    return avg_pairwise_pre, avg_pairwise_rec, avg_pairwise_f1


def pairwise_evaluate(correct_labels, pred_labels):
    TP = 0.0  # Pairs Correctly Predicted To SameAuthor
    TP_FP = 0.0  # Total Pairs Predicted To SameAuthor
    TP_FN = 0.0  # Total Pairs To SameAuthor

    for i in range(len(correct_labels)):
        for j in range(i + 1, len(correct_labels)):
            if correct_labels[i] == correct_labels[j]:
                TP_FN += 1
            if pred_labels[i] == pred_labels[j]:
                TP_FP += 1
            if (correct_labels[i] == correct_labels[j]) and (
                pred_labels[i] == pred_labels[j]
            ):
                TP += 1

    if TP == 0:
        pairwise_precision = 0
        pairwise_recall = 0
        pairwise_f1 = 0
    else:
        pairwise_precision = TP / TP_FP
        pairwise_recall = TP / TP_FN
        pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (
            pairwise_precision + pairwise_recall
        )

    return pairwise_precision, pairwise_recall, pairwise_f1


def get_true_labels(ground_truth, name):
    """
    Extract the true labels for a specific name from ground_truth data.

    Args:
        ground_truth (dict or str): Ground truth data (dict or path to JSON file).
        name (str): Name key for which to retrieve true labels.

    Returns:
        pubs (list): List of paper IDs.
        true_labels (list): List of corresponding true labels.
    """
    if isinstance(ground_truth, str):
        ground_truth = load_json(ground_truth)

    pubs = []
    true_labels = []
    ilabel = 0

    for aid in ground_truth[name]:
        pubs.extend(ground_truth[name][aid])
        true_labels.extend([ilabel] * len(ground_truth[name][aid]))
        ilabel += 1

    return true_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate predictions against ground truth."
    )
    parser.add_argument("--predict", required=True, help="Path to prediction results")
    parser.add_argument("--ground_truth", required=True, help="Path to ground truth")

    args = parser.parse_args()

    evaluate(args.predict, args.ground_truth)
