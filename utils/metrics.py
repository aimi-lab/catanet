import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import sklearn.metrics
import warnings

def roc_auc_multilabel(y_true, y_predicted, mask=None):
    samples, classes = y_true.shape
    if mask is None:
        mask = np.ones(classes, dtype=bool)
    y_true[y_true == 0.5] = 0
    mask = mask & (np.sum(y_true, axis=0) > 0)
    auc_score = np.ones(classes)*np.nan
    average_auc_score = 0.0
    for i in range(classes):
        if mask[i]:
            auc_score[i] = roc_auc_score(y_true[:, i], y_predicted[:, i], average='macro', sample_weight=None)
            average_auc_score += auc_score[i]
    average_auc_score /= np.sum(mask)

    return auc_score, average_auc_score


def roc_multilabel(y_true, y_predicted):
    samples, classes = y_true.shape
    y_true[y_true == 0.5] = 0
    mask = np.sum(y_true, axis=0) > 0
    fpr = [[] for i in range(classes)]
    tpr = [[] for i in range(classes)]
    thr = [[] for i in range(classes)]
    for i in range(classes):
        if mask[i]:
            fpr[i], tpr[i], thr[i] = roc_curve(y_true[:, i], y_predicted[:, i])

    return fpr, tpr, thr


def pr_multilable(y_true, y_predicted):
    samples, classes = y_true.shape
    y_true[y_true == 0.5] = 0
    mask = np.sum(y_true, axis=0) > 0
    precision = [[] for i in range(classes)]
    recall = [[] for i in range(classes)]
    thr = [[] for i in range(classes)]
    for i in range(classes):
        if mask[i]:
            precision[i], recall[i], thr[i] = precision_recall_curve(y_true[:, i], y_predicted[:, i])

    return precision, recall, thr

def average_precision_multilable(y_true, y_predicted, mask=None):
    samples, classes = y_true.shape
    if mask is None:
        mask = np.ones(classes, dtype=bool)
    y_true[y_true == 0.5] = 0
    mask = mask & (np.sum(y_true, axis=0) > 0)
    ap_score = np.ones(classes) * np.nan
    map_score = 0.0
    for i in range(classes):
        if mask[i]:
            ap_score[i] = average_precision_score(y_true[:, i], y_predicted[:, i], average='macro', sample_weight=None)
            map_score += ap_score[i]
    map_score /= np.sum(mask)

    return ap_score, map_score


def counts_from_confusion(confusion):
    """
    Obtain TP, FN FP, and TN for each class in the confusion matrix
    """

    TP = np.zeros(confusion.shape[0])
    FN = np.zeros(confusion.shape[0])
    FP = np.zeros(confusion.shape[0])
    TN = np.zeros(confusion.shape[0])

    # Iterate through classes and store the counts
    for i in range(confusion.shape[0]):
        tp = confusion[i, i]

        fn_mask = np.zeros(confusion.shape)
        fn_mask[i, :] = 1
        fn_mask[i, i] = 0
        fn = np.sum(np.multiply(confusion, fn_mask))

        fp_mask = np.zeros(confusion.shape)
        fp_mask[:, i] = 1
        fp_mask[i, i] = 0
        fp = np.sum(np.multiply(confusion, fp_mask))

        tn_mask = 1 - (fn_mask + fp_mask)
        tn_mask[i, i] = 0
        tn = np.sum(np.multiply(confusion, tn_mask))

        TP[i] = tp
        FN[i] = fn
        FP[i] = fp
        TN[i] = tn

    return TP, FN, FP, TN

def f1_score(y_true, y_pred, n_classes):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1 = sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average=None)
    f1_filtered = np.ones(n_classes)*np.nan
    running_id = 0
    for val in range(n_classes):
        counts1 = np.sum(y_true == val)
        counts2 = np.sum(y_pred == val)
        if counts1 > 0:
            f1_filtered[val] = f1[running_id]
            running_id += 1
        elif counts2 > 0:
            f1_filtered[val] = np.nan
            running_id += 1
    return f1_filtered

def conf2metrics(conf_mat):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
    precision[np.isnan(precision)] = 0.0

    recall = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
    recall[np.isnan(recall)] = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1 = 2 * (precision * recall) / (precision + recall)
    f1[np.isnan(f1)] = 0.0

    accuracy = np.trace(conf_mat) / np.sum(conf_mat)
    return precision, recall, f1, accuracy


def time_series_metrics(prediction, label, n_classes, alpha=0.5):
    # precision and recall for time series -
    # N. Tatbul 2018 - Precision and Recall for Time Series

    def existenceReward(range1, setOfRanges2):
        # rewards existence of range1 in the set of ranges 2
        if np.sum(range1 & setOfRanges2) > 0:
            return 1
        else:
            return 0

    def overlapReward(range1, setOfRanges2):
        def cardinality(range1, setOfRanges2):
            # penalizes split of ranges
            overlap = range1 & setOfRanges2
            n_edges = np.max((1, np.sum(np.diff(1*overlap) == 1)))
            return 1/n_edges

        def size(range1, sefOfRanges2):
            # overlap size of ranges - positional bias is not implemented
            overlap = range1 & sefOfRanges2
            return np.sum(overlap) / np.sum(range1)

        reward = cardinality(range1, setOfRanges2) * size(range1, setOfRanges2)
        return reward

    def range_based_recall(prediction, label, channel, alpha):
        R_set = (label==channel)  # real class ranges
        P_set = (prediction==channel)  # predicted class ranges

        recall_list = []
        rising_edges = np.where(np.diff(1*R_set, prepend=0, append=0) == 1)[0]
        falling_edges = np.where(np.diff(1*R_set, prepend=0, append=0) == -1)[0]

        for r,f in zip(rising_edges, falling_edges):
            mask = np.zeros_like(R_set, dtype=bool)
            mask[r:f+1] = True
            R = R_set & mask
            recall = alpha*existenceReward(R, P_set) + (1-alpha) * overlapReward(R, P_set)
            recall_list.append(recall)
        recall = np.sum(recall_list) / len(recall_list)
        return recall


    def range_based_precision(prediction, label, channel, alpha):
        R_set = (label==channel)  # real class ranges
        P_set = (prediction==channel)  # predicted class ranges
        precision_list = []
        rising_edges = np.where(np.diff(1*P_set, prepend=0, append=0) == 1)[0]
        falling_edges = np.where(np.diff(1*P_set, prepend=0, append=0) == -1)[0]
        for r, f in zip(rising_edges, falling_edges):
            mask = np.zeros_like(P_set, dtype=bool)
            mask[r:f+1] = True
            P = P_set & mask
            precision = alpha * existenceReward(P, R_set) + (1 - alpha) * overlapReward(P, R_set)
            precision_list.append(precision)
        precision = np.sum(precision_list) / len(precision_list)
        return precision

    recall = np.asarray([range_based_recall(prediction, label, cl, 0.5) for cl in range(n_classes)])
    precision = np.asarray([range_based_precision(prediction, label, cl, 0.5) for cl in range(n_classes)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1_score = np.asarray([2*r*p / (r+p) for (r,p) in zip(recall, precision)])
    f1_score[np.isnan(f1_score)] = 0.0

    return precision, recall, f1_score


def get_cardinality(prediction, label, n_classes):
    def cardinality(range1, setOfRanges2):
        # penalizes split of ranges
        overlap = range1 & setOfRanges2
        n_edges = np.max((1, np.sum(np.diff(1 * overlap) == 1)))
        if n_edges > 0:
            return 1 / n_edges
        else:
            return np.nan

    def channel_wise_card(prediction, label, channel):
        R_set = (label == channel)  # real class ranges
        P_set = (prediction == channel)  # predicted class ranges

        card_list = []
        rising_edges = np.where(np.diff(1 * R_set, prepend=0, append=0) == 1)[0]
        falling_edges = np.where(np.diff(1 * R_set, prepend=0, append=0) == -1)[0]

        for r, f in zip(rising_edges, falling_edges):
            mask = np.zeros_like(R_set, dtype=bool)
            mask[r:f + 1] = True
            R = R_set & mask
            card = cardinality(R, P_set)
            card_list.append(card)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            card = np.nanmean(card_list)
        return card


    reward = np.asarray([channel_wise_card(prediction, label, cl) for cl in range(n_classes)])
    return reward

# gt = np.zeros(100)
# pred1 = np.zeros(100)
# pred2 = np.zeros(100)
#
# gt[10:30] = 1
# pred1[[10, 12, 15, 18]] = 1
# pred1[20:35] = 1
# pred1[37:39] = 1
# pred2[16:37] = 1
#
#
# card1 = get_cardinality(pred1, gt, 1)
# card2 = get_cardinality(pred2, gt, 1)
#
# def f1_pointbased(prediction, label):
#     tp = np.sum((prediction == 1) & (label == 1))
#     fp = np.sum((prediction == 1) & (label == 0))
#     fn = np.sum((prediction == 0) & (label == 1))
#     precision = tp/(tp+fp)
#     recall = tp/(tp+fn)
#     f1 = 2*(precision*recall)/(precision+recall)
#     return f1
#
# f1_point1 = f1_pointbased(pred1, gt)
# f1_point2 = f1_pointbased(pred2, gt)
#
# import matplotlib.pyplot as plt
# tt = np.arange(100)
# plt.subplot(3,1,1)
# plt.step(tt, pred1)
# plt.legend(['f1 {0:.2f}, card {1:.2f}'.format(f1_point1, card1.squeeze())])
# plt.subplot(3,1,2)
# plt.step(tt, pred2, color='orange')
# plt.legend(['f1 {0:.2f}, card {1:.2f}'.format(f1_point2, card2.squeeze())])
# plt.subplot(3,1,3)
# plt.step(tt, gt, color='green')
# plt.legend(['ground-truth'])
# plt.show()