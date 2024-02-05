import numpy as np
from sklearn import metrics
import copy


def bootstrap(label_list, binary_list, times, metric):
    bootstrap_value = []
    sample_num = int(binary_list.shape[0] / 50176)

    for i in range(times):
        np.random.seed(i)
        test_index_bootstrap = np.random.choice(range(sample_num), sample_num)
        label_list_bootstrap = np.array([label_list[j * 50176:(j + 1) * 50176] for j in test_index_bootstrap]).flatten()
        prob_list_bootstrap = np.array([binary_list[j * 50176:(j + 1) * 50176] for j in test_index_bootstrap]).flatten()

        if metric == "auc":
            fpr, tpr, thresholds = metrics.roc_curve(label_list_bootstrap, prob_list_bootstrap)
            bootstrap_value.append(metrics.auc(fpr, tpr))

        if metric == "iou":
            bootstrap_value.append(metrics.jaccard_score(label_list_bootstrap, prob_list_bootstrap))

        if metric == "dice":
            bootstrap_value.append(metrics.f1_score(label_list_bootstrap, prob_list_bootstrap))

        print(f'Current bootstrap sampling iteration is {i+1}')

    value_l, value_u = format(np.percentile(bootstrap_value, 2.5), '.3f'), format(np.percentile(bootstrap_value, 97.5), '.3f')
    print(f"{metric} 95% CI is ({value_l}-{value_u})")
    return bootstrap_value, value_l, value_u


def bootstrap_sample(value_list, times):
    bootstrap_value = []

    for i in range(times):
        np.random.seed(i)
        test_index_bootstrap = np.random.choice(range(len(value_list)), len(value_list))
        bootstrap_value.append(value_list[test_index_bootstrap].astype('float').mean())

    value_l, value_u = format(np.percentile(bootstrap_value, 2.5), '.3f'), format(np.percentile(bootstrap_value, 97.5), '.3f')
    print(f"95% CI is ({value_l}-{value_u})")
    return bootstrap_value, value_l, value_u


def splitter(txt, delim):
    for i in txt:
        if i in delim:
            txt = txt.replace(i, ' ')
    return txt.split()


def bootstrap_cls(prob_list, label_list, threshold, times):
    bootstrap_auc = []
    bootstrap_prc = []
    bootstrap_acc = []
    bootstrap_bac = []
    bootstrap_f1s = []
    bootstrap_sen = []
    bootstrap_spe = []
    bootstrap_ppv = []
    bootstrap_npv = []

    for i in range(times):
        np.random.seed(i)
        test_index_bootstrap = np.random.choice(range(len(prob_list)), len(prob_list))
        prob_tmp = prob_list[test_index_bootstrap]
        label_tmp = label_list[test_index_bootstrap]
        # Calculate accuracy, auroc, sensitivity, specificity
        fpr, tpr, thresholds = metrics.roc_curve(label_tmp, prob_tmp)
        precision, recall, _ = metrics.precision_recall_curve(label_tmp, prob_tmp)
        # i.e. point on ROC curve which maximises the sensitivity and specificity
        prob_tmp[np.where(prob_tmp >= threshold)] = 1
        prob_tmp[np.where(prob_tmp != 1)] = 0
        tn, fp, fn, tp = metrics.confusion_matrix(label_tmp, prob_tmp).ravel()
        bootstrap_auc.append(metrics.auc(fpr, tpr))
        bootstrap_prc.append(metrics.auc(recall, precision))
        bootstrap_acc.append((tp + tn) / (tp + tn + fp + fn))
        bootstrap_bac.append((tp / (tp + fn) + tn / (tn + fp))/2)
        bootstrap_f1s.append((2 * tp) / (2 * tp + fp + fn))
        bootstrap_sen.append(tp / (tp + fn))
        bootstrap_spe.append(tn / (tn + fp))
        bootstrap_ppv.append(tp / (tp + fp))
        bootstrap_npv.append(tn / (tn + fn))

    auc_l, auc_u = np.percentile(bootstrap_auc, 2.5), np.percentile(bootstrap_auc, 97.5)
    prc_l, prc_u = np.percentile(bootstrap_prc, 2.5), np.percentile(bootstrap_prc, 97.5)
    acc_l, acc_u = np.percentile(bootstrap_acc, 2.5), np.percentile(bootstrap_acc, 97.5)
    bac_l, bac_u = np.percentile(bootstrap_bac, 2.5), np.percentile(bootstrap_bac, 97.5)
    f1s_l, f1s_u = np.percentile(bootstrap_f1s, 2.5), np.percentile(bootstrap_f1s, 97.5)
    sen_l, sen_u = np.percentile(bootstrap_sen, 2.5), np.percentile(bootstrap_sen, 97.5)
    spe_l, spe_u = np.percentile(bootstrap_spe, 2.5), np.percentile(bootstrap_spe, 97.5)
    ppv_l, ppv_u = np.percentile(bootstrap_ppv, 2.5), np.percentile(bootstrap_ppv, 97.5)
    npv_l, npv_u = np.percentile(bootstrap_npv, 2.5), np.percentile(bootstrap_npv, 97.5)

    return format((auc_u - auc_l) / (2 * 1.96), '.3f'), format((prc_u - prc_l) / (2 * 1.96), '.3f'), \
           format((acc_u - acc_l) / (2 * 1.96), '.3f'), format((bac_u - bac_l) / (2 * 1.96), '.3f'), \
           format((f1s_u - f1s_l) / (2 * 1.96), '.3f'), format((sen_u - sen_l) / (2 * 1.96), '.3f'), \
           format((spe_u - spe_l) / (2 * 1.96), '.3f'), format((ppv_u - ppv_l) / (2 * 1.96), '.3f'), \
           format((npv_u - npv_l) / (2 * 1.96), '.3f')


def bootstrap_cls_mul(prob_list, label_list, threshold, times):
    bootstrap_auc = []
    bootstrap_prc = []
    bootstrap_acc = []
    bootstrap_bac = []
    bootstrap_f1s = []
    bootstrap_sen = []
    bootstrap_spe = []
    bootstrap_ppv = []
    bootstrap_npv = []

    for i in range(times):
        np.random.seed(i)
        test_index_bootstrap = np.random.choice(range(len(prob_list)), len(prob_list))
        prob_tmp = prob_list[test_index_bootstrap]
        label_tmp = label_list[test_index_bootstrap]
        # Calculate accuracy, auroc, sensitivity, specificity
        fpr, tpr, thresholds = metrics.roc_curve(label_tmp, prob_tmp)
        precision, recall, _ = metrics.precision_recall_curve(label_tmp, prob_tmp)
        # i.e. point on ROC curve which maximises the sensitivity and specificity
        prob_tmp[np.where(prob_tmp >= threshold)] = 1
        prob_tmp[np.where(prob_tmp != 1)] = 0
        tn, fp, fn, tp = metrics.confusion_matrix(label_tmp, prob_tmp).ravel()
        bootstrap_auc.append(metrics.auc(fpr, tpr))
        bootstrap_prc.append(metrics.auc(recall, precision))
        bootstrap_acc.append((tp + tn) / (tp + tn + fp + fn))
        bootstrap_bac.append((tp / (tp + fn) + tn / (tn + fp))/2)
        bootstrap_f1s.append((2 * tp) / (2 * tp + fp + fn))
        bootstrap_sen.append(tp / (tp + fn))
        bootstrap_spe.append(tn / (tn + fp))
        bootstrap_ppv.append(tp / (tp + fp))
        bootstrap_npv.append(tn / (tn + fn))
    return bootstrap_auc, bootstrap_prc, bootstrap_acc, bootstrap_bac, bootstrap_f1s, bootstrap_sen, \
           bootstrap_spe, bootstrap_ppv, bootstrap_npv