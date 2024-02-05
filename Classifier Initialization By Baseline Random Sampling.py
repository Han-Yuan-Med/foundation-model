import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset
import pandas as pd
from Classification_Functions import *
from Bootstrap_Functions import *
from Sampling_Functions import create_sampled_instance
from tqdm import tqdm

setup_seed(2024)

image_path = "D:\\ChestX-Det-Dataset"
train_csv = pd.read_csv("train_1.csv").iloc[:, 1:]
val_csv = pd.read_csv("val_1.csv").iloc[:, 1:]
test_csv = pd.read_csv("test_1.csv").iloc[:, 1:]
sample_num = round(len(pd.read_csv("train_1.csv")) * 0.1)

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

val_set_cls = Lung_cls(csv_file=val_csv, img_dir=image_path)
val_loader_cls = torch.utils.data.DataLoader(val_set_cls, batch_size=1, shuffle=False)

test_set_cls = Lung_cls(csv_file=test_csv, img_dir=image_path)
test_loader_cls = torch.utils.data.DataLoader(test_set_cls, batch_size=1, shuffle=False)

sample_times = 10
# create_sampled_dataset(train_csv=train_csv, random_seed=2024, dataset_path="dataset_initialization",
#                        sample_times=10)
# create_sampled_instance(train_csv=train_csv, random_seed=2024, dataset_path="dataset_initialization", sample_times=10)

for sample_id in range(sample_times):
    for idx in range(10, 101, 10):
        train_csv_tmp = pd.read_csv(f"D:\\PTX Active Learning Nov\\dataset_initialization\\"
                                    f"train_id_{sample_id}_{idx}.csv")
        train_set_cls_tmp = Lung_cls(csv_file=train_csv_tmp, img_dir=image_path)
        train_loader_cls_tmp = torch.utils.data.DataLoader(train_set_cls_tmp, batch_size=16, shuffle=True)

        cls_model = torchvision.models.vgg11(weights=torchvision.models.VGG11_Weights.DEFAULT)
        # Model description
        cls_model.classifier[6] = nn.Linear(4096, 2)
        cls_model.to(device)

        weights = [len(np.where(train_csv_tmp.iloc[:, 4] == 1)[0]) / len(train_csv_tmp),
                   len(np.where(train_csv_tmp.iloc[:, 4] == 0)[0]) / len(train_csv_tmp)]
        class_weights = torch.FloatTensor(weights).cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, cls_model.parameters()),
                                    lr=0.001, momentum=0.9)
        # optimizer = torch.optim.Adam(cls_model.parameters(), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        epoch_num = 100
        past_prc = 0
        early_stopping = 0

        for epoch in range(epoch_num):
            train_cls(train_loader=train_loader_cls_tmp, device=device, cls_model=cls_model,
                      criterion=criterion, optimizer=optimizer)
            print(f"Finish training at epoch {epoch + 1}; \n"
                  f"Start validation:")

            cls_model.eval()
            prc_valid = valid_cls_prc(val_loader=val_loader_cls, device=device, cls_model=cls_model)
            print(f"Prc on validation set is {prc_valid} \n"
                  f"Previous optimal prc is {past_prc}")

            if prc_valid > past_prc:
                past_prc = prc_valid
                PATH = "D:\\PTX Active Learning Nov\\model_classification" + \
                       f"\\VGG11-Data-id-{sample_id}-{idx}.pt"
                torch.save(cls_model, PATH)
                early_stopping = 0
                print("Update classification model\n")
            else:
                early_stopping += 1
                print(f"Not update classification model\n"
                      f"Add early stop index to {int(early_stopping)} \n")

            if early_stopping == 5:
                print("No improvement in the last 5 epochs; \n Decrease learning rate")
                before_lr = optimizer.param_groups[0]["lr"]
                scheduler.step()
                after_lr = optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch + 1} SGD lr {before_lr} -> {after_lr} \n")
                early_stopping = 0

        del cls_model
        del criterion
        del optimizer
        del scheduler
        del train_csv_tmp
        del train_set_cls_tmp
        del train_loader_cls_tmp

for sample_id in range(sample_times):
    for idx in range(10, 101, 10):
        results_df = []
        cls_model = torch.load("D:\\PTX Active Learning Nov\\model_classification" + \
                               f"\\VGG11-Data-id-{sample_id}-{idx}.pt")

        cls_model.eval()
        prob_list = []
        label_list = []
        with torch.no_grad():
            for data in tqdm(val_loader_cls):
                images, labels = data[0].float().to(device), data[1].cpu().numpy()
                outputs = cls_model(images)
                label_list = np.concatenate((label_list, labels), axis=None)
                prob_list = np.concatenate((prob_list, torch.sigmoid(outputs)[:, 1].detach().cpu().numpy()),
                                           axis=None)

        # Calculate accuracy, auroc, sensitivity, specificity
        fpr, tpr, thresholds = metrics.roc_curve(label_list, prob_list)
        thres_val = round(thresholds[np.argmax(tpr - fpr)], 3)
        prob_list[np.where(prob_list >= thres_val)] = 1
        prob_list[np.where(prob_list != 1)] = 0
        tn, fp, fn, tp = metrics.confusion_matrix(label_list, prob_list).ravel()
        print(f'AUROC on the validation set is {round(metrics.auc(fpr, tpr), 3)}')
        print(f'Threshold determined by Youden index is {thres_val}')
        print(f'Accuracy on the validation set is {round((tp + tn) / (tp + tn + fp + fn), 3)}')
        print(f'Sensitivity on the validation set is {round(tp / (tp + fn), 3)}')
        print(f'Specificity on the validation set is {round(tn / (tn + fp), 3)}')

        prob_list = []
        label_list = []
        with torch.no_grad():
            for data in tqdm(test_loader_cls):
                images, labels = data[0].float().to(device), data[1].cpu().numpy()
                outputs = cls_model(images)
                label_list = np.concatenate((label_list, labels), axis=None)
                prob_list = np.concatenate((prob_list, torch.sigmoid(outputs)[:, 1].detach().cpu().numpy()),
                                           axis=None)

        # Calculate accuracy, auroc, sensitivity, specificity
        fpr, tpr, thresholds = metrics.roc_curve(label_list, prob_list)
        precision, recall, thresholds = metrics.precision_recall_curve(label_list, prob_list)
        auc_std, prc_std, acc_std, bac_std, f1s_std, sen_std, spe_std, ppv_std, npv_std = \
            bootstrap_cls(prob_list=prob_list, label_list=label_list, threshold=thres_val, times=100)

        prob_list[np.where(prob_list >= thres_val)] = 1
        prob_list[np.where(prob_list != 1)] = 0
        tn, fp, fn, tp = metrics.confusion_matrix(label_list, prob_list).ravel()

        print(f'AUROC on the test set is {round(metrics.auc(fpr, tpr), 3)}')
        print(f'AUPRC on the test set is {round(metrics.auc(recall, precision), 3)}')
        print(f'Threshold determined by Youden index is {thres_val}')
        print(f'Accuracy on the test set is {round((tp + tn) / (tp + tn + fp + fn), 3)}')
        print(f'Balanced Accuracy on the test set is {round((tp / (tp + fn) + tn / (tn + fp)) / 2, 3)}')
        print(f'F1 Score on the test set is {round((2 * tp) / (2 * tp + fp + fn), 3)}')
        print(f'Sensitivity on the test set is {round(tp / (tp + fn), 3)}')
        print(f'Specificity on the test set is {round(tn / (tn + fp), 3)}')
        print(f'PPV on the test set is {round(tp / (tp + fp), 3)}')
        print(f'NPV on the test set is {round(tn / (tn + fn), 3)}')

        results_df.append([f"{idx}", f"{thres_val}",
                           f"{format(metrics.auc(fpr, tpr), '.3f')} ({auc_std})",
                           f"{format(metrics.auc(recall, precision), '.3f')} ({prc_std})",
                           f"{format((tp + tn) / (tp + tn + fp + fn), '.3f')} ({acc_std})",
                           f"{format((tp / (tp + fn) + tn / (tn + fp)) / 2, '.3f')} ({bac_std})",
                           f"{format((2 * tp) / (2 * tp + fp + fn), '.3f')} ({f1s_std})",
                           f"{format(tp / (tp + fn), '.3f')} ({sen_std})",
                           f"{format(tn / (tn + fp), '.3f')} ({spe_std})",
                           f"{format(tp / (tp + fp), '.3f')} ({ppv_std})",
                           f"{format(tn / (tn + fn), '.3f')} ({npv_std})",
                           ])

        results_df = pd.DataFrame(results_df)
        results_df.columns = ['Model', 'Threshold', 'AUROC', 'AUPRC', 'Accuracy', 'Balanced Accuracy',
                              'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
        results_df.to_csv("D:\\PTX Active Learning Nov\\results_classification" + \
                          f"\\VGG11-Cls-{sample_id}-{idx}.csv", index=False, encoding="cp1252")


auc_stdv = []
prc_stdv = []
acc_stdv = []
bac_stdv = []
f1s_stdv = []
sen_stdv = []
spe_stdv = []
ppv_stdv = []
npv_stdv = []
for idx in range(10, 101, 10):
    bootstrap_auc = []
    bootstrap_prc = []
    bootstrap_acc = []
    bootstrap_bac = []
    bootstrap_f1s = []
    bootstrap_sen = []
    bootstrap_spe = []
    bootstrap_ppv = []
    bootstrap_npv = []
    for sample_id in range(sample_times):
        cls_model = torch.load("D:\\PTX Active Learning Nov\\model_classification" + \
                               f"\\VGG11-Data-id-{sample_id}-{idx}.pt")

        cls_model.eval()
        prob_list = []
        label_list = []
        with torch.no_grad():
            for data in tqdm(val_loader_cls):
                images, labels = data[0].float().to(device), data[1].cpu().numpy()
                outputs = cls_model(images)
                label_list = np.concatenate((label_list, labels), axis=None)
                prob_list = np.concatenate((prob_list, torch.sigmoid(outputs)[:, 1].detach().cpu().numpy()),
                                           axis=None)

        # Calculate accuracy, auroc, sensitivity, specificity
        fpr, tpr, thresholds = metrics.roc_curve(label_list, prob_list)
        thres_val = round(thresholds[np.argmax(tpr - fpr)], 3)
        prob_list[np.where(prob_list >= thres_val)] = 1
        prob_list[np.where(prob_list != 1)] = 0
        tn, fp, fn, tp = metrics.confusion_matrix(label_list, prob_list).ravel()
        print(f'AUROC on the validation set is {round(metrics.auc(fpr, tpr), 3)}')
        print(f'Threshold determined by Youden index is {thres_val}')
        print(f'Accuracy on the validation set is {round((tp + tn) / (tp + tn + fp + fn), 3)}')
        print(f'Sensitivity on the validation set is {round(tp / (tp + fn), 3)}')
        print(f'Specificity on the validation set is {round(tn / (tn + fp), 3)}')

        prob_list = []
        label_list = []
        with torch.no_grad():
            for data in tqdm(test_loader_cls):
                images, labels = data[0].float().to(device), data[1].cpu().numpy()
                outputs = cls_model(images)
                label_list = np.concatenate((label_list, labels), axis=None)
                prob_list = np.concatenate((prob_list, torch.sigmoid(outputs)[:, 1].detach().cpu().numpy()),
                                           axis=None)
        auc_std, prc_std, acc_std, bac_std, f1s_std, sen_std, spe_std, ppv_std, npv_std = \
            bootstrap_cls_mul(prob_list=prob_list, label_list=label_list, threshold=thres_val, times=100)

        bootstrap_auc = bootstrap_auc + auc_std
        bootstrap_prc = bootstrap_prc + prc_std
        bootstrap_acc = bootstrap_acc + acc_std
        bootstrap_bac = bootstrap_bac + bac_std
        bootstrap_f1s = bootstrap_f1s + f1s_std
        bootstrap_sen = bootstrap_sen + sen_std
        bootstrap_spe = bootstrap_spe + spe_std
        bootstrap_ppv = bootstrap_ppv + ppv_std
        bootstrap_npv = bootstrap_npv + npv_std
    auc_l, auc_u = np.percentile(bootstrap_auc, 2.5), np.percentile(bootstrap_auc, 97.5)
    prc_l, prc_u = np.percentile(bootstrap_prc, 2.5), np.percentile(bootstrap_prc, 97.5)
    acc_l, acc_u = np.percentile(bootstrap_acc, 2.5), np.percentile(bootstrap_acc, 97.5)
    bac_l, bac_u = np.percentile(bootstrap_bac, 2.5), np.percentile(bootstrap_bac, 97.5)
    f1s_l, f1s_u = np.percentile(bootstrap_f1s, 2.5), np.percentile(bootstrap_f1s, 97.5)
    sen_l, sen_u = np.percentile(bootstrap_sen, 2.5), np.percentile(bootstrap_sen, 97.5)
    spe_l, spe_u = np.percentile(bootstrap_spe, 2.5), np.percentile(bootstrap_spe, 97.5)
    ppv_l, ppv_u = np.percentile(bootstrap_ppv, 2.5), np.percentile(bootstrap_ppv, 97.5)
    npv_l, npv_u = np.percentile(bootstrap_npv, 2.5), np.percentile(bootstrap_npv, 97.5)

    auc_stdv.append(format((auc_u - auc_l) / (2 * 1.96), '.3f'))
    prc_stdv.append(format((prc_u - prc_l) / (2 * 1.96), '.3f'))
    acc_stdv.append(format((acc_u - acc_l) / (2 * 1.96), '.3f'))
    bac_stdv.append(format((bac_u - bac_l) / (2 * 1.96), '.3f'))
    f1s_stdv.append(format((f1s_u - f1s_l) / (2 * 1.96), '.3f'))
    sen_stdv.append(format((sen_u - sen_l) / (2 * 1.96), '.3f'))
    spe_stdv.append(format((spe_u - spe_l) / (2 * 1.96), '.3f'))
    ppv_stdv.append(format((ppv_u - ppv_l) / (2 * 1.96), '.3f'))
    npv_stdv.append(format((npv_u - npv_l) / (2 * 1.96), '.3f'))

results_avg = []
for idx in range(10, 101, 10):
    auc = 0
    prc = 0
    acc = 0
    bac = 0
    f1s = 0
    sen = 0
    spe = 0
    ppv = 0
    npv = 0
    for sample_id in range(sample_times):
        result_tmp = pd.read_csv(f"D:\\PTX Active Learning Nov\\results_classification\\"
                                 f"\\VGG11-Cls-{sample_id}-{idx}.csv")

        auc_value = splitter(result_tmp.iloc[0, 2], ['(', ')', '-', ' '])
        prc_value = splitter(result_tmp.iloc[0, 3], ['(', ')', '-', ' '])
        acc_value = splitter(result_tmp.iloc[0, 4], ['(', ')', '-', ' '])
        bac_value = splitter(result_tmp.iloc[0, 5], ['(', ')', '-', ' '])
        f1s_value = splitter(result_tmp.iloc[0, 6], ['(', ')', '-', ' '])
        sen_value = splitter(result_tmp.iloc[0, 7], ['(', ')', '-', ' '])
        spe_value = splitter(result_tmp.iloc[0, 8], ['(', ')', '-', ' '])
        ppv_value = splitter(result_tmp.iloc[0, 9], ['(', ')', '-', ' '])
        npv_value = splitter(result_tmp.iloc[0, 10], ['(', ')', '-', ' '])

        auc += float(auc_value[0])
        # auc_std += float(auc_value[1]) ** 2

        prc += float(prc_value[0])
        # prc_std += float(prc_value[1]) ** 2

        acc += float(acc_value[0])
        # acc_std += float(acc_value[1]) ** 2

        bac += float(bac_value[0])
        # bac_std += float(bac_value[1]) ** 2

        f1s += float(f1s_value[0])
        # f1s_std += float(f1s_value[1]) ** 2

        sen += float(sen_value[0])
        # sen_std += float(sen_value[1]) ** 2

        spe += float(spe_value[0])
        # spe_std += float(spe_value[1]) ** 2

        ppv += float(ppv_value[0])
        # ppv_std += float(ppv_value[1]) ** 2

        npv += float(npv_value[0])
        # npv_std += float(npv_value[1]) ** 2

    results_avg.append([f"{idx}", "test set",
                        f"{format(auc / 10, '.3f')} ({auc_stdv[int(idx/10-1)]})",
                        f"{format(prc / 10, '.3f')} ({prc_stdv[int(idx/10-1)]})",
                        f"{format(acc / 10, '.3f')} ({acc_stdv[int(idx/10-1)]})",
                        f"{format(bac / 10, '.3f')} ({bac_stdv[int(idx/10-1)]})",
                        f"{format(f1s / 10, '.3f')} ({f1s_stdv[int(idx/10-1)]})",
                        f"{format(sen / 10, '.3f')} ({sen_stdv[int(idx/10-1)]})",
                        f"{format(spe / 10, '.3f')} ({spe_stdv[int(idx/10-1)]})",
                        f"{format(ppv / 10, '.3f')} ({ppv_stdv[int(idx/10-1)]})",
                        f"{format(npv / 10, '.3f')} ({npv_stdv[int(idx/10-1)]})"
                        ])
results_avg = pd.DataFrame(results_avg)
results_avg.columns = ['Ratio', 'Dataset', 'AUROC', 'AUPRC', 'Accuracy', 'Blanaced Accuracy', 'F1 Score',
                       'Sensitivity', 'Specificity', 'PPV', 'NPV']
results_avg.to_csv(f"D:\\PTX Active Learning Nov\\VGG11-rs-10.csv", index=False)
