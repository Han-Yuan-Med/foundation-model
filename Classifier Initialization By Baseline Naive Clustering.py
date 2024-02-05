import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset
import pandas as pd
from Classification_Functions import *
from Bootstrap_Functions import *
from Sampling_Functions import create_sampled_dataset
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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

train_set_cls = Lung_cls(csv_file=train_csv, img_dir=image_path)

feature_list = []
with torch.no_grad():
    for data in tqdm(train_set_cls):
        feature_list.append(data[0][0].detach().cpu().flatten().numpy())

feature_list = np.array(feature_list)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_list)

for idx in range(2, 51):
    print(f"Current cluster number: {idx}")
    # kmeans = KMeans(init="random", n_clusters=round(len(train_csv) * idx), random_state=2024, n_init=10)
    kmeans = KMeans(init="random", n_clusters=idx, random_state=2024, n_init=10)
    kmeans.fit_predict(scaled_features)

    id_array = np.zeros(idx)
    for i in range(idx):
        id_array[i] = np.argsort(kmeans.transform(scaled_features)[:, i])[0]

    if idx == 2:
        train_csv.iloc[np.concatenate([id_array])]. \
            to_csv(f"dataset_initialization\\train_km_{idx}.csv", index=False)
    if idx > 2:
        train_csv_past = pd.read_csv(f"dataset_initialization\\train_km_{idx-1}.csv")
        pd.concat([train_csv_past, train_csv.iloc[np.concatenate([id_array])]]).drop_duplicates().\
            reset_index(drop=True).to_csv(f"dataset_initialization\\train_km_{idx}.csv", index=False)


for idx in range(10, 101, 10):
    train_csv_tmp = pd.read_csv(f"dataset_initialization\\train_km_50.csv").iloc[:idx, :]
    print(f"Current sample number: {len(train_csv_tmp)}")
    train_set_cls_tmp = Lung_cls(csv_file=train_csv_tmp, img_dir=image_path)
    train_loader_cls_tmp = torch.utils.data.DataLoader(train_set_cls_tmp, batch_size=10, shuffle=True)

    cls_model = torchvision.models.vgg11(weights=torchvision.models.VGG11_Weights.DEFAULT)
    cls_model.classifier[6] = nn.Linear(4096, 2)
    cls_model.to(device)

    weights = [len(np.where(train_csv_tmp.iloc[:, 4] == 1)[0]) / len(train_csv_tmp),
               len(np.where(train_csv_tmp.iloc[:, 4] == 0)[0]) / len(train_csv_tmp)]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # optimizer = torch.optim.Adam(cls_model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, cls_model.parameters()),
                                lr=0.001, momentum=0.9)

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
        print(f"prc on validation set is {prc_valid} \n"
              f"Previous optimal prc is {past_prc}")

        if prc_valid > past_prc:
            past_prc = prc_valid
            PATH = "D:\\PTX Active Learning Nov\\model_classification" + \
                   f"\\VGG11-km-{idx}.pt"
            torch.save(cls_model, PATH)
            early_stopping = 0
            print("Update classification model\n")
        else:
            early_stopping += 1
            print(f"Not update classification model\n"
                  f"Add early stop index to {int(early_stopping)} \n")

        if early_stopping == 10:
            # break
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


results_df = []
for idx in range(10, 101, 10):
    cls_model = torch.load(f"D:\\PTX Active Learning Nov\\model_classification\\VGG11-km-{idx}.pt")
    cls_model.eval()
    prob_list = []
    label_list = []
    with torch.no_grad():
        for data in tqdm(val_loader_cls):
            images, labels = data[0].float().to(device), data[1].cpu().numpy()
            outputs = cls_model(images)
            label_list = np.concatenate((label_list, labels), axis=None)
            prob_list = np.concatenate((prob_list, torch.sigmoid(outputs)[:, 1].detach().cpu().numpy()), axis=None)

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
            prob_list = np.concatenate((prob_list, torch.sigmoid(outputs)[:, 1].detach().cpu().numpy()), axis=None)

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
    del cls_model

results_df = pd.DataFrame(results_df)
results_df.columns = ['Sample Number', 'Threshold', 'AUROC', 'AUPRC', 'Accuracy', 'Balanced Accuracy',
                      'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
results_df.to_csv("D:\\PTX Active Learning Nov\\VGG11-km-10.csv", index=False, encoding="cp1252")
