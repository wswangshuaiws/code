import torch
import numpy as np
from torch import nn

from config.configs import DATA_PATH, DATASET_PATH, CONFIG, DEVICE, MODEL_MLP_PATH, SCORE_MLP_PATH
from model.Model_GAN import Model_GAN_GEN, Model_GAN_DISC
from model.Model_MLP import Model_MLP

from torch.utils.data import DataLoader
from util.utils import Datasets


def split_train_test_set():
    # 从文件读入数据
    data = np.loadtxt(DATA_PATH, dtype=float)

    # 打乱数据
    np.random.shuffle(data)

    # 划分五倍交叉验证的训练集和测试集
    train_index = []
    test_index = []
    sum_fold = int(data.shape[0] / 5)
    for i in range(5):
        test_index.append(data[i * sum_fold:(i + 1) * sum_fold])
        tmp = []
        for j in range(5):
            if i != j:
                tmp.extend(data[j * sum_fold:(j + 1) * sum_fold])
        train_index.append(np.array(tmp))

    np.savez(DATASET_PATH, train_set=train_index, test_set=test_index)


def run_train_Model_MLP():
    train_index = np.load(DATASET_PATH)['train_set']
    train_index = torch.FloatTensor(train_index)

    for fold in range(5):  # 开始五倍交叉验证
        train_dataset = DataLoader(Datasets(train_index[fold]), CONFIG["batch_size"], shuffle=True)

        net = Model_MLP(dropout_rate=CONFIG["dropout_rate"]).to(DEVICE)
        optimizer = torch.optim.Adam(net.parameters(), lr=CONFIG["lr"])
        criterion = nn.MSELoss()

        net.train()  # 开始训练
        for epoch in range(CONFIG["epoch"]):
            train_loss_records = []
            for step, (x, label) in enumerate(train_dataset):
                pre = net(x)
                train_loss = torch.sqrt(criterion(pre, label))
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                train_loss_records.append(train_loss.item())
            train_loss = round(sum(train_loss_records) / len(train_loss_records), 4)
            print(f"[train]   Fold: {fold + 1} / {5}, Epoch: {epoch + 1} / epoch, Loss: {train_loss}")
        torch.save(net.state_dict(), MODEL_MLP_PATH % fold)


def run_train_Model_GAN():
    pass


def run_test_Model_MLP():
    test_index = np.load(DATASET_PATH)['train_set']
    test_index = torch.FloatTensor(test_index)

    matrices = []
    for fold in range(5):  # 五倍交叉验证
        test_dataset = DataLoader(Datasets(test_index[fold]), CONFIG["batch_size"], shuffle=False)

        # 从文件加载模型
        net = Model_MLP(dropout_rate=CONFIG["dropout_rate"]).to(DEVICE)
        net.load_state_dict(torch.load(MODEL_MLP_PATH % fold, map_location=torch.device("cpu")))

        net.eval()  # 开始测试
        tmp = []
        for step, (x, _) in enumerate(test_dataset):
            with torch.no_grad():
                scores = net(x).numpy()
            tmp.extend(scores)
        matrices.append(tmp)

    # 存储得分矩阵
    np.save(SCORE_MLP_PATH, np.array(matrices))


def run_test_Model_GAN():
    pass


if __name__ == '__main__':
    # split_train_test_set()
    # run_train_Model_MLP()
    # run_test_Model_MLP()
    run_train_Model_GAN()
    run_test_Model_GAN()
