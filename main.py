import torch
import numpy as np
from torch import nn
from torch.autograd import Variable

from config.configs import DATA_PATH, DATASET_PATH, CONFIG, DEVICE, MODEL_MLP_PATH, SCORE_MLP_PATH, MODEL_GAN_GEN_PATH, \
    MODEL_GAN_DISC_PATH, SCORE_GAN_PATH
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
    train_index = np.load(DATASET_PATH)['train_set']
    train_index = torch.FloatTensor(train_index)

    for fold in range(5):  # 开始五倍交叉验证
        # 定义生成器和鉴别器
        generator = Model_GAN_GEN(dropout_rate=CONFIG["dropout_rate"]).to(DEVICE)
        discriminator = Model_GAN_DISC(dropout_rate=CONFIG["dropout_rate"]).to(DEVICE)

        # 定义优化器
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=CONFIG["lr"])
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=CONFIG["lr"])

        # 定义损失函数
        criterion_1 = nn.CrossEntropyLoss()
        criterion_2 = nn.MSELoss()

        for k in range(CONFIG["GAN_train_num"]):
            # 训练鉴别器
            train_dataset = DataLoader(Datasets(train_index[fold]), CONFIG["batch_size"], shuffle=True)
            generator.eval()
            discriminator.train()
            for epoch in range(CONFIG["epoch"]):
                train_loss_records = []
                for step, (x, label) in enumerate(train_dataset):
                    fake_data = generator(x)
                    output_fake = discriminator(fake_data.detach())
                    output_real = discriminator(label)
                    real_labels = Variable(torch.ones(x.shape[0])).long()
                    fake_labels = Variable(torch.zeros(x.shape[0])).long()

                    loss_real = criterion_1(output_real, real_labels)
                    loss_fake = criterion_1(output_fake, fake_labels)
                    loss_d = loss_real + loss_fake

                    optimizer_d.zero_grad()
                    loss_d.backward()
                    optimizer_d.step()
                    train_loss_records.append(loss_d.item())
                train_loss = round(sum(train_loss_records) / len(train_loss_records), 4)
                print(f"[train-disc]   Fold: {fold + 1} / {5}, Epoch: {epoch + 1} / epoch, Loss: {train_loss}")

            # 训练生成器
            train_dataset = DataLoader(Datasets(train_index[fold]), CONFIG["batch_size"], shuffle=True)
            generator.train()  # 开始训练
            discriminator.eval()
            for epoch in range(CONFIG["epoch"]):
                train_loss_records = []
                for step, (x, label) in enumerate(train_dataset):
                    fake_data = generator(x)
                    output_fake = discriminator(fake_data)
                    real_labels = Variable(torch.ones(x.shape[0])).long()
                    label = Variable(label).long()

                    loss_g = criterion_1(output_fake, real_labels) + criterion_2(fake_data, label)

                    optimizer_g.zero_grad()
                    loss_g.backward()
                    optimizer_g.step()
                    train_loss_records.append(loss_g.item())
                train_loss = round(sum(train_loss_records) / len(train_loss_records), 4)
                print(f"[train-gen]   Fold: {fold + 1} / {5}, Epoch: {epoch + 1} / epoch, Loss: {train_loss}")

        torch.save(generator.state_dict(), MODEL_GAN_GEN_PATH % fold)
        torch.save(discriminator.state_dict(), MODEL_GAN_DISC_PATH % fold)


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
    test_index = np.load(DATASET_PATH)['train_set']
    test_index = torch.FloatTensor(test_index)

    matrices = []
    for fold in range(5):  # 五倍交叉验证
        test_dataset = DataLoader(Datasets(test_index[fold]), CONFIG["batch_size"], shuffle=False)

        # 从文件加载模型
        net = Model_GAN_GEN(dropout_rate=CONFIG["dropout_rate"]).to(DEVICE)
        net.load_state_dict(torch.load(MODEL_GAN_GEN_PATH % fold, map_location=torch.device("cpu")))

        net.eval()  # 开始测试
        tmp = []
        for step, (x, _) in enumerate(test_dataset):
            with torch.no_grad():
                scores = net(x).numpy()
            tmp.extend(scores)
        matrices.append(tmp)

    # 存储得分矩阵
    np.save(SCORE_GAN_PATH, np.array(matrices))


if __name__ == '__main__':
    split_train_test_set()
    run_train_Model_MLP()
    run_test_Model_MLP()
    run_train_Model_GAN()
    run_test_Model_GAN()
