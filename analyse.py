import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from config.configs import DATASET_PATH, SCORE_MLP_PATH

train_index = np.load(DATASET_PATH)['train_set']
target = train_index[:, :, 3:6]
pre = np.load(SCORE_MLP_PATH)

target = target.reshape(320, 3)
pre = pre.reshape(320, 3)

a_pre = pre[:, 0]
b_pre = pre[:, 1]
c_pre = pre[:, 2]

a_target = target[:, 0]
b_target = target[:, 1]
c_target = target[:, 2]

# 误差均方根
print(mean_squared_error(a_pre, a_target))
print(mean_squared_error(b_pre, b_target))
print(mean_squared_error(c_pre, c_target))

# 误差均值
print(np.mean(abs(a_pre - a_target)))
print(np.mean(abs(b_pre - b_target)))
print(np.mean(abs(c_pre - c_target)))

# 确定性系数
print(r2_score(a_pre, a_target))
print(r2_score(b_pre, b_target))
print(r2_score(c_pre, c_target))
