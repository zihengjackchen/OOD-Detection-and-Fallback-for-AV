import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

clf = joblib.load(naive_bayes_model_left.pkl)
# 假设已经有一个训练好的高斯朴素贝叶斯模型clf
 means = clf.theta_
 variances = clf.sigma_

# 为了演示，这里定义了一些假设的均值和方差
#means = [0, 1]
#variances = [1, 1]

# 定义x轴范围
x = np.linspace(-3, 3, 1000)

# 为每个类别画出高斯分布图
for mean, variance in zip(means, variances):
    plt.plot(x, norm.pdf(x, mean, np.sqrt(variance)), label=f"Mean: {mean}, Variance: {variance}")

plt.title('Gaussian Distributions of the Trained Gaussian Naive Bayes Model')
plt.xlabel('Feature Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
