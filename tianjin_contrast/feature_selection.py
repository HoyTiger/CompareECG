from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score


file = '/data/0shared/hanyuhu/features_hr_label.csv'
df = pd.read_csv(file)

X = df[df.columns[1:-2]]
y = df['hr']

# --- Extra-Trees
model = ExtraTreesClassifier()
model.fit(X, y)
# display the relative importance of each attribute
rank1 = np.argsort(model.feature_importances_)
print(rank1[-12:])
# plt.plot(model.feature_importances_)
# plt.savefig("feature_importances.png")

# --- RFE search
# model = LogisticRegression()
# # create the RFE model and select 3 attributes
# rfe = RFE(model, 3)
# rfe = rfe.fit(X, y)
# # summarize the selection of the attributes
# print(rfe.support_)
# print(rfe.ranking_)


# --- LassoCV
# X_train = X
# def rmse_cv(model):
#     rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 3))
#     return(rmse)

# #调用LassoCV函数，并进行交叉验证，默认cv=3
# #model_lasso = LassoCV(alphas = [0.1, 1, 0.001, 0.0005]).fit(X_train, y)
# model_lasso = LassoCV(alphas = [0.1, 1, 0.5, 0.25]).fit(X_train, y)
# #模型所选择的最优正则化参数alpha
# print(model_lasso.alpha_)

# #各特征列的参数值或者说权重参数，为0代表该特征被模型剔除了
# print(model_lasso.coef_)

# #输出看模型最终选择了几个特征向量，剔除了几个特征向量
# coef = pd.Series(model_lasso.coef_, index = X_train.columns)
# print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

# #输出所选择的最优正则化参数情况下的残差平均值，因为是3折，所以看平均值
# print(rmse_cv(model_lasso).mean())


# #画出特征变量的重要程度，这里面选出前3个重要，后3个不重要的举例
# imp_coef = pd.concat([coef.sort_values().head(10),
#                      coef.sort_values().tail(10)])

# matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
# imp_coef.plot(kind = "barh")
# plt.title("Coefficients in the Lasso Model")
# plt.savefig("LassoCV_20_big.png")

