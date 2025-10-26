from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

iris_data = load_iris()
data = iris_data.data

target = iris_data.target
# 数据标准化
scaler = StandardScaler()
data_scaler = scaler.fit_transform(data)
# 分割数据
x_train, x_test, y_train, y_test = train_test_split(data_scaler, target, test_size = 0.2, random_state = 0)
# 网格调参
knn = KNeighborsClassifier(n_neighbors=19)
# 配置参数
knn.fit(x_train, y_train)
param_grid = [
    {
        'weights': ['uniform', 'distance'],
        'n_neighbors': [k for k in range(3, 20, 2)],
        'p':[p for p in range(1, 3)],
    }
]
# 代入网格模型
knncv = GridSearchCV(knn, param_grid=param_grid)
knncv.fit(x_train, y_train)
knncv.score(x_test, y_test)
print(f'beat_score: {knncv.best_score_}')
print(f'best_param: {knncv.best_params_}')
print(f'best_model: {knncv.best_estimator_}')


