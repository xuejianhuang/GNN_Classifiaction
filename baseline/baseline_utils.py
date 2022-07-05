from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler

def pca_components_analysis(n, x_train, x_test):
    pca = PCA(n_components=n)
    print("降维之前的维度：{}，{}".format(x_train.shape,x_test.shape))
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    print("降维之后的维度：{}，:{}".format(x_train.shape,x_test.shape))


    return x_train,x_test


def grid_search(estimator, param_grid, cv, x_train, x_test, y_train, y_test):
    gridsearch = GridSearchCV(estimator, param_grid, cv)
    gridsearch.fit(x_train, y_train)
    score = gridsearch.score(x_test, y_test)
    print("准确率为:\n", score)
    print("最好的参数为：\n", gridsearch.best_params_)

def under_sampling(x,y):
    print("下采样之前的维度x：{}，y:{}".format(x.shape,y.shape))
    rus = RandomUnderSampler(random_state=0)
    x_resampled, y_resampled = rus.fit_resample(x, y)
    print("下采样之后的维度x：{}，y:{}".format(x_resampled.shape, y_resampled.shape))
    return x_resampled,y_resampled
