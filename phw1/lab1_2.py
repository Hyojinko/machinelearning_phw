import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


#load Data
df = pd.read_csv('breast-cancer-wisconsin.csv')
df.columns = ['Sample code number','Clump Thickness', 'Uniformity of Cell Size',
                         'Uniformity of Cell Shape', 'Marginal Adhension','Single Epithelial Cell Size',
                         'Bare Nuclei', 'Bland Chromatin','Normal Nucleoli','Mitoses','Class']
print(df.dtypes)
print(df.isnull().any(0))

def ordinal_encode(df):
    ordinalEncoder = preprocessing.OrdinalEncoder()
    df= ordinalEncoder.fit_transform(df)
    df = pd.DataFrame(df)


def oneHot_encode(df):
    onehotEncoder = preprocessing.OneHotEncoder()
    df = onehotEncoder.fit_transform(df)
    df = pd.DataFrame(df)




#
# def ordinal_scale(df, targetName):
#     ordinalEncoder = preprocessing.OrdinalEncoder()
#     X = pd.DataFrame(df[targetName])
#     ordinalEncoder.fit(X)
#     df[targetName] = pd.DataFrame(ordinalEncoder.transform(X))
#
# def oneHot_scale(df, targetName):
#     oneHotEncoder = preprocessing.OneHotEncoder()
#     X = pd.DataFrame(df[targetName])
#     oneHotEncoder.fit(X)
#     df[targetName] = pd.DataFrame(oneHotEncoder.transform(X))

# def label_scale(df, targetName):
#     labelEncoder = preprocessing.LabelEncoder()
#     X = pd.DataFrame(df[targetName])
#     labelEncoder.fit(X)
#     df[targetName] = pd.DataFrame(labelEncoder.transform(X))








def scale_module(df, targetName):
    #Encoding data
    y_ordinal=df[targetName]
    X_ordinal=df.drop([targetName],1)
    ordinal_encode(X_ordinal)
    X_train_ordinal, X_test_ordinal, y_train_ordinal, y_test_ordinal = train_test_split(X_ordinal,y_ordinal,random_state=0)

    y_oneHot = df[targetName]
    X_oneHot = df.drop([targetName], 1)
    oneHot_encode(X_oneHot)
    X_train_oneHot, X_test_oneHot, y_train_oneHot, y_test_oneHot = train_test_split(X_oneHot, y_oneHot,
                                                                                        random_state=0)

    # Normalization with 4 Scaling methods
    maxAbsScaler = preprocessing.MaxAbsScaler()
    minmaxScaler = preprocessing.MinMaxScaler()
    robustScaler = preprocessing.RobustScaler()
    standardScaler = preprocessing.StandardScaler()

    #Scaling ordinal encoded data
    df_maxAbs_ordinal_train = maxAbsScaler.fit_transform(X_train_ordinal)
    df_maxAbs_ordinal_train = pd.DataFrame(df_maxAbs_ordinal_train, columns=X_train_ordinal.columns)
    df_maxAbs_ordinal_test = maxAbsScaler.fit_transform(X_test_ordinal)
    df_maxAbs_ordinal_test = pd.DataFrame(df_maxAbs_ordinal_test, columns=X_test_ordinal.columns)

    df_minMax_ordinal_train = minmaxScaler.fit_transform(X_train_ordinal)
    df_minMax_ordinal_train = pd.DataFrame(df_minMax_ordinal_train, columns=X_train_ordinal.columns)
    df_minMax_ordinal_test = minmaxScaler.fit_transform(X_test_ordinal)
    df_minMax_ordinal_test = pd.DataFrame(df_minMax_ordinal_test, columns=X_test_ordinal.columns)

    df_robust_ordinal_train = robustScaler.fit_transform(X_train_ordinal)
    df_robust_ordinal_train = pd.DataFrame(df_robust_ordinal_train, columns=X_train_ordinal.columns)
    df_robust_ordinal_test = robustScaler.fit_transform(X_test_ordinal)
    df_robust_ordinal_test = pd.DataFrame(df_robust_ordinal_test, columns=X_test_ordinal.columns)

    df_standard_ordinal_train = standardScaler.fit_transform(X_train_ordinal)
    df_standard_ordinal_train = pd.DataFrame(df_standard_ordinal_train, columns=X_train_ordinal.columns)
    df_standard_ordinal_test = standardScaler.fit_transform(X_test_ordinal)
    df_standard_ordinal_test = pd.DataFrame(df_standard_ordinal_test, columns=X_test_ordinal.columns)

    # Scaling oneHot encoded data
    df_maxAbs_oneHot_train = maxAbsScaler.fit_transform(X_train_oneHot)
    df_maxAbs_oneHot_train = pd.DataFrame(df_maxAbs_oneHot_train, columns=X_train_oneHot.columns)
    df_maxAbs_oneHot_test = maxAbsScaler.fit_transform(X_test_oneHot)
    df_maxAbs_oneHot_test = pd.DataFrame(df_maxAbs_oneHot_test, columns=X_test_oneHot.columns)

    df_minMax_oneHot_train = minmaxScaler.fit_transform(X_train_oneHot)
    df_minMax_oneHot_train = pd.DataFrame(df_minMax_oneHot_train, columns=X_train_oneHot.columns)
    df_minMax_oneHot_test = minmaxScaler.fit_transform(X_test_oneHot)
    df_minMax_oneHot_test = pd.DataFrame(df_minMax_oneHot_test, columns=X_test_oneHot.columns)

    df_robust_oneHot_train = robustScaler.fit_transform(X_train_oneHot)
    df_robust_oneHot_train = pd.DataFrame(df_robust_oneHot_train, columns=X_train_oneHot.columns)
    df_robust_oneHot_test = robustScaler.fit_transform(X_test_oneHot)
    df_robust_oneHot_test = pd.DataFrame(df_robust_oneHot_test, columns=X_test_oneHot.columns)

    df_standard_oneHot_train = standardScaler.fit_transform(X_train_oneHot)
    df_standard_oneHot_train = pd.DataFrame(df_standard_oneHot_train, columns=X_train_oneHot.columns)
    df_standard_oneHot_test = standardScaler.fit_transform(X_test_oneHot)
    df_standard_oneHot_test = pd.DataFrame(df_standard_oneHot_test, columns=X_test_oneHot.columns)

    # Alogrithm
    print("\n------------------------- Using maxAbs scaled dataset -------------------------")
    max_score_maxAbs_ordinal = algorithm_module(df_maxAbs_ordinal_train, df_maxAbs_ordinal_test, y_train_ordinal, y_test_ordinal)
    print("\n------------------------- Using minMax scaled dataset -------------------------")
    max_score_minMax_ordinal = algorithm_module(df_minMax_ordinal_train, df_minMax_ordinal_test, y_train_ordinal, y_test_ordinal)
    print("\n------------------------- Using robust scaled dataset -------------------------")
    max_score_robust_ordinal = algorithm_module(df_robust_ordinal_train, df_robust_ordinal_test, y_train_ordinal, y_test_ordinal)
    print("\n------------------------- Using standard scaled dataset -------------------------")
    max_score_standard_ordinal = algorithm_module(df_standard_ordinal_train, df_standard_ordinal_test, y_train_ordinal, y_test_ordinal)

    # Result
    max_score_result_oneHot = max(max_score_maxAbs_oneHot, max_score_minMax_oneHot, max_score_robust_oneHot, max_score_standard_oneHot)
    print("\n\n============================== oneHot encoded Result ==============================")
    print("Final maximum score for oneHot encoded: %.6f" % max_score_result_oneHot)

    print("\n------------------------- Using maxAbs scaled dataset -------------------------")
    max_score_maxAbs_oneHot = algorithm_module(df_maxAbs_oneHot_train, df_maxAbs_oneHot_test, y_train_oneHot,
                                                y_test_oneHot)
    print("\n------------------------- Using minMax scaled dataset -------------------------")
    max_score_minMax_oneHot = algorithm_module(df_minMax_oneHot_train, df_minMax_oneHot_test, y_train_oneHot,
                                                y_test_oneHot)
    print("\n------------------------- Using robust scaled dataset -------------------------")
    max_score_robust_oneHot = algorithm_module(df_robust_oneHot_train, df_robust_oneHot_test, y_train_oneHot,
                                                y_test_oneHot)
    print("\n------------------------- Using standard scaled dataset -------------------------")
    max_score_standard_oneHot = algorithm_module(df_standard_oneHot_train, df_standard_oneHot_test, y_train_oneHot,
                                                  y_test_oneHot)

    # Result
    max_score_result_oneHot = max(max_score_maxAbs_oneHot, max_score_minMax_oneHot, max_score_robust_oneHot,
                                   max_score_standard_oneHot)
    print("\n\n============================== oneHot encoded Result ==============================")
    print("Final maximum score for oneHot encoded: %.6f" % max_score_result_oneHot)


def algorithm_module(X_train, X_test, y_train, y_test):
    #Decision tree classifier
    dt_params = {"max_depth": [2,3,4],
                 "max_features": randint(1,10),
                 "min_samples_leaf":randint(1,10),
                 "criterion": ["gini",'entropy']}
    tree_clf = DecisionTreeClassifier()
    fold5 = KFold(n_splits=5, shuffle=True, random_state=0)
    fold10 = KFold(n_splits=10, shuffle=True, random_state=0)
    grid_cv = GridSearchCV(tree_clf, param_grid=dt_params, scoring='accuracy', cv=fold5)
    grid_cv.fit(X_train,y_train)
    print("Best score of Decision tree is(cv=5): {}".format(grid_cv.best_score_))
    print("Best parameter of Decision tree is(cv=5): {}".format(grid_cv.best_params_))
    fold5_dt_bestScore=grid_cv.best_score
    grid_cv = GridSearchCV(tree_clf, param_grid=dt_params, scoring='accuracy', cv=fold10)
    grid_cv.fit(X_train, y_train)
    print("Best score of Decision tree is(cv=10): {}".format(grid_cv.best_score_))
    print("Best parameter of Decision tree is(cv=10): {}".format(grid_cv.best_params_))
    fold10_dt_bestScore=grid_cv.best_score
    max_dt_score = max(fold10_dt_bestScore,fold5_dt_bestScore)
    #Logistic regression
    log_param = {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'max_iter': [100,1000]
    }
    log_reg = LogisticRegssion()
    grid_cv = GridSearchCV(log_reg, param_grid=log_param, scoring='accuracy', cv=fold5)
    grid_cv.fit(X_train, y_train)
    print("Best score of Logistic Regression is(cv=5): {}".format(grid_cv.best_score_))
    print("Best parameter of Logistic Regression is(cv=5): {}".format(grid_cv.best_params_))
    fold5_log_bestScore = grid_cv.best_score
    grid_cv = GridSearchCV(log_reg, param_grid=log_param, scoring='accuracy', cv=fold10)
    grid_cv.fit(X_train, y_train)
    print("Best score of Logistic Regression is(cv=10): {}".format(grid_cv.best_score_))
    print("Best parameter of Logistic Regression is(cv=10): {}".format(grid_cv.best_params_))
    fold10_log_bestScore = grid_cv.best_score
    max_log_score = max(fold10_log_bestScore,fold5_log_bestScore)

    #SVM
    svm_clf = SVC()
    svm_params = {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'gamma':[0.01,0.1,1.0,10.0]
    }
    grid_cv = GridSearchCV(svm_clf, param_grid=svm_params, scoring='accuracy', cv=fold5)
    grid_cv.fit(X_train, y_train)
    print("Best score of SVM is(cv=5): {}".format(grid_cv.best_score_))
    print("Best parameter of SVM is(cv=5): {}".format(grid_cv.best_params_))
    fold5_svm_bestScore = grid_cv.best_score
    grid_cv = GridSearchCV(svm_clf, param_grid=svm_params, scoring='accuracy', cv=fold10)
    grid_cv.fit(X_train, y_train)
    print("Best score of SVM is(cv=10): {}".format(grid_cv.best_score_))
    print("Best parameter of SVM is(cv=10): {}".format(grid_cv.best_params_))
    fold10_svm_bestScore = grid_cv.best_score
    max_svm_score = max(fold5_svm_bestScore, fold10_svm_bestScore)

    max_score = max(max_dt_score, score_poly, max_log_score, max_svm_score)
    return max_score

print(scale_module(df, "Class"))




