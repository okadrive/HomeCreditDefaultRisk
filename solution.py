# public score: 0.7589
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

# csv ファイルパス，随時変更
file_path = 'drive/My Drive/Colab Notebooks/competition2/input/'


def one_hot_encoder(data, nan_as_category=True):
    original_columns = list(data.columns)
    categorical_columns = [f for f in data.columns if df[f].dtype == 'object']
    for c in categorical_columns:
        if nan_as_category:
            data[c].fillna('NaN', inplace=True)
        values = list(data[c].unique())
        for v in values:
            data[str(c) + '_' + str(v)] = (data[c] == v).astype(np.uint8)
    data.drop(categorical_columns, axis=1, inplace=True)
    return data, [c for c in data.columns if c not in original_columns]


def application_train_test(file_path=file_path, nan_as_category=True):
    # ファイル読み込み
    df = pd.read_csv(file_path + 'train.csv')
    df_test = pd.read_csv(file_path + 'test.csv')

    df = df.append(df_test).reset_index()

    # Remove some rows with values not present in test set
    df = df[df['CODE_GENDER'] != 'XNA']
    df.drop(df[df['NAME_FAMILY_STATUS'] == 'Unknown'].index, inplace=True)

    # Categorical features with Binary encode (0 or 1; two categories)
    bin_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
    for bin_feature in bin_cols:
        df[bin_feature], _ = pd.factorize(df[bin_feature])

    # Categorical features with One-Hot encode
    cat_cols = [f for f in df.columns if df[f].dtype == 'object']
    for cat_feature in cat_cols:
        df[cat_feature], _ = pd.factorize(df[cat_feature])

    #df, _ = one_hot_encoder(df, nan_as_category)

    # Replace some outliers
    # https://www.kaggle.com/aantonova/797-lgbm-and-bayesian-optimization
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df.loc[df['OWN_CAR_AGE'] > 80, 'OWN_CAR_AGE'] = np.nan
    df.loc[df['REGION_RATING_CLIENT_W_CITY'] < 0,
           'REGION_RATING_CLIENT_W_CITY'] = np.nan
    df.loc[df['AMT_INCOME_TOTAL'] > 1e8, 'AMT_INCOME_TOTAL'] = np.nan
    df.loc[df['AMT_REQ_CREDIT_BUREAU_QRT'] > 10,
           'AMT_REQ_CREDIT_BUREAU_QRT'] = np.nan
    df.loc[df['OBS_30_CNT_SOCIAL_CIRCLE'] > 40,
           'OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan

    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_PERC'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']

    # original
    return df


df = application_train_test(file_path, True)

# ベースラインモデルの構築

# 必要なライブラリ等のインポート


def kfold_lightgbm(df, num_folds, debug=False):
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(
        train_df.shape, test_df.shape))

    # Cross validation model
    folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        # Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
        clf = lgb.LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.03,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

        clf.fit(train_x, train_y,
                eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc',
                verbose=200,
                early_stopping_rounds=200)

        oof_preds[valid_idx] = clf.predict_proba(
            valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[
            :, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' %
              (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

    # ファイル格納先，随時変更
    submission_file_name = 'drive/My Drive/Colab Notebooks/competition2/default_submission.csv'
    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(
            submission_file_name, index=False)

        submission = pd.read_csv(submission_file_name)
        print(submission)

    display_importances(feature_importance_df)
    return feature_importance_df


def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby(
        "feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(
        cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(
        by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')


kfold_lightgbm(df, num_folds=10, debug=False)
