from models import *
from utils import *
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import warnings
from datetime import datetime
from dotenv import load_dotenv
import os
from models import DecisionTree
from collections import Counter
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

load_dotenv()
api_key = os.getenv('API_KEY')
warnings.filterwarnings("ignore")

def initialize_results_dict(train_size_list):
    return {size: {'lm': [], 'dtree': [], 'true_dtree': []} for size in train_size_list}

def calculate_accuracy(predictions, labels):
    return np.mean([pred == label for pred, label in zip(predictions, labels)])

def calculate_log_loss(predictions, labels):
    res = np.mean([-np.log(pred) if label == 1 else -np.log(1-pred) for pred, label in zip(predictions, labels)])
    return res

class DataLoader:
    def __init__(self, filename, train_size, test_size):
        self.filename = filename
        self.train_size = train_size
        self.test_size = test_size
        self.data = self.load_data()
        self.train_data, self.test_data = self.train_test_split()
    
    def load_data(self):
        data = pd.read_csv(self.filename)
        data.dropna(inplace=True)
        data = data.sample(frac=1)
        # 50/50 split of labels
        data = balance_data(data)
        # handle categorical data
        return data
    
    def balance_data(self):
        return balance_data(self.data)
    
    def one_hot_encode_data(self):
        return one_hot_encode(self.data)
    
    def train_test_split(self):
        if self.train_size + self.test_size > len(self.data):
            print('Train size + test size is greater than data size. Setting train size to half of data size.')
            self.test_size = len(self.data) - self.train_size
        train_data = self.data.iloc[:self.train_size]
        test_data = self.data.iloc[-self.test_size:]
        return train_data, test_data


def test(task_name, max_depth, train_size=250, test_size=250, train_size_list=[4, 8, 16, 32, 64, 128], num_samples=50, verbose=False, temperature=0.7, prompt_num=1):
    # config data
    data_file = 'data/' + task_name + '.csv'
    data_loader = DataLoader(data_file, train_size, test_size)
    data, train_data, test_data = data_loader.data, data_loader.train_data, data_loader.test_data

    # train decision tree on full dataset for comparison, call it "true" dtree
    true_dtree = DecisionTree(data, max_depth=max_depth, task_name=task_name, lm=False, verbose=False)
    true_dtree_feat_importances_result = true_dtree.feature_importance
    print('Train data size:', len(train_data))
    print('Test data size:', len(test_data))
    print()
    test_data.reset_index(inplace=True, drop=True)
    results = []

    
    for train_size in train_size_list:
        lm_feat_importances_list = []
        dtree_feat_importances_list = []
        xgboost_feat_importances_list = []
        logreg_feat_importances_list = []

        for iter in range(num_samples):
            sample_data = train_data.sample(train_size)
            # force sample data to have at least 2 labels
            while len(sample_data['label'].unique()) < 2:
                sample_data = train_data.sample(train_size)
            lm_model = DecisionTree(sample_data, max_depth=max_depth, task_name=task_name, lm=True, verbose=verbose, prompt_num=prompt_num, temperature=temperature)
            dtree_model = DecisionTree(sample_data, max_depth=max_depth, task_name=task_name, lm=False, verbose=False)
            xgboost_model = XGBClassifier(random_state=0, max_depth=max_depth).fit(sample_data.drop('label', axis=1), sample_data['label'])
            logreg_model = LogisticRegression(random_state=0).fit(sample_data.drop('label', axis=1), sample_data['label'])
            lm_feat_importances = {key: lm_model.feature_importance[key] if key in lm_model.feature_importance else 0 for key in lm_model.feature_importance }
            lm_feat_importances_list.append(lm_feat_importances)
            dtree_feat_importances = {key: dtree_model.feature_importance[key] if key in dtree_model.feature_importance else 0 for key in dtree_model.feature_importance }
            dtree_feat_importances_list.append(dtree_feat_importances)
            xgboost_feat_importances = {key: value for key, value in zip(sample_data.drop('label', axis=1).columns, xgboost_model.feature_importances_)}
            xgboost_feat_importances_list.append(xgboost_feat_importances)
            # logreg_feat_importances_list.append(logreg_model.feature_importances_)

            labels = []
            # for printing to screen

            lm_preds, dtree_preds, true_dtree_preds, xgboost_model_preds, logreg_model_preds = [], [], [], [], []
            lm_pred_probas, dtree_pred_probas, true_dtree_pred_probas, xgboost_model_pred_probas, logreg_model_pred_probas = [], [], [], [], []
            for i, test_point in test_data.iterrows():
                label = int(test_point['label'])
                lm_pred = lm_model.predict(test_point.drop('label'))
                lm_pred_proba = lm_model.predict_proba(test_point.drop('label'))
                dtree_pred = dtree_model.predict(test_point.drop('label'))
                dtree_pred_proba = dtree_model.predict_proba(test_point.drop('label'))
                true_dtree_pred = true_dtree.predict(test_point.drop('label'))
                true_dtree_pred_proba = true_dtree.predict_proba(test_point.drop('label'))
                xgboost_model_pred = xgboost_model.predict(pd.DataFrame(test_point.drop('label')).T)[0]
                xgboost_model_pred_proba = xgboost_model.predict_proba(pd.DataFrame(test_point.drop('label')).T)[0][1]
                logreg_model_pred = logreg_model.predict(pd.DataFrame(test_point.drop('label')).T)[0]
                logreg_model_pred_proba = logreg_model.predict_proba(pd.DataFrame(test_point.drop('label')).T)[0][1]
                results.append({'train_size': train_size, 'iter': iter,
                                'lm_pred': lm_pred, 'lm_pred_proba': lm_pred_proba,
                                'dtree_pred': dtree_pred, 'dtree_pred_proba': dtree_pred_proba, 
                                'true_dtree_pred': true_dtree_pred, 'true_dtree_pred_proba': true_dtree_pred_proba, 
                                'xgboost_pred': xgboost_model_pred, 'xgboost_pred_proba': xgboost_model_pred_proba,
                                'logreg_pred': logreg_model_pred, 'logreg_pred_proba': logreg_model_pred_proba,
                                'label': label})
                labels.append(label)
                lm_preds.append(lm_pred)
                dtree_preds.append(dtree_pred)
                true_dtree_preds.append(true_dtree_pred)
                lm_pred_probas.append(lm_pred_proba)
                dtree_pred_probas.append(dtree_pred_proba)
                true_dtree_pred_probas.append(true_dtree_pred_proba)
                xgboost_model_preds.append(xgboost_model_pred)
                xgboost_model_pred_probas.append(xgboost_model_pred_proba)
                logreg_model_preds.append(logreg_model_pred)
                logreg_model_pred_probas.append(logreg_model_pred_proba)

            lm_acc = calculate_accuracy(lm_preds, labels)
            dtree_acc = calculate_accuracy(dtree_preds, labels)
            true_dtree_acc = calculate_accuracy(true_dtree_preds, labels)
            lm_roc_auc = roc_auc_score(labels, lm_pred_probas)
            dtree_roc_auc = roc_auc_score(labels, dtree_pred_probas)
            true_dtree_roc_auc = roc_auc_score(labels, true_dtree_pred_probas)
            xgboost_model_acc = calculate_accuracy(xgboost_model_preds, labels)
            xgboost_model_roc_auc = roc_auc_score(labels, xgboost_model_pred_probas)
            logreg_model_acc = calculate_accuracy(logreg_model_preds, labels)
            logreg_model_roc_auc = roc_auc_score(labels, logreg_model_pred_probas)

            print(f'Train Size: {train_size}, Iter {iter+1}/{num_samples}\n\
        \tAcc:\t\t LM: {lm_acc:.2f}, DTree: {dtree_acc:.2f}, True DTree: {true_dtree_acc:.2f}, XGBoost: {xgboost_model_acc:.2f}, LogReg: {logreg_model_acc:.2f}\n\
        \tLog Loss:\t LM: {calculate_log_loss(lm_pred_probas, labels):.2f}, DTree: {calculate_log_loss(dtree_pred_probas, labels):.2f}, True DTree: {calculate_log_loss(true_dtree_pred_probas, labels):.2f}, XGBoost: {calculate_log_loss(xgboost_model_pred_probas, labels):.2f}, LogReg: {calculate_log_loss(logreg_model_pred_probas, labels):.2f}\n\
        \tROC AUC:\t LM: {lm_roc_auc:.2f}, DTree: {dtree_roc_auc:.2f}, True DTree: {true_dtree_roc_auc:.2f}, XGBoost: {xgboost_model_roc_auc:.2f}, LogReg: {logreg_model_roc_auc:.2f}')
            print()



        # sum all of the feature importances and divide by num_samples
        # lm_feat_importances_list is a list of dictionaries
        lm_feat_importances_result = {}
        for i in range(len(lm_feat_importances_list)):
            for key in lm_feat_importances_list[i]:
                if key not in lm_feat_importances_result:
                    lm_feat_importances_result[key] = 0
                lm_feat_importances_result[key] += lm_feat_importances_list[i][key]
        for key in lm_feat_importances_result:
            lm_feat_importances_result[key] /= num_samples
        
        dtree_feat_importances_result = {}
        for i in range(len(dtree_feat_importances_list)):
            for key in dtree_feat_importances_list[i]:
                if key not in dtree_feat_importances_result:
                    dtree_feat_importances_result[key] = 0
                dtree_feat_importances_result[key] += dtree_feat_importances_list[i][key]
        for key in dtree_feat_importances_result:
            dtree_feat_importances_result[key] /= num_samples
        
        xgboost_feat_importances_result = {}
        for i in range(len(xgboost_feat_importances_list)):
            for key in xgboost_feat_importances_list[i]:
                if key not in xgboost_feat_importances_result:
                    xgboost_feat_importances_result[key] = 0
                xgboost_feat_importances_result[key] += xgboost_feat_importances_list[i][key]
        for key in xgboost_feat_importances_result:
            xgboost_feat_importances_result[key] /= num_samples
        
        # logreg_feat_importances_result = {}
        # for i in range(len(logreg_feat_importances_list)):
        #     for key in logreg_feat_importances_list[i]:
        #         if key not in logreg_feat_importances_result:
        #             logreg_feat_importances_result[key] = 0
        #         logreg_feat_importances_result[key] += logreg_feat_importances_list[i][key]

        print('Results for train size {}'.format(train_size))
        print('LM Feature Importances:', {key: round(lm_feat_importances_result[key], 2) for key in lm_feat_importances_result})
        print('DTree Feature Importances:', {key: round(dtree_feat_importances_result[key], 2) for key in dtree_feat_importances_result})
        print('True DTree Feature Importances:', {key: round(true_dtree_feat_importances_result[key], 2) for key in true_dtree_feat_importances_result})
        print()
        # save to csv
        dir = 'results/results4/feature-importances/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        lm_feat_importances_result_df = pd.DataFrame.from_dict(lm_feat_importances_result, orient='index', columns=['importance'])
        lm_feat_importances_result_df.to_csv(dir + 'lm_' + str(train_size) + '.csv')
        dtree_feat_importances_result_df = pd.DataFrame.from_dict(dtree_feat_importances_result, orient='index', columns=['importance'])
        dtree_feat_importances_result_df.to_csv(dir + 'dtree_' + str(train_size) + '.csv')
        true_dtree_feat_importances_result_df = pd.DataFrame.from_dict(true_dtree_feat_importances_result, orient='index', columns=['importance'])
        true_dtree_feat_importances_result_df.to_csv(dir + 'true_dtree_' + str(train_size) + '.csv')
        xgboost_feat_importances_result_df = pd.DataFrame.from_dict(xgboost_feat_importances_result, orient='index', columns=['importance'])
        xgboost_feat_importances_result_df.to_csv(dir + 'xgboost_' + str(train_size) + '.csv')
        # logreg_feat_importances_result_df = pd.DataFrame.from_dict(logreg_feat_importances_result, orient='index', columns=['importance'])
        # logreg_feat_importances_result_df.to_csv(dir + 'logreg_' + str(train_size) + '.csv')

    # save results
    results_df = pd.DataFrame(results)
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_save_filename = 'results/results4/results/' + task_name + '_' + time_str + '.csv'
    results_df.to_csv(results_save_filename, index=False)
    summary_df = results_df.copy()
    summary_df['lm_acc'] = summary_df['lm_pred'] == summary_df['label']
    summary_df['dtree_acc'] = summary_df['dtree_pred'] == summary_df['label']
    summary_df['true_dtree_acc'] = summary_df['true_dtree_pred'] == summary_df['label']
    summary_df['lm_log_loss'] = summary_df.apply(lambda row: calculate_log_loss([row['lm_pred_proba']], [row['label']]), axis=1)
    summary_df['dtree_log_loss'] = summary_df.apply(lambda row: calculate_log_loss([row['dtree_pred_proba']], [row['label']]), axis=1)
    summary_df['true_dtree_log_loss'] = summary_df.apply(lambda row: calculate_log_loss([row['true_dtree_pred_proba']], [row['label']]), axis=1)
    summary_df['lm_sim'] = summary_df['lm_pred'] == summary_df['true_dtree_pred']
    summary_df['dtree_sim'] = summary_df['dtree_pred'] == summary_df['true_dtree_pred']
    summary_df['xgboost_acc'] = summary_df['xgboost_pred'] == summary_df['label']
    summary_df['xgboost_log_loss'] = summary_df.apply(lambda row: calculate_log_loss([row['xgboost_pred_proba']], [row['label']]), axis=1)
    summary_df['xgboost_sim'] = summary_df['xgboost_pred'] == summary_df['true_dtree_pred']
    summary_df['logreg_acc'] = summary_df['logreg_pred'] == summary_df['label']
    summary_df['logreg_log_loss'] = summary_df.apply(lambda row: calculate_log_loss([row['logreg_pred_proba']], [row['label']]), axis=1)
    summary_df['logreg_sim'] = summary_df['logreg_pred'] == summary_df['true_dtree_pred']
    summary_df = summary_df.groupby('train_size').mean().drop('iter', axis=1)
    summary_df.to_csv('results/results4/summary/' + task_name + '_' + time_str + '_summary.csv')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--taskname', type=str, default='housing')
    parser.add_argument('--maxdepth', type=int, default=3)
    parser.add_argument('--trainsize', type=int, default=250)
    parser.add_argument('--testsize', type=int, default=250)
    parser.add_argument('--trainsizelist', type=str, default='4,8,16,32,64,128')
    parser.add_argument('--numsamples', type=int, default=20)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--temp', type=float, default=0.7)
    parser.add_argument('--promptnum', type=int, default=2)
    args = parser.parse_args()

    task_name = args.taskname
    max_depth = args.maxdepth
    train_size = args.trainsize
    test_size = args.testsize
    train_size_list = [int(size) for size in args.trainsizelist.split(',')]
    num_samples = args.numsamples
    verbose = args.verbose
    temp = args.temp
    prompt_num = args.promptnum

    test(task_name, max_depth, train_size, test_size, train_size_list, num_samples, verbose, temp, prompt_num)

