from models import *
from utils import *
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt
import warnings
from datetime import datetime
from dotenv import load_dotenv
import os

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
        data = one_hot_encode(data)
        return data
    
    def balance_data(self):
        return balance_data(self.data)
    
    def one_hot_encode_data(self):
        return one_hot_encode(self.data)
    
    def train_test_split(self):
        if self.train_size + self.test_size > len(self.data):
            print('Train size + test size is greater than data size. Setting train size to half of data size.')
            self.train_size = len(self.data) // 2
            self.test_size = len(self.data) // 2
        train_data = self.data.iloc[:self.train_size]
        test_data = self.data.iloc[-self.test_size:]
        return train_data, test_data


def test(task_name, max_depth, train_size=250, test_size=250, train_size_list=[4, 8, 16, 32, 64, 128], num_samples=50):
    # config data
    data_file = 'data/' + task_name + '.csv'
    data_loader = DataLoader(data_file, train_size, test_size)
    data, train_data, test_data = data_loader.data, data_loader.train_data, data_loader.test_data

    # train deicison tree on full dataset for comparison, call it "true" dtree
    true_dtree = DecisionTreeClassifier(max_depth=max_depth)
    true_dtree.fit(data.drop('label', axis=1), data['label'])

    train_data = data.iloc[:train_size]
    test_data = data.iloc[-test_size:]
    print('Train data size:', len(train_data))
    print('Test data size:', len(test_data))
    print()
    test_data.reset_index(inplace=True, drop=True)
    assert len(test_data) == test_size
    results = []
    
    for train_size in train_size_list:
        for iter in range(num_samples):
            sample_data = train_data.sample(train_size)
            lm_model = DecisionTree(sample_data, max_depth=max_depth, task_name=task_name, lm=True)
            dtree_model = DecisionTree(sample_data, max_depth=max_depth, task_name=task_name, lm=False)
            
            labels = []
            # for printing to screen

            lm_preds, dtree_preds, true_dtree_preds = [], [], []
            lm_pred_probas, dtree_pred_probas, true_dtree_pred_probas = [], [], []
            for i, test_point in test_data.iterrows():
                label = int(test_point['label'])
                try:
                    lm_pred = lm_model.predict(test_point.drop('label'))
                    lm_pred_proba = lm_model.predict_proba(test_point.drop('label'))
                    dtree_pred = dtree_model.predict(test_point.drop('label'))
                    dtree_pred_proba = dtree_model.predict_proba(test_point.drop('label'))
                    true_dtree_pred = true_dtree.predict(test_point.drop('label').values.reshape(1, -1))[0]
                    true_dtree_pred_proba = true_dtree.predict_proba(test_point.drop('label').values.reshape(1, -1))[0][1]
                    results.append({'train_size': train_size, 'iter': iter, 'lm_pred': lm_pred, 'lm_pred_proba': lm_pred_proba,
                                    'dtree_pred': dtree_pred, 'dtree_pred_proba': dtree_pred_proba, 'true_dtree_pred': true_dtree_pred,
                                    'true_dtree_pred_proba': true_dtree_pred_proba, 'label': label})
                    labels.append(label)
                    lm_preds.append(lm_pred)
                    dtree_preds.append(dtree_pred)
                    true_dtree_preds.append(true_dtree_pred)
                    lm_pred_probas.append(lm_pred_proba)
                    dtree_pred_probas.append(dtree_pred_proba)
                    true_dtree_pred_probas.append(true_dtree_pred_proba)
                    
                except:
                    continue
            
            lm_acc = calculate_accuracy(lm_preds, labels)
            dtree_acc = calculate_accuracy(dtree_preds, labels)
            true_dtree_acc = calculate_accuracy(true_dtree_preds, labels)
            print(f'Train Size: {train_size}, Iter {iter+1}/{num_samples}\n\
                \tAcc:\t\t LM: {lm_acc:.2f}, DTree: {dtree_acc:.2f}, True DTree: {true_dtree_acc:.2f}\n\
                \tLog Loss:\t LM: {calculate_log_loss(lm_pred_probas, labels):.2f}, DTree: {calculate_log_loss(dtree_pred_probas, labels):.2f}, True DTree: {calculate_log_loss(true_dtree_pred_probas, labels):.2f}\n')

    # save results
    results_df = pd.DataFrame(results)
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_save_filename = 'results/' + task_name + '_' + time_str + '.csv'
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
    summary_df = summary_df.groupby('train_size').mean().drop('iter', axis=1)
    print(summary_df)
    summary_df.to_csv('results/' + task_name + '_' + time_str + '_summary.csv')

if __name__ == '__main__':
    for task in ['diabetes', 'wine', 'abalone']:
        test(task_name=task, max_depth=3, train_size_list=[4, 8, 16, 32, 64, 128])
    # for task in ['diabetes']:
    #     test(task_name=task, max_depth=3, train_size_list=[64], num_samples=1)

