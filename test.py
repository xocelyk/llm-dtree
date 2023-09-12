from models import *
# import decision tree from scikit-learn
from sklearn.tree import DecisionTreeClassifier
data_file = 'data/housing.csv'
data = pd.read_csv(data_file)
data.dropna(inplace=True)
data = data.sample(frac=1)
print(data['label'].value_counts())
assert False
true_dtree = DecisionTreeClassifier(max_depth=3)
true_dtree.fit(data.drop('label', axis=1), data['label'])

test_data = data.iloc[-1000:]
test_data.reset_index(inplace=True, drop=True)
data = data.iloc[:-1000]

num_samples = 10
train_size_list = [32, 64]
max_depth = 3
results_header = ['train_size', 'iter', 'test_point', 'lm_pred', 'dtree_pred', 'true_dtree_pred', 'label']
results = []


lm_results = {}
dtree_results = {}
true_dtree_results = {}
for train_size in train_size_list:
    lm_results[train_size] = []
    dtree_results[train_size] = []
    true_dtree_results[train_size] = []
    for iter in range(num_samples):
        print(iter)
        train_data = data.sample(train_size)
        lm_acc_list = []
        dtree_acc_list = []
        true_dtree_acc_list = []
        lm_model = DecisionTree(train_data, max_depth=max_depth, lm=True)
        dtree_model = DecisionTree(train_data, max_depth=max_depth, lm=False)

        for i in range(len(test_data)):
            test_point = test_data.iloc[i]
            label = test_point['label']
            lm_pred = lm_model.predict(test_point.drop('label'))
            dtree_pred = dtree_model.predict(test_point.drop('label'))
            true_dtree_pred = true_dtree.predict(test_point.drop('label').values.reshape(1, -1))[0]
            results.append([train_size, iter, i, lm_pred, dtree_pred, true_dtree_pred, label])
            # print(results[-1])



results_df = pd.DataFrame(results, columns=results_header)
results_df.to_csv('results.csv', index=False)




