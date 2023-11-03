import pandas as pd
import numpy as np
import openai
import os

task_factory = {
    'housing': "whether the median price of homes in a housing block is greater than $200,000",
    'medical': "whether a person in this hospital died",
    'titanic': 'whether a passenger on the Titanic survived',
    'weather': 'whether a city is in North America based on its weather patterns',
    'wine': 'whether a wine is of high quality based on its chemical properties',
    'abalone': 'whether an abalone is over 11 years old based on its physical properties',
    'glass': 'whether a piece of glass was manufactured for building windows or vehicle windows based on its chemical properties',
    'diabetes': 'whether a person has diabetes based on certain diagnostic measurements',
    'mushroom': 'whether a mushroom is poisonous',
    'adult': "whether an individual's income is greater than $50,000",
}

class Node:
    def __init__(self, feature=None, value=None, left=None, right=None, result=None, proba=None):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.result = result
        self.proba = proba
        self.is_leaf = self.result is not None

class DecisionTree:
    def __init__(self, training_data, max_depth, task_name, lm=False, verbose=False, prompt_num=1):
        self.training_data = training_data
        self.feature_importance = {}
        self.data_type_map = self.get_data_type_map(training_data)
        self.max_depth = max_depth
        self.lm = lm
        self.classification_task = task_factory[task_name]
        self.chat = GPT()
        self.prompt_num = prompt_num
        self.root = self.build_tree_iterative(self.training_data, verbose=verbose)

    @staticmethod
    def get_data_type_map(df):
        data_type_map = {}
        for feature in df.columns:
            # Check data type of the column
            dtype = df[feature].dtype
            if np.issubdtype(dtype, np.number):
                data_type_map[feature] = 'numerical'
            else:
                data_type_map[feature] = 'categorical'
        return data_type_map


    def get_proba(self, data):
        # proba is proportion of data that has label 1
        if len(data) == 0:
            return 0
        elif data['label'].nunique() == 1:
            return np.clip(data['label'].iloc[0], .01, .99)
        else:
            return data['label'].mean()


    def entropy(self, data):
        total = len(data)
        counts = data.value_counts()
        probs = counts / total
        entropy = -sum(p * np.log2(p) for p in probs)
        return entropy

    def get_feature(self, data):
        original_entropy = self.entropy(data['label'])
        best_gain = 0
        best_feature = None
        # if all labels are the same,
        #  return the first feature
        # TODO: why does this happen?
        assert data['label'].nunique() > 1, data.head()
        if data['label'].nunique() == 1:
            return None
        for feature in data.columns:
            if feature != 'label':
                unique_values = data[feature].unique()
                for value in unique_values:
                    split1 = data[data[feature] <= value]
                    split2 = data[data[feature] > value]
                    total_weighted_entropy = sum((len(split) / len(data) * self.entropy(split['label'])) for split in [split1, split2])
                    gain = original_entropy - total_weighted_entropy
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
        return best_feature

    def get_value(self, data, feature):
        if self.data_type_map[feature] == 'numerical':
            return self.get_value_numerical(data, feature)
        else:
            return self.get_value_categorical(data, feature)

    def get_value_numerical(self, data, feature):
        unique_values = sorted(list(data[feature].unique()))
        if len(unique_values) == 1:
            return 0
        possible_splits = [(unique_values[i] + unique_values[i+1]) / 2 for i in range(len(unique_values) - 1)]
        best_gain = -float('inf')
        best_value = None
        original_entropy = self.entropy(data['label'])
        for value in possible_splits:
            split1 = data[data[feature] <= value]
            split2 = data[data[feature] > value]
            total_weighted_entropy = sum((len(split) / len(data) * self.entropy(split['label'])) for split in [split1, split2])
            gain = original_entropy - total_weighted_entropy
            if gain > best_gain:
                best_gain = gain
                best_value = value
        if feature not in self.feature_importance:
            self.feature_importance[feature] = best_gain
        else:
            self.feature_importance[feature] += best_gain
        return round(best_value, 3)
        
    def get_value_categorical(self, data, feature):
        unique_values = data[feature].unique()
        best_gain = -float('inf')
        best_value = None
        original_entropy = self.entropy(data['label'])
        for value in unique_values:
            split1 = data[data[feature] == value]
            split2 = data[data[feature] != value]
            total_weighted_entropy = sum((len(split) / len(data) * self.entropy(split['label'])) for split in [split1, split2])
            gain = original_entropy - total_weighted_entropy
            if gain > best_gain:
                best_gain = gain
                best_value = value

        if feature not in self.feature_importance:
            self.feature_importance[feature] = best_gain
        else:
            self.feature_importance[feature] += best_gain
        return best_value
    
    def build_tree_iterative(self, data, verbose=False):
        self.root = Node()
        stack = [(self.root, data, 0)]
        while stack:
            current_node, current_data, current_depth = stack.pop()
            current_node.dist = (current_data['label'].value_counts()).to_dict()
            # Base cases
            if len(current_data['label'].unique()) == 1:
                current_node.result = current_data['label'].iloc[0]
                current_node.proba = self.get_proba(current_data)
                continue
            if current_depth == self.max_depth:
                current_node.result = current_data['label'].value_counts().idxmax()
                current_node.proba = self.get_proba(current_data)
                continue
            # Feature selection
            if self.lm:
                check = False
                while check == False:
                    try:
                        prompt = self.create_prompt(data, self.root, 0, None, current_node)
                        # print(prompt)
                        if verbose:
                            print(prompt)
                            print()
                        feature_response = self.chat(prompt)
                        if verbose:
                            print(feature_response)
                            print()
                            print()
                        best_feature = self.parse_response(feature_response)
                        assert best_feature in data.columns
                        check = True
                    except:
                        print('retry')
                        continue
            else:
                best_feature = self.get_feature(current_data)
            assert best_feature in data.columns
            assert best_feature is not None
            best_value = self.get_value(current_data, best_feature)
            if self.data_type_map[best_feature] == 'numerical':
                left_data = current_data[current_data[best_feature] <= best_value]
                right_data = current_data[current_data[best_feature] > best_value]
            else:
                left_data = current_data[current_data[best_feature] == best_value]
                right_data = current_data[current_data[best_feature] != best_value]
            if len(left_data) == 0 or len(right_data) == 0:
                current_node.result = current_data['label'].value_counts().idxmax()
                current_node.proba = self.get_proba(current_data)
                continue
   
            # Update current node and push children to stack
            current_node.feature = best_feature
            current_node.value = best_value
            assert current_node.feature is not None
            assert current_node.value is not None
            current_node.left = Node()
            stack.append((current_node.left, left_data, current_depth + 1))
            current_node.right = Node()
            stack.append((current_node.right, right_data, current_depth + 1))
        # normalize feature_importances
        self.feature_importance = {k: v / sum(self.feature_importance.values()) for k, v in self.feature_importance.items()}
        return self.root


    def predict(self, test_point, node=None):
        if node is None:
            node = self.root
        
        if node.result is not None:
            return node.result
        if node.feature is None:
            print(self.print_tree())
        if self.data_type_map[node.feature] == 'numerical':
            if test_point[node.feature] <= node.value:
                return self.predict(test_point, node.left)
            else:
                return self.predict(test_point, node.right)
        else:
            if test_point[node.feature] == node.value:
                return self.predict(test_point, node.left)
            else:
                return self.predict(test_point, node.right)
        
    def predict_proba(self, test_point, node=None):
        if node is None:
            node = self.root

        if node.proba is not None:
            return node.proba
        
        if self.data_type_map[node.feature] == 'numerical':
            if test_point[node.feature] <= node.value:
                return self.predict_proba(test_point, node.left)
            else:
                return self.predict_proba(test_point, node.right)
        else:
            if test_point[node.feature] == node.value:
                return self.predict_proba(test_point, node.left)
            else:
                return self.predict_proba(test_point, node.right)


            
    def print_tree(self, node=None, depth=0, buffer=None, current_node=None):
        buffer = buffer or []

        if node is None:
            node = self.root

        if node == current_node:
            highlight = "[CURRENT NODE] "
        else:
            highlight = ""

        if node.result is not None:
            buffer.append((' ' * depth + highlight + f"Result: {node.result}"))
        else:
            if node.feature is not None:
                if self.data_type_map[node.feature] == 'numerical':
                    buffer.append((' ' * depth + highlight + f"{node.feature} <= {node.value}"))
                    # buffer.append((' ' * depth + highlight + f"{node.feature} <= {node.value}" + f" Distribution: {node.dist}"))

                    self.print_tree(node.left, depth+2, buffer, current_node)
                    buffer.append((' ' * depth + highlight + f"{node.feature} > {node.value}"))
                    # buffer.append((' ' * depth + highlight + f"{node.feature} > {node.value}" + f" Distribution: {node.dist}"))
                    self.print_tree(node.right, depth+2, buffer, current_node)
                else:
                    buffer.append((' ' * depth + highlight + f"{node.feature} == {node.value}"))
                    # buffer.append((' ' * depth + highlight + f"{node.feature} <= {node.value}" + f" Distribution: {node.dist}"))

                    self.print_tree(node.left, depth+2, buffer, current_node)
                    buffer.append((' ' * depth + highlight + f"{node.feature} != {node.value}"))
                    # buffer.append((' ' * depth + highlight + f"{node.feature} > {node.value}" + f" Distribution: {node.dist}"))
                    self.print_tree(node.right, depth+2, buffer, current_node)
            else:
                buffer.append((' ' * depth + highlight + "Split on ?"))

        return '\n'.join(buffer)


    @staticmethod
    def parse_response(response):
        # response should end with "Feature: <feature>"
        response = response[response.find('Feature:') + len('Feature:'):].strip()
        response = response.replace("','", "")
        response = response.replace('"', "")
        response = response.replace("'", "")
        response = response.replace(".", "")
        return response

    def read_prompt(self, filename):
        with open(filename, 'r') as f:
            prompt = f.read()
        return prompt
    
    def add_parameters_to_prompt(self, prompt, **kwargs):
        for key, value in kwargs.items():
            prompt = prompt.replace(f"#{key}#", str(value))
        return prompt

    def create_prompt(self, data, node, depth, buffer=None, current=None):
        # Construct the tree's current state
        filename = f'prompts/prompt{self.prompt_num}.txt'
        task_description = self.classification_task
        feature_list = [col for col in list(data.columns) if col != 'label']
        tree_state = self.print_tree(node, depth, buffer, current)
        prompt_template = self.read_prompt(filename)
        prompt = self.add_parameters_to_prompt(prompt_template, task_description=task_description, feature_list=feature_list, tree_state=tree_state)
        return prompt
    
    def list_features(self, features=[], node='root'):
        if node == 'root':
            node = self.root
        if node is not None:
            return features + [node.feature for node in [node] if node.feature is not None] + self.list_features(features, node.left) + self.list_features(features, node.right)
        else:
            return features


class GPT():
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv(override=True)
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = 'gpt-3.5-turbo'

    def __call__(self, prompt, **kwargs):
        # This is just an example, you can construct the prompt based on your requirement
        messages = [{'role': 'user', 'content': prompt}]
        response = openai.ChatCompletion.create(messages=messages, model=self.model, api_key=self.api_key, **kwargs, temperature=0)
        return response.choices[0].message['content']


class Forest():
    def __init__(self, data, num_trees, max_depth, task_name, sample_frac=1, lm=False):
        self.data = data
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.task_name = task_name
        self.sample_frac = sample_frac
        self.data_type_map = None
        self.classification_task = None
        self.lm = lm
        self.gpt = GPT()
        self.trees = self.build_trees()

    def build_single_tree(self, training_data):
        tree = DecisionTree(training_data, max_depth=self.max_depth, task_name=self.task_name, lm=self.lm)
        return tree

    def build_trees(self):
        sample_frac = self.sample_frac
        trees = []
        for _ in range(self.num_trees):
            print(f'Building tree {_+1} of {self.num_trees}', end='\r')
            sample = self.data.sample(frac=sample_frac)
            trees.append(self.build_single_tree(sample))
        return trees

    def predict(self, test_point):
        predictions = []
        for tree in self.trees:
            predictions.append(DecisionTree.predict(test_point, tree))
        return max(set(predictions), key=predictions.count)

    def predict_proba(self, test_point):
        predictions = []
        for tree in self.trees:
            predictions.append(DecisionTree.predict(test_point, tree))
        return predictions.count(1) / len(predictions)

    def get_node_from_position(self, position: str, node) -> Node:
        if node is None:
            return None
        if position == '':
            return node
        else:
            if position[0] == '0':
                return self.get_node_from_position(position[1:], node.left)
            else:
                return self.get_node_from_position(position[1:], node.right)

    def get_prob_next_nodes(self, position: str, trees: list) -> dict:
        '''
        Here is how position works:
        '': root
        '0': left child of root
        '1': right child of root
        '00': left child of left child of root
        '01': right child of left child of root
        '10': left child of right child of root
        '11': right child of right child of root
        '''
        results = []
        for tree in trees:
            root = tree.root
            node = self.get_node_from_position(position, root)
            assert node is not None
            if node.is_leaf:
                results.append(None)
            else:
                results.append(node.feature)
        return {feature: results.count(feature) / len(results) for feature in set(results)}

    def get_prob_next_nodes_conditioned_on_features(self, position_feature_dict: dict, test_position) -> dict:
        '''
        Conditions the trees based on the features and positions given.
        Then returns the probability of the next nodes at a given position.
        '''
        position_feature_lst = sorted(list(position_feature_dict.items()), key=lambda x: len(x[0]))
        sample_trees = self.trees.copy()
        for position, feature in position_feature_lst:
            new_sample_trees = []
            for tree in sample_trees:
                if self.get_node_from_position(position, tree.root).feature == feature:
                    new_sample_trees.append(tree)
            sample_trees = new_sample_trees
        
        return self.get_prob_next_nodes(test_position, sample_trees)
        
    