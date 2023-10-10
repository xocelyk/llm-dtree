import pandas as pd
import numpy as np
import openai
import os

task_factory = {
    'housing': "whether the median price of a housing block is greater than $200,000",
    'medical': "whether a person in this hospital died",
    'titanic': 'whether a passenger on the Titanic survived',
    'weather': 'whether a city is in North America based on its weather patterns',
    'wine': 'whether a wine is of high quality based on its chemical properties',
    'abalone': 'whether an abalone is over 11 years old based on its physical properties',
    'glass': 'whether a piece of glass was manufactured for building windows or vehicle windows based on its chemical properties',
    'diabetes': 'whether a person has diabetes based on certain diagnostic measurements',
}

class DecisionTree:
    def __init__(self, training_data, max_depth, task_name, lm=False):
        self.training_data = training_data
        self.max_depth = max_depth
        self.lm = lm
        self.classification_task = task_factory[task_name]
        self.chat = GPT()
        self.root = self.build_tree_iterative(self.training_data)

    class Node:
        def __init__(self, feature=None, value=None, left=None, right=None, result=None, proba=None):
            self.feature = feature
            self.value = value
            self.left = left
            self.right = right
            self.result = result
            self.proba = proba

    def get_proba(self, data):
        # proba is proportion of data that has label 1
        if len(data) == 0:
            return 0
        elif data['label'].nunique() == 1:
            return data['label'].iloc[0]
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
        unique_values = sorted(list(data[feature].unique()))
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
        return best_value
    
    
    def build_tree_iterative(self, data):
        self.root = self.Node()
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
                        feature_response = self.chat(prompt)
                        best_feature = self.parse_response(feature_response)
                        assert best_feature in data.columns
                        check = True
                    except:
                        print('retry')
                        continue
            else:
                best_feature = self.get_feature(current_data)
            best_value = self.get_value(current_data, best_feature)
            left_data = current_data[current_data[best_feature] <= best_value]
            right_data = current_data[current_data[best_feature] > best_value]
   
            # Update current node and push children to stack
            current_node.feature = best_feature
            current_node.value = best_value
            current_node.left = self.Node()
            current_node.right = self.Node()
            if len(left_data) > 0:
                stack.append((current_node.left, left_data, current_depth + 1))
            if len(right_data) > 0:
                stack.append((current_node.right, right_data, current_depth + 1))
        return self.root


    def predict(self, test_point, node=None):
        if node is None:
            node = self.root
        
        if node.result is not None:
            return node.result
        if test_point[node.feature] <= node.value:
            return self.predict(test_point, node.left)
        else:
            return self.predict(test_point, node.right)
        
    def predict_proba(self, test_point, node=None):
        if node is None:
            node = self.root

        if node.proba is not None:
            return node.proba
        if test_point[node.feature] <= node.value:
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
                buffer.append((' ' * depth + highlight + f"{node.feature} <= {node.value}"))
                # buffer.append((' ' * depth + highlight + f"{node.feature} <= {node.value}" + f" Distribution: {node.dist}"))

                self.print_tree(node.left, depth+2, buffer, current_node)
                buffer.append((' ' * depth + highlight + f"{node.feature} > {node.value}"))
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


    def create_prompt(self, data, node, depth, buffer=None, current=None):
        # Construct the tree's current state
        tree_state = self.print_tree(node, depth, buffer, current)
        # print(tree_state)
        # print()
        # print()
        prompt = ''
        prompt += 'You are trying to classify {}.'.format(self.classification_task)
        prompt += '\n'
        prompt += f"Given the current tree state:\n{tree_state}\n"
        prompt += f"and a dataset with features {[col for col in list(data.columns) if col != 'label']} and label distribution {data['label'].value_counts().to_dict()}, which feature should we split on next?"
        prompt += '\n'
        prompt += 'Consider which features you think are most important to the classification task. If the feature has already been used in the decision tree, you should be less likely to use it again.'
        prompt += '\n'
        prompt += 'Explain you reasoning in one or two sentences, and then provide the feature name. Make sure the feature name matches one of the features contained in the feature list. Your response should end with "Feature: <feature>". If you do not follow the response format, someone will die.'
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