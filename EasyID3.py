# coding: utf-8
import pandas as pd
from collections import Counter
from scipy import stats
class EasyID3:
    class TargetNotFoundError(Exception):
        pass
    
    class RedundantDefinition(Exception):
        pass
    
    class Node:
        class Comparator:
            def __init__(self, attr, compare, value):
                self.attr = attr
                self.cmp = compare
                self.value = value

            def compare(self, row):
                if self.cmp == '=':
                    return row[self.attr] == self.value


            def __str__(self):
                return '%s %s %s' % (self.attr, self.cmp, self.value)

        def __init__(self, deep, name, entropy, table, target):
            self.deep = deep
            self.name = name
            self.entropy = entropy
            self.count = table[target].value_counts().to_dict()
            self.childs = {}
            
        def add_child(self, name, child_node):
            self.childs[name] = child_node
            
        def __str__(self):
            num_tabs = self.deep * '\t'
            str_acc = ''
            str_acc += num_tabs + 'Name: ' + str(self.name) + '\n'
            str_acc += num_tabs + 'Count: ' + str(self.count) + '\n'
            str_acc += num_tabs + 'Entropy: ' + str(self.entropy) + '\n'
            
            if len(self.childs) > 0:
                str_acc += num_tabs + 'Childs:\n'
            
            for child_name, child_info in self.childs.items():
                str_acc += num_tabs + str(child_name)
                str_acc += str(child_info)
            
            return str_acc
    
    def __init__(self):
        self.root = None
        
    # Metrics
    def __entropy(table, target):
        return stats.entropy(list(Counter(table[target].values).values()), base=2)

    def __gain(entropy, table, field_name, target):
        options = table[field_name].unique()
        entropies_of_field = {}
        
        cond_entropy = 0.0
        for opt_name in options:
            subtable = table[table[field_name] == opt_name]
            
            entropy_field_opt = EasyID3.__entropy(subtable, target)
            entropies_of_field[opt_name] = entropy_field_opt
            
            cond_entropy += (len(subtable) / len(table)) * entropy_field_opt
        return entropy - cond_entropy, entropies_of_field
    
    def __select_best_split(entropy, table, target):
        max_gain = float('-inf')
        max_field = None
        for field_name in table.columns:
            if field_name == target:
                continue
                
            value, entropies_of_field = EasyID3.__gain(entropy, table, field_name, target)
            
            if max_gain <= value:
                max_gain = value
                max_field = (field_name, entropies_of_field)
                
        return max_field
    
    # Fit
    def fit(self, X, y=None, target=None):
        self.__table = X
        
        # Cases
        if target is None:  # No target defined
            if y is None: # No y defined
                raise EasyID3.TargetNotFoundError
            elif isinstance(y, str):
                self.__target = target
            elif isinstance(y, pd.DataFrame): # Use first column as target
                self.__target = y.columns[0]
                self.__table = pd.concat([X, y], axis=1)
            
            elif isinstance(y, pd.Series): # Use name series as target
                self.__target = y.name
                self.__table = pd.concat([X, y], axis=1)
        else: # Target defined
            if y is not None:
                raise EasyID3.RedundantDefinition
            else:
                self.__target = target
        
        entropy = EasyID3.__entropy(self.__table, self.__target)
        self.root = EasyID3.Node(0, 'root', entropy, self.__table, self.__target)


        # Do Step
        def do_step(deep, parent_node, parent_entropy, parent_table):
            deep += 1
            
            field_name, entropies_of_field = EasyID3.__select_best_split(parent_entropy, parent_table, self.__target)
            
            for opt_field_name, opt_entropy in entropies_of_field.items():
                opt_table = parent_table[parent_table[field_name] == opt_field_name]
                opt_table = opt_table.loc[:, opt_table.columns != field_name]
                
                opt_name = EasyID3.Node.Comparator(field_name, '=', opt_field_name) #'%s = %s' % (field_name, opt_field_name)
                
                opt_node = EasyID3.Node(deep, opt_name, opt_entropy, opt_table, self.__target)
                parent_node.add_child(opt_name, opt_node)
                
                if len(opt_table.columns) > 1 and opt_entropy > 0:
                    do_step(deep, opt_node, opt_entropy, opt_table)
                
        do_step(0, self.root, entropy, self.__table)

    # Predict
    def predict(self, X):
        def do_step(parent_node, row):
            if len(parent_node.childs) == 0:
                return max(parent_node.count.items(), key=lambda key: key[1])[0]
            else:
                for k, v in parent_node.childs.items():
                    if k.compare(row):
                        return do_step(v, row)

        y_estim = []
        for i in range(X.shape[0]):
            y_estim.append(do_step(self.root, X.iloc[i]))
        return y_estim
                
    def show(self):
        print(self.root)
        






if __name__ == '__main__':
    # Implementación ID3
    # En esta sección implementaremos nuestro propio ID3, veremos que no es nada dificil.
    # Para ello, usaremos un dataset de prueba.
    # Cargamos el dataset...
    data = [
        ["Outlook", "Temp", "Humidity", "Windy", "Play Golf"],
        ["Rainy",   "Hot",  "High",     "False", "No"],
        ["Rainy",   "Hot",  "High",     "True",  "No"],
        ["Overcast","Hot",  "High",     "False", "Yes"],
        ["Sunny",   "Mild", "High",     "False", "Yes"],
        ["Sunny",   "Cool", "Normal",   "False", "Yes"],
        ["Sunny",   "Cool", "Normal",   "True",  "No"],
        ["Overcast","Cool", "Normal",   "True",  "Yes"],
        ["Rainy",   "Mild", "High",     "False", "No"],
        ["Rainy",   "Cold", "Normal",   "False", "Yes"],
        ["Sunny",   "Mild", "Normal",   "False", "Yes"],
        ["Rainy",   "Mild", "Normal",   "True",  "Yes"],
        ["Overcast","Mild", "High",     "True",  "Yes"],
        ["Overcast","Hot",  "Normal",   "False", "Yes"],
        ["Sunny",   "Mild", "High",     "True",  "No"],
    ]


    # Lo convertimos a pandas DataFrame para ver que datos contiene...
    df = pd.DataFrame(data[1:], columns=data[0])
    print(df)

    # Learn process
    id3 = EasyID3()
    id3.fit(df, target='Play Golf')
    id3.show()

    # Predict
    print(id3.predict(df.loc[:, df.columns != 'Play Golf']))

