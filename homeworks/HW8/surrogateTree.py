# Surrogate Split Tree implementation

def SurrogateTreeFit(data_train, response, max_depth):
    
    x_train = data_train.drop(response, axis=1)
    y_train = data_train[response]
    
    def gini_coeff(left, right):
        gini_1 = 1 - np.sum([(i/left.shape[0])**2 for i in left['Class'].value_counts()])
        gini_2 = 1 - np.sum([(i/right.shape[0])**2 for i in right['Class'].value_counts()])
        n = right.shape[0] + left.shape[0]
        return gini_1 * left.shape[0]/n + gini_2 * right.shape[0]/n

    all_columns = list(range(x_train.shape[1]))
    col_names = x_train.columns.values

    def find_best_split(data):
        best_gini = 2
        best_col = None
        best_t = None
        for col in all_columns:
            complete = data.dropna(subset=[col_names[col]])
            for t in range(complete.shape[0]):
                left = complete[complete.iloc[:,col] <= complete.iloc[t,col]]
                right = complete[complete.iloc[:,col] > complete.iloc[t,col]]
                gini = gini_coeff(left, right)
                if gini < best_gini:
                    best_gini, best_col, best_t = gini, col, complete.iloc[t,col]
        complete = data.dropna(subset=[col_names[best_col]])
        return best_col, best_t, complete
    
    tree = {}
    stack = [{'data':data_train, 'depth':1}]
    n_nodes = int(np.sum([2**i for i in range(max_depth)]))
    groups = [0 for i in range(n_nodes)]
    while stack:
        data = stack.pop()
        if data['depth'] == max_depth:
            tree[len(tree)] = {'class': data['data']['Class'].value_counts().idxmax()}
        else:
            n = len(tree)
            best_col, best_t, complete = find_best_split(data['data'])
            if len(complete['Class'].unique()) == 1:
                for i in range(n, n+1+np.sum([2**i for i in range(1, max_depth-data['depth']+1)])):
                    tree[i] = {'class': complete['Class'].value_counts().idxmax()}
            else:
                left = complete[complete.iloc[:,best_col] <= best_t]
                right = complete[complete.iloc[:,best_col] > best_t]
                tree[n] = {'feature': best_col, 'threshold': best_t, 'left_node': n+1, 'right_node': int(n+1+np.sum([2**i for i in range(max_depth-data['depth'])])), 
                           'left': set(left.index), 'right': set(right.index)}
                groups[n] = complete
                stack.extend([{'data':right, 'depth':data['depth']+1}, {'data':left, 'depth':data['depth']+1}])
    
    # Add surrogate splits
    for i in range(n_nodes):
        if 'class' not in tree[i]:
            other_columns = all_columns[:tree[i]['feature']] + all_columns[tree[i]['feature']+1:]
            closeness = []
            for col in other_columns:
                complete = groups[i].dropna(subset=[col_names[col]])
                best_c = 0
                best_t = None
                for t in range(complete.shape[0]):
                    left = set(complete[complete.iloc[:,col] <= complete.iloc[t,col]].index)
                    right = set(complete[complete.iloc[:,col] > complete.iloc[t,col]].index)
                    c = (len(tree[i]['left'].intersection(left)) + len(tree[i]['right'].intersection(right)))/complete.shape[0]
                    if c > best_c:
                        best_c = c
                        best_t = complete.iloc[t,col]
                closeness.append([best_c, col, best_t])
            closeness = sorted(closeness, reverse=True, key = lambda x: x[0])
            tree[i]['surrogate'] = closeness
    return tree
    
def SurrogateTreePredict(x, tree):
    pred_test = []
    for i in range(x.shape[0]):
        cur = 0
        while 'class' not in tree[cur]:
            value = x.iloc[i, tree[cur]['feature']]
            if not np.isnan(value):
                cur = tree[cur]['left_node'] if value <= tree[cur]['threshold'] else tree[cur]['right_node']
                continue
            else:
                for sur in tree[cur]['surrogate']:
                    value = x.iloc[i, sur[1]]
                    if not np.isnan(value):
                        cur = tree[cur]['left_node'] if value <= sur[2] else tree[cur]['right_node']
                        break
        pred_test.append(tree[cur]['class'])
    return pred_test