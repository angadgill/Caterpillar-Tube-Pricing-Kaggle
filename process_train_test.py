import pandas as pd
import rmsle
from preprocess import scale

from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV
from sklearn.decomposition import PCA


print "Importing data..."
train = pd.read_csv('data/train_set.csv', parse_dates=[2])
test = pd.read_csv('data/test_set.csv', parse_dates=[3])
tube = pd.read_csv('data/tube.csv')
train2 = pd.merge(train, tube, on='tube_assembly_id')
test2 = pd.merge(test, tube, on='tube_assembly_id')

# Eliminate bracket_pricing and min_order_quanity dimensions
# by consolidating their info in quantity
q = train2[train2.bracket_pricing=='No'].min_order_quantity
train2.loc[train2.bracket_pricing=='No', 'quantity'] = q
train2 = train2.drop(['min_order_quantity', 'bracket_pricing'], axis=1)

q = test2[test2.bracket_pricing=='No'].min_order_quantity
test2.loc[test2.bracket_pricing=='No', 'quantity'] = q
test2 = test2.drop(['min_order_quantity', 'bracket_pricing'], axis=1)

#specs = pd.read_csv('data/specs.csv')

print "Scale dimensions..."
scale_dimensions  = ['annual_usage', 'quantity', 'diameter', 'bend_radius', 'wall', 'length', 'num_bends', 'num_boss', 'num_bracket']
train2[scale_dimensions] = scale(train2[scale_dimensions])
test2[scale_dimensions] = scale(test2[scale_dimensions])

print "Creating dimensions..."

# Process data to create dimensions related to components
bill_of_materials = pd.read_csv('data/bill_of_materials.csv')
b = pd.wide_to_long(bill_of_materials, ['component_id_', 'quantity_'], i='tube_assembly_id', j='count')
b = b.reset_index()
b = b.drop('count', axis=1)

# Adding a dimension for number of components
# b1 = b.drop('component_id_', axis=1)
# b1 = b1.groupby('tube_assembly_id').sum()
# b1 = b1.reset_index()
# b1.columns = ['tube_assembly_id', 'component_quantity']
# train2 = pd.merge(train2, b1, on='tube_assembly_id', how='left')
# train2 = train2.fillna(0)
# test2 = pd.merge(test2, b1, on='tube_assembly_id', how='left')
# test2 = test2.fillna(0)

# Adding a dimension for each component
b2 = b.pivot_table(index='tube_assembly_id', columns='component_id_', values='quantity_')
b2 = b2.reset_index()
b2 = b2.fillna(0)
train2 = pd.merge(train2, b2, on='tube_assembly_id', how='left')
train2 = train2.fillna(0)
test2 = pd.merge(test2, b2, on='tube_assembly_id', how='left')
test2 = test2.fillna(0)

# TODO: Create dimension using component specs if needed
# components = pd.read_csv('data/components.csv')
# from os import listdir, path
# comp_files = [f for f in listdir('data') if 'comp_' in f]
# for f in comp_files:
#     c = pd.read_csv(path.join('data',f))
#     print c.columns
#     components = pd.merge(components, c, how='left')

# TODO: Create dimension using material_id, end_a, end_x
# TODO: Create dimension using end_a
# TODO: Create dimension using end_a
# TODO: Create dimension using material_id

X = train2[[c for c in train2.columns if c!='cost']]
X = X.drop(['tube_assembly_id', 'supplier', 'material_id', 'end_a', 'end_x', 'quote_date'], axis=1)
X = X.replace('Yes', 1)
X = X.replace('No', 0)
X = X.replace('Y', 1)
X = X.replace('N', 0)
# X = scale(X, axis=0, copy=False)

X_test = test2
X_test = X_test.drop(['tube_assembly_id', 'supplier', 'material_id', 'end_a', 'end_x', 'quote_date' ,'id'], axis=1)
X_test = X_test.replace('Yes', 1)
X_test = X_test.replace('No', 0)
X_test = X_test.replace('Y', 1)
X_test = X_test.replace('N', 0)
# X_test = scale(X_test, axis=0, copy=False)


y = train2['cost']
(m, _) = X.shape
split = int(m*0.8)

#PCA
# pca = PCA(n_components=10)
# pca.fit(X,y)
# X = pca.transform(X)

# Linear Regression
# model = LinearRegression(normalize=False)
alphas = [0.001, 0.01, 0.1, 0.3, 1, 3, 10]
# for alpha in alphas:
#     print "Alpha:", alpha
# model = ElasticNetCV(alphas=alphas)
model = LassoCV(alphas=alphas)

print "Training model..."
model.fit(X[:split], y[:split])
alpha = model.alpha_
print "Alpha:", alpha

score = model.score(X[split:], y[split:])
print "Score:", score

prediction = model.predict(X[split:])
actual = y[split:]

#zero negative predictions
for i,p in enumerate(prediction):
    if p < 0:
        prediction[i] = 0

error = rmsle.error(prediction, actual)
print "Error:", error

raw_input("Press Enter to continue...")

# Train model on full data and create submission file
# pca = PCA(n_components=5)
# pca.fit(X_test,y)
# X_test = pca.transform(X_test)
#
# model = ElasticNet(alpha=alpha)
model = Lasso(alpha=alpha)
print "Training model..."
model.fit(X, y)
print "Running model on test data..."
output = model.predict(X_test)
for i,p in enumerate(output):
    if p < 0:
        output[i] = 0

output = pd.DataFrame(output, index=range(1, len(output)+1))
output.to_csv('output.csv', index=True, header=['cost'], index_label='id')


print "Done!"