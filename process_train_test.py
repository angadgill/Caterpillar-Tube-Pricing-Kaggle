import pandas as pd
import rmsle
import preprocess
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import ShuffleSplit

print "Importing data..."
train = pd.read_csv('data/train_set.csv', parse_dates=[2])
test = pd.read_csv('data/test_set.csv', parse_dates=[3])
tube = pd.read_csv('data/tube.csv')

print "Creating dimensions..."
# Process material_id
# borrowed and modified from 'Keras starter code' by fchollet
tube.material_id.fillna('SP-9999', inplace=True)

#Process end_a
tube.end_a = tube.end_a.replace('9999', 'EF-9999')

#Process end_x
tube.end_x = tube.end_x.replace('9999', 'EF-9999')


# Convert the following dimensions to multiple dimensions:
# 'material_id', 'end_a', 'end_x'
columns = ['material_id', 'end_a', 'end_x']
for c in columns:
    tube = preprocess.long_to_wide(tube, 'tube_assembly_id', c)

train2 = pd.merge(train, tube, on='tube_assembly_id', how='left')
test2 = pd.merge(test, tube, on='tube_assembly_id', how='left')

# Convert 'supplier' into multiple dimensions
# train2 = preprocess.long_to_wide(train2, 'tube_assembly_id', 'supplier')
# test2 = preprocess.long_to_wide(test2, 'tube_assembly_id', 'supplier')
#

# Eliminate bracket_pricing and min_order_quanity dimensions
# by consolidating their info in quantity
q = train2[train2.bracket_pricing=='No'].min_order_quantity
train2.loc[train2.bracket_pricing=='No', 'quantity'] = q
train2 = train2.drop(['min_order_quantity', 'bracket_pricing'], axis=1)

q = test2[test2.bracket_pricing=='No'].min_order_quantity
test2.loc[test2.bracket_pricing=='No', 'quantity'] = q
test2 = test2.drop(['min_order_quantity', 'bracket_pricing'], axis=1)

# Encode the labels of the categorical variables
# borrowed and modified from 'Keras starter code' by fchollet
# columns = ['end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x']
columns = ['end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x']
for c in columns:
    lbl = LabelEncoder()
    lbl.fit(list(train2[c]) + list(test2[c]))
    train2[c] = lbl.transform(train2[c])
    test2[c] = lbl.transform(test2[c])

# Process data to create dimensions related to components
# bill_of_materials = pd.read_csv('data/bill_of_materials.csv')
# b = pd.wide_to_long(bill_of_materials, ['component_id_', 'quantity_'], i='tube_assembly_id', j='count')
# b = b.reset_index()
# b = b.drop('count', axis=1)

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
# b2 = b.pivot_table(index='tube_assembly_id', columns='component_id_', values='quantity_')
# b2 = b2.reset_index()
# b2 = b2.fillna(0)
# train2 = pd.merge(train2, b2, on='tube_assembly_id', how='left')
# train2 = train2.fillna(0)
# test2 = pd.merge(test2, b2, on='tube_assembly_id', how='left')
# test2 = test2.fillna(0)

# TODO: Create dimension using components and specs if needed
# specs = pd.read_csv('data/specs.csv')
# components = pd.read_csv('data/components.csv')
# from os import listdir, path
# comp_files = [f for f in listdir('data') if 'comp_' in f]
# for f in comp_files:
#     c = pd.read_csv(path.join('data',f))
#     print c.columns
#     components = pd.merge(components, c, how='left')

# train2['quantity'] = np.log(train2.quantity)
# test2['quantity'] = np.log(test2.quantity)

print "Scale dimensions..."
scale_dimensions = ['annual_usage', 'quantity', 'diameter', 'bend_radius', 'wall', 'length', 'num_bends', 'num_boss', 'num_bracket']
train2[scale_dimensions] = preprocess.scale(train2[scale_dimensions])
test2[scale_dimensions] = preprocess.scale(test2[scale_dimensions])


X = train2
# X = X.drop(['tube_assembly_id', 'quote_date', 'cost', 'supplier', 'material_id', 'end_a', 'end_x'], axis=1)
X = X.drop(['tube_assembly_id', 'quote_date', 'cost', 'supplier'], axis=1)

X_test = test2
# X_test = X_test.drop(['tube_assembly_id', 'quote_date','id', 'supplier', 'material_id', 'end_a', 'end_x'], axis=1)
X_test = X_test.drop(['tube_assembly_id', 'quote_date','id', 'supplier'], axis=1)


y = train2['cost']

m = X.shape[0]

split = ShuffleSplit(n=int(m*0.9), n_iter=2, test_size=0.2)

for (split_train, split_test) in split:
    model = RandomForestRegressor()
    print "Training model..."
    model.fit(X.loc[split_train,:],y[split_train])
    score_test = model.score(X.loc[split_test,:], y[split_test])
    print "Test Score:", score_test
    error_test = rmsle.error(model.predict(X.loc[split_test,:]), y[split_test])
    print "Test Error:", error_test

raw_input("Press Enter to train model and save output:")

print "Training model with all training data..."
model = RandomForestRegressor()
model.fit(X, y)
output = model.predict(X_test)

#zero-out negative predictions
# for i,p in enumerate(output):
#     if p < 0:
#         output[i] = 0

output = pd.DataFrame(output, index=range(1, len(output)+1))
print "Saving output.csv..."
output.to_csv('output.csv', index=True, header=['cost'], index_label='id')

print "Done!"