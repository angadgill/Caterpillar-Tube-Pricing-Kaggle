import pandas as pd
import rmsle

train = pd.read_csv('data/train_set.csv', parse_dates=[2])
test = pd.read_csv('data/test_set.csv', parse_dates=[3])
tube = pd.read_csv('data/tube.csv')
train2 = pd.merge(train, tube, on='tube_assembly_id')
test2 = pd.merge(test, tube, on='tube_assembly_id')

#specs = pd.read_csv('data/specs.csv')

# TODO: Create dimension: number of components
# TODO: Create dimension for each component
# Adding a dimension for each component
# bill_of_materials = pd.read_csv('data/bill_of_materials.csv')
# b = pd.wide_to_long(bill_of_materials, ['component_id_', 'quantity_'], i='tube_assembly_id', j='count')
# b = b.reset_index()
# b = b.drop('count', axis=1)
# b = b.dropna()
# b = b.pivot_table(index='tube_assembly_id', columns='component_id_', values='quantity_')
# b = b.reset_index()
# b = b.fillna(0)
# train2 = pd.merge(train2, b, on='tube_assembly_id')
# test2 = pd.merge(test2, b, on='tube_assembly_id')

# TODO: Create dimensions from date

# TODO: Create dimension using component specs if needed
# components = pd.read_csv('data/components.csv')
# from os import listdir, path
# comp_files = [f for f in listdir('data') if 'comp_' in f]
# for f in comp_files:
#     c = pd.read_csv(path.join('data',f))
#     print c.columns
#     components = pd.merge(components, c, how='left')

X = train2[[c for c in train2.columns if c!='cost']]
X = X.drop(['tube_assembly_id', 'supplier', 'material_id', 'end_a', 'end_x', 'quote_date'], axis=1)
X = X.replace('Yes', 1)
X = X.replace('No', 0)
X = X.replace('Y', 1)
X = X.replace('N', 0)

X_test = test2
X_test = X_test.drop(['tube_assembly_id', 'supplier', 'material_id', 'end_a', 'end_x', 'quote_date' ,'id'], axis=1)
X_test = X_test.replace('Yes', 1)
X_test = X_test.replace('No', 0)
X_test = X_test.replace('Y', 1)
X_test = X_test.replace('N', 0)


y = train2['cost']
(m, _) = X.shape

from sklearn.linear_model import LinearRegression

model = LinearRegression(normalize=True)
model.fit(X[:m/2], y[:m/2])
score = model.score(X[m/2:], y[m/2:])
print "Score:", score

prediction = model.predict(X[m/2:])
actual = y[m/2:]

#zero negative predictions
for i,p in enumerate(prediction):
    if p < 0:
        prediction[i] = 0

error = rmsle.error(prediction, actual)
print "Error:", error


# Train model on full data and create submission file
model = LinearRegression(normalize=True)
model.fit(X, y)
output = model.predict(X_test)
for i,p in enumerate(output):
    if p < 0:
        output[i] = 0

output = pd.DataFrame(output, index=range(1, len(output)+1))
output.to_csv('output.csv', index=True, header=['cost'], index_label='id')