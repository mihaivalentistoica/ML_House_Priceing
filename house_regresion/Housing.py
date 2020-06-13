from house_regresion.housing_data import load_housing_data
from house_regresion.custom_transformer import DataFrameSelector, CombinedAttributesAdder, CustomLabelBinarizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import numpy as np

housing = load_housing_data()

housing['income-cat'] = np.ceil(housing['median_income'] / 1.5)
housing['income-cat'].where(housing['income-cat'] < 5, 5.0, inplace=True)

stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
strat_train = None
strat_test = None
for train_index, test_index in stratified_split.split(housing, housing['income-cat']):
    strat_train = housing.loc[train_index]
    strat_test = housing.loc[test_index]

for set in (strat_train, strat_test):
    set.drop(['income-cat'], axis=1, inplace=True)

housing_num = strat_train.copy().drop(['ocean_proximity', 'median_house_value'], axis=1)
num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

# print("num attribs: ", num_attribs)
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', CustomLabelBinarizer())
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)
])

# print(strat_train.keys())
# housing_label = list(strat_train) + ['rooms_per_household', 'population_per_household', 'bedrooms_per_room']
# print("Housing label", housing_label)
# strat_train = strat_train.drop('median_house_value', axis=1)

housing_label = strat_train['median_house_value'].copy()
# strat_train.drop('median_house_value', axis=1, inplace=True)
housing_prep = full_pipeline.fit_transform(strat_train)

# print(strat_train.keys())
# lnr_reg = LinearRegression()
# lnr_reg.fit(housing_prep, housing_label)

test_label = strat_test["median_house_value"].copy()
# strat_test.drop('median_house_value', axis=1, inplace=True)
test_data_prep = full_pipeline.fit_transform(strat_test)

# prediction = lnr_reg.predict(test_data_prep)
# lin_mse = mean_squared_error(test_label, prediction)
# lin_mse = np.sqrt(lin_mse)
# print(lin_mse)
# print("Prediction: ", lnr_reg.predict(test_data_prep))
# print("Labels: ", test_label)
# index = 0
# for l in test_label:
#     print(prediction[index])
#     print(l)
#     print("..............")
#     index += 1


# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(housing_prep, housing_label)
# tree_prediction = tree_reg.predict(test_data_prep)
# tree_mse = mean_squared_error(test_label, tree_prediction)
# tree_mse = np.sqrt(tree_mse)
# print(tree_mse)


# Support Vector Machine regressor
svp_reg = SVR()
svp_reg.fit(housing_prep, housing_label)
svp_prediction = svp_reg.predict(test_data_prep)
svp_mse = mean_squared_error(test_label, svp_prediction)
svp_mse = np.sqrt(svp_mse)
print(svp_mse)

