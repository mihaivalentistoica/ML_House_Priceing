from house_regresion.housing_data import load_housing_data
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer
from house_regresion.custom_transformer import CombinedAttributesAdder
from pandas import DataFrame
import numpy as np

# fetch_housing_data()
housing = load_housing_data()
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# creem o categorie de venit
housing['income-cat'] = np.ceil(housing['median_income'] / 1.5)
housing['income-cat'].where(housing['income-cat'] < 5, 5.0, inplace=True)
# creem noi categorii
# housing['room_per_household'] = housing['total_rooms'] / housing['households']
# housing['bedroom_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
# housing['population_per_household'] = housing['population'] / housing['households']

print(housing.keys())
# print(housing.values[:4])

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_added_attr = attr_adder.transform(housing.values)


# print(housing_added_attr)

# Replace missing values whit median value
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop('ocean_proximity', axis=1)

imputer.fit(housing_num)
X = imputer.transform(housing_num)
# Back into pandas DataFrame from numpy array
housing_tr = DataFrame(X, columns=housing_num.columns)

# Convert ocean_proximity labels tu numbers
# encoder = LabelEncoder()
housing_cat = housing['ocean_proximity']
# housing_cat_encoded = encoder.fit_transform(housing_cat)
# print(encoder.classes_)
# print(housing_cat_encoded)

# hot_encoder = OneHotEncoder()
# housing_cat_1hot = hot_encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))

encoder = LabelBinarizer(sparse_output=True)
housing_cat_1hot = encoder.fit_transform(housing_cat)
print(housing_cat_1hot.toarray())

# print(imputer.statistics_[0])
# print(housing_num.median().values)

# print(housing['income-cat'].value_counts())
# print(housing['ocean_proximity'])

# Creem o esalonare stratificata in functie de categoria de pret
stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
strat_train = None
strat_test = None

for train_index, test_index in stratified_split.split(housing, housing['income-cat']):
    strat_train = housing.loc[train_index]
    strat_test = housing.loc[test_index]

for set in (strat_train, strat_test):
    set.drop(['income-cat'], axis=1, inplace=True)

# play_data = strat_train.copy()
# play_data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)

# corr_matrix = housing.corr()
# print(corr_matrix['median_house_value'].sort_values(ascending=False))

# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"] / 100, label="population",
#              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
# plt.legend()
# plt.show()

# print(strat_train.describe().to_string())
# print(f'train: {len(train_set)}, test: {len(test_set)}')
# print(housing.info())
# housing.hist(bins=50, figsize=(25,15))
# plt.show()


# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12,8))
# plt.show()
