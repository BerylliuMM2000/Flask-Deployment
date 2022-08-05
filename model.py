from pandas import *
from numpy import *
from pickle import *

# Retrieving the iris dataset

iris = read_csv("iris.csv")
iris.fillna(0, inplace=True)

# Converting text entries to numbers
def convert_to_int(word):
    word_dict = {'Setosa':1, 'Versicolor':2, 'Virginica':3}
    return word_dict[word]

iris['variety'] = iris['variety'].apply(lambda x : convert_to_int(x))

data = iris.iloc[:, :-1]
target = iris.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(data, target)

# Saving model to disk
dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = load(open('model.pkl','rb'))

def outputConvert(output):
    if output < 1.5: return "Setosa"
    if output < 2.5: return "Versicolor"
    return "Virginica"

print(outputConvert(model.predict([[6, 3, 6, 2]])))