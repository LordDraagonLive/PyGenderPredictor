# Importing a decision tree from scikit-learn package
from sklearn import tree

# py list of lists with [height, weight, shoe size] data of each person (includes 11 people)
# These data can be used for our gender prediction
pplList = [[182, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40],
    [159, 55, 37], [171, 75, 42], [181, 85, 43]]

# Genders of the group of people
pplGenderList = ['male', 'female','female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male',]

# Storing decision tree classifer in 'clf' var
clf = tree.DecisionTreeClassifier()

# Now we are training the classifier using the fit() method
# by using the values of 'pplList' and 'pplGenderList' Lists
# fit() method builds a decision tree for prediction
clf = clf.fit(pplList,pplGenderList)

# We are using the trained clasifier var to predict the gender
# of the given person (person's height, weight, and shoe size)
prediction = clf.predict([[190, 70, 43]])

print(prediction)