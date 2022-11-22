weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Overcast','Sanny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool''Mild','Cold','Mild','Cold','Mild','Mild','Mild','Hot','Mild']
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
weather_encoded=le.fit_transform(weather)
print("Weather : ",weather_encoded)
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)
print("Temp : ",temp_encoded)
print("Play : ",label)
features=zip(weather_encoded,temp_encoded)
print(features)
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(features.label)
predicted=model.predict([[0,2]])
print("Predicted Value : ",predicted)