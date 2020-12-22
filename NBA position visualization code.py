import requests
import json
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

#Use api to get json file
BaseURL="https://api.sportsdata.io/v3/nba/stats/json/PlayerSeasonStats/2019?key=5b39b3ec1f67456bbd90458cb1de00b1"
 
response1=requests.get(BaseURL)
print(response1)
jsontxt = response1.json()
print(jsontxt)

#save json file read in 
with open('nba.json', 'w') as outfile:
    json.dump(jsontxt, outfile)
    
#read json file by panda and save to csv
df = pd.read_json ('nba.json')
df.to_csv ('nba.csv')
df
#Cleaning
list(df.columns) 
#1. drop column
df1=df.drop(["Started","Name", "Team", "StatID", "TeamID", "PlayerID", "SeasonType", "Season", "GlobalTeamID", "Updated", "FantasyPoints", "FantasyPointsFanDuel", "FantasyPointsDraftKings", "FantasyPointsYahoo", "FantasyPointsFantasyDraft", "IsClosed", "LineupConfirmed", "LineupStatus","PlayerEfficiencyRating"], axis=1)
list(df1.columns) 
df1

df2=df1.drop(['PlusMinus','FieldGoalsPercentage',"OffensiveRebounds","DefensiveRebounds", 'TwoPointersPercentage', 'ThreePointersPercentage', 'FreeThrowsPercentage', 'EffectiveFieldGoalsPercentage', 'OffensiveReboundsPercentage', 'DefensiveReboundsPercentage', 'TotalReboundsPercentage', 'TrueShootingPercentage', 'AssistsPercentage', 'StealsPercentage', 'BlocksPercentage', 'TurnOversPercentage'	, 'UsageRatePercentage'], axis=1)
list(df2.columns) 
df2

#2. drop less game plays
sns.distplot(df2['Games'], bins=20, kde=False).set_title("Games histogram")
df2.drop(df2[df2['Games'] < 10].index, inplace = True) 
sns.distplot(df2['Games'], bins=20, kde=False).set_title("Games histogram after cleaning")

#3. minute and second convert
fig=sns.kdeplot(df2['Minutes'], shade=True, color="r").set_title("Minutes density")
plt.xlabel("Minutes")
plt.ylabel("Density")
plt.show(fig)
fig=sns.kdeplot(df2['Seconds'], shade=True, color="b").set_title("Seconds density")
plt.xlabel("Seconds")
plt.ylabel("Density")
plt.show(fig)

df3=df2
df3['TotalSeconds']=(df3['Minutes']*60+df3['Seconds'])

fig=sns.kdeplot(df3['TotalSeconds'], shade=True, color="g").set_title("TotalSeconds density")
plt.xlabel("Total Seconds")
plt.ylabel("Density")
plt.show(fig)

df3=df3.drop(['Seconds', 'Minutes'], axis=1)


#4. PerSecond convert
fig = sns.scatterplot(x="Rebounds", y="Assists", hue="Position",
                     data=df3,s=12).set_title("Rebounds vs Assists")
plt.xlabel("Rebounds")
plt.ylabel("Assists")
plt.show(fig)
df3
df4=df3
list(df4.columns) 

##################################################
df4['ReboundsPS']=df4['Rebounds']/(df4['TotalSeconds'])
df4['AssistsPS']=df4['Assists']/(df4['TotalSeconds'])
df4['BlockedShotsPS']=df4['BlockedShots']/(df4['TotalSeconds'])
df4['FieldGoalsAttemptedPS']=df4['FieldGoalsAttempted']/(df4['TotalSeconds'])
df4['FieldGoalsMadePS']=df4['FieldGoalsMade']/(df4['TotalSeconds'])
df4['FreeThrowsAttemptedPS']=df4['FreeThrowsAttempted']/(df4['TotalSeconds'])
df4['FreeThrowsMadePS']=df4['FreeThrowsMade']/(df4['TotalSeconds'])
df4['PersonalFoulsPS']=df4['PersonalFouls']/(df4['TotalSeconds'])
df4['PointsPS']=df4['Points']/(df4['TotalSeconds'])
df4['StealsPS']=df4['Points']/(df4['TotalSeconds'])
df4['ThreePointersAttemptedPS']=df4['ThreePointersAttempted']/(df4['TotalSeconds'])
df4['ThreePointersMadePS']=df4['ThreePointersMade']/(df4['TotalSeconds'])
df4['TrueShootingAttemptsPS']=df4['TrueShootingAttempts']/(df4['TotalSeconds'])
df4['TurnoversPS']=df4['Turnovers']/(df4['TotalSeconds'])
df4['TwoPointersAttemptedPS']=df4['TwoPointersAttempted']/(df4['TotalSeconds'])
df4['TwoPointersMadePS']=df4['TwoPointersMade']/(df4['TotalSeconds'])

df4=df4.drop(['Assists',
 'BlockedShots',
 'FieldGoalsAttempted',
 'FieldGoalsMade',
 'FreeThrowsAttempted',
 'FreeThrowsMade',
 'PersonalFouls',
 'Points',
 'Rebounds',
 'Steals',
 'ThreePointersAttempted',
 'ThreePointersMade',
 'TrueShootingAttempts',
 'Turnovers',
 'TwoPointersAttempted',
 'TwoPointersMade'], axis=1)

##################################################

fig = sns.scatterplot(x="ReboundsPS", y="AssistsPS", hue="Position",
                     data=df4,s=12).set_title("Rebounds per Second vs Assists per Second")
plt.xlabel("Rebounds per Second")
plt.ylabel("Assists per Second")
plt.show(fig)
df4=df4.reset_index()
DF = df4 #final DF
list(DF.columns) 



#EDA

#pie chart
import matplotlib.pyplot as plt
DF.Position.value_counts()
# Data to plot
labels = ["SG","PG","SF","PF","C"]
sizes = [104,109,93,99,64]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','purple']
explode = (0, 0.1, 0, 0,0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.title("Pie chart for Position")
plt.show()


#violin plot

fig = sns.violinplot( y=DF["Position"], x=DF["PointsPS"] ).set_title("Violin plots for points")
plt.xlabel("Points per Second")
plt.ylabel("Position")
plt.show(fig)

#correlogram
DF1 = DF[["FieldGoalsMadePS","FreeThrowsMadePS","ThreePointersMadePS","TwoPointersMadePS","Position"]]
DF1
sns.pairplot(DF1, kind="scatter", hue="Position")

#box
fig = sns.boxplot(x="Position",y="StealsPS",data=DF).set_title("Steals boxplot")
plt.ylabel("Steals per Second")
plt.show(fig)

#Marginal Plots
sns.set(style="white", color_codes=True)
fig =sns.jointplot(x=DF["PointsPS"], y=DF["Games"], kind='kde', color="skyblue")

#boxplot 
fig = sns.boxenplot(x="Position",y="DoubleDoubles",data=DF).set_title("Double Doubles")
plt.ylabel("Count of Double Doubles ")
plt.show(fig)

#swarm
fig = sns.swarmplot(x="Position",y="TripleDoubles",palette="viridis",data=DF).set_title("Triple Doubles")
plt.ylabel("Count of Triple Doubles ")
plt.show(fig)

#Machine Learning
#Decision Tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

#split dataset in features and target variable
feature_cols = ['DoubleDoubles',
 'TripleDoubles',
 'ReboundsPS',
 'AssistsPS',
 'BlockedShotsPS',
 'FieldGoalsAttemptedPS',
 'FieldGoalsMadePS',
 'FreeThrowsAttemptedPS',
 'FreeThrowsMadePS',
 'PersonalFoulsPS',
 'PointsPS',
 'StealsPS',
 'ThreePointersAttemptedPS',
 'ThreePointersMadePS',
 'TrueShootingAttemptsPS',
 'TurnoversPS',
 'TwoPointersAttemptedPS',
 'TwoPointersMadePS']
X = DF[feature_cols] # Features
y = DF.Position # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=74) # 70% training and 30% test

train=y_train.to_frame()
test=y_test.to_frame()
sns.set(style="whitegrid")
ax = sns.countplot(x="Position",data=train,order=["SG","PG","SF","PF","C"]).set_title("Train Set Position Distribution")
ax = sns.countplot(x="Position",data=test,order=["SG","PG","SF","PF","C"]).set_title("Test Set Position Distribution")


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
class_names = clf.classes_
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names=feature_cols,class_names=class_names )
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

 disp = plot_confusion_matrix(clf, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=None)
 disp.ax_.set_title("Confusion Matrix of Decision Tree Prediction")

print(disp.confusion_matrix)
### Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
clfnb=gnb.fit(X_train, y_train)
y_prednb = clfnb.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_prednb))
 disp = plot_confusion_matrix(clfnb, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=None)
 disp.ax_.set_title("Confusion Matrix of Naive Bayes Prediction")


