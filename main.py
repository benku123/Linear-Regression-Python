import pandas as pd
import numpy as np
import matplotlib.pyplot as pit
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#to write from google
'''from google.colab import drive
drive.mount('content')
df = pd.read_csv('/content/content/MyDrive/Colab Notebooks/student_scores.csv')'''

df = pd.read_csv("student_scores.csv")
print(df.head())



y = df['Hours']
x = df['Scores']

pit.scatter(x, y,  color='blue')
pit.title('Scatter plot')
pit.xlabel('Hours')
pit.ylabel('Scores')
pit.xticks(())
pit.yticks(())
pit.show()

#Different representation
sns.regplot(x, y)
pit.title('Linear regression plot', size=8)
pit.ylabel('Hours', size=8)
pit.xlabel('Marks', size=8)
pit.show()


x= df.iloc[:,:-1].values
y= df.iloc[:,1].values
print(x, y)
#Second column 'Hours', first 15 values

train_X, val_X, train_y, val_y = train_test_split(x, y, random_state = 0)

regression = LinearRegression()
regression.fit(train_X, train_y)
print("model was trained")

r_sq = regression.score(x, y)
print(f"coefficient of determination r = {round(r_sq, 3)}\n")


if r_sq > 0:
  print("Positive Scatter Plot")
else:
  print("Negetave Scatter Plot")

#Predict Response
#Regression Equation: Y = a + bx
YPred = regression.predict(x)
print(f"predicted value: {YPred}")

sns.regplot(x, YPred)
pit.title('Linear regression plot',size=8)
pit.ylabel('Hours', size=8)
pit.xlabel('Marks', size=8)
pit.show()
