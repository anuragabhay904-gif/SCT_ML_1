import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# dataset was loaded 
data = pd.read_csv('train.csv')

# input features for the predicition
features = [
    'GrLivArea',
    'BedroomAbvGr',
    'FullBath',
    'HalfBath',
    'TotRmsAbvGrd',
    'GarageCars',
    'GarageArea',
    'OverallQual',
    'YearBuilt'
]

# feature for the output
target = 'SalePrice'

# Now a DataFrame is created with the input features and the targeted value 
df = data[features + [target]]

# removing all the missing value using the dropna method
df = df.dropna()

# arranging the input and output features 
X = df[features]
y = df[target]

# we will split the data into training and the testing data to train the model
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# now we have split the data , we have the inputs(features) and we know the value we have to predict(SalePrice)
# All the missing values has been removed already 
# lets train the model now to have hypotheses to get the output
from sklearn.linear_model import LinearRegression

model = LinearRegression()
# we have created a model ,lets train the model.
model.fit(X_train,y_train)

# Training is done , lets predict the value now 
predict = model.predict(X_test)

# Now we need some parameters to evaluate the model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

mae = mean_absolute_error(y_test, predict)
mse = mean_squared_error(y_test, predict)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predict)

# converting R2 score into percentage
accuracy = r2 * 100

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
print("Accuracy:", accuracy, "%")


# lets also add the coeffeciets to check how the different features affect the model 
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})

print("\n===== MODEL COEFFICIENTS =====")
print(coefficients)


# Now we have the model lets predict for some new values
new_house = pd.DataFrame({
    'GrLivArea': [2000],
    'BedroomAbvGr': [3],
    'FullBath': [2],
    'HalfBath': [1],
    'TotRmsAbvGrd': [7],
    'GarageCars': [2],
    'GarageArea': [500],
    'OverallQual': [7],
    'YearBuilt': [2005]
})

predicted_price = model.predict(new_house)

print("\n===== NEW HOUSE PREDICTION =====")
print("Predicted House Price:", predicted_price[0])

# residual error graph to check the errors in the prediction.
errors = y_test - predict

plt.scatter(predict, errors)

plt.axhline(y=0, color='red')

plt.xlabel("Predicted")
plt.ylabel("Errors")
plt.title("Residual Plot")

plt.savefig("residual_plot.png")

#actual v/s predicted values.
plt.figure(figsize=(8,6))

plt.scatter(y_test, predict, color='blue')

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")

plt.savefig("prediction_graph.png")

# Load the Kaggle test data
test_data = pd.read_csv("test.csv")

# Use only the features your model needs
X_kaggle_test = test_data[features].fillna(0)  # fill missing values

# Predict using your trained model
kaggle_predictions = model.predict(X_kaggle_test)

# Create submission dataframe
submission = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": kaggle_predictions
})

# Save to CSV
submission.to_csv("my_first_submission.csv", index=False)

print("Submission file created: my_first_submission.csv")