#data wrangling with mongodb
from pprint import PrettyPrinter
import pandas as pd
!pip install pymongo 
from pymongo import MongoClient

#Instantiating a PrettyPrinter
pp = PrettyPrinter(indent=2)

#Creating a client that connects to the database running at `localhost` on port `27017`
client = MongoClient(host="localhost", port=27017 )

#Printing a list of the databases available on `client`
from sys import getsizeof
pp.pprint(list(client.list_databases()))
db = client["air-quality"]
db
#printing a list of the collections available in db
for c in db.list_collections():
    print(c["name"])

#there's lagos, dar-es-salaam and nairobi. the interest is in nairobi
nairobi = db["nairobi"]

#documents in the nairobi collection
nairobi.count_documents({})

#retrieving a document from the nairobi collection
result = nairobi.find_one({})
pp.pprint(result)
#each document has info like latitude, longitude, sensor information, site, temeperature and time stamp..

#sensor sites
nairobi.distinct("metadata.site")
#there are 2 sensor sites 29, 6

#readings in each sites
result = nairobi.aggregate(
    [
        {"$group": {"_id": "$metadata.site", "count":{"$count":{}}}}
    ]

)
pp.pprint(list(result))
#[{'_id': 29, 'count': 131852}, {'_id': 6, 'count': 70360}]

#MEASUREMENTS
nairobi.distinct("metadata.measurement")
#['humidity', 'P1', 'temperature', 'P2']

#READINGS FOR EACH MEASUREMENT IN SITE 6
result = nairobi.aggregate(
    [
        {"$match": {"metadata.site" : 6}},
        {"$group": {"_id": "$metadata.measurement", "count":{"$count":{}}}}
    ]

)
#[{'_id': 'P1', 'count': 18169},
#  {'_id': 'humidity', 'count': 17011},
#{'_id': 'P2', 'count': 18169},
#{'_id': 'temperature', 'count': 17011}]
pp.pprint(list(result))

#READINGS FOR EACH MEASUREMENT IN SITE 29
result = nairobi.aggregate(
    [
        {"$match": {"metadata.site" : 29}},
        {"$group": {"_id": "$metadata.measurement", "count":{"$count":{}}}}
    ]

)
pp.pprint(list(result))
#[ {'_id': 'P1', 'count': 32907},
# {'_id': 'humidity', 'count': 33019},
# {'_id': 'P2', 'count': 32907},
#{'_id': 'temperature', 'count': 33019}]

#site 29 has more readings. so it's going to be the data to be used

#retrieving PM2.5(P2) readings and timestamps
result = nairobi.find(
    {"metadata.site":29, "metadata.measurement": "P2"},
    projection={"P2": 1, "timestamp":1, "_id":0}
)

df = pd.DataFrame(result).set_index("timestamp")


#LINEAR REGRESSION WITH TIME SERIES DATA 
#wrangle function
def wrangle(collection):
    results = collection.find(
        {"metadata.site": 29, "metadata.measurement": "P2"},
        projection={"P2": 1, "timestamp": 1, "_id": 0},
    )

    df = pd.DataFrame(results).set_index("timestamp")
    
    #localize timezone
    df.index = df.index.tz_localize("UTC").tz_convert("Africa/Nairobi")
    
    #removing outliers 
    df=df[df["P2"] < 500]
    
    #resampling to 1H window, forward fill missing values
    df=df["P2"].resample("1H").mean().fillna(method="ffill").to_frame()
    
    return df
  
df = wrangle(nairobi)
df.head(10)

#boxplot of the "P2" readings
fig, ax = plt.subplots(figsize=(15, 6))
df["P2"].plot(kind="box", vert=False, title="Distribution of PM2.5 Readings", ax=ax)
#presence of outliers

#time series plot for the p2 readings
fig, ax = plt.subplots(figsize=(15, 6))
df["P2"].plot(xlabel="Time", ylabel="PM2.5", title="PM2.5 Time Series", ax=ax);

#Plotting the rolling average of the "P2" readings
df["P2"].rolling(168).mean().plot(ax=ax, ylabel="PM2.5", title = "Weekly Rolling Average")

#correlation matrix
df.corr()

#autocorrelation plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x=df["P2.L1"], y=df["P2"])
ax.plot([0, 120], [0,120], linestyle= "--", color="orange")
plt.xlabel("P2.L1")
plt.ylabel("P2")
plt.title("PM2.5 Autocorrelation");

#Splitting the DataFrame df into the feature matrix X and the target vector y
target = "P2"
y = df[target]
X = df.drop(columns=target)

#Splitting X and y into training and test sets
cutoff = int(len(X) * 0.8)

X_train, y_train = X.iloc[:cutoff], y.iloc[:cutoff]
X_test, y_test = X.iloc[cutoff:], y.iloc[cutoff:]

#Calculating the baseline mean absolute error for the model
y_pred_baseline = [y_train.mean()] * len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

print("Mean P2 Reading:", round(y_train.mean(), 2))
print("Baseline MAE:", round(mae_baseline, 2))

#fitting a model to the training data
model = LinearRegression()
model.fit(X_train, y_train)

#calculating the maes for the training and the testing data
training_mae = mean_absolute_error(y_train, model.predict(X_train))
test_mae = mean_absolute_error(y_test, model.predict(X_test))
print("Training MAE:", round(training_mae, 2))
print("Test MAE:", round(test_mae, 2))

#intercept and coefficient
intercept = model.intercept_.round(2)
coefficient = model.coef_.round(2)

print(f"P2 = {intercept} + ({coefficient} * P2.L1)")

#creating a dataframe for the test set values andd model predictions
df_pred_test = pd.DataFrame(
    {
        "y_test": y_test,
        "y_pred": model.predict(X_test)
        
    }


)
df_pred_test.head()

#creating a time series plot for the test predictions
fig = px.line(df_pred_test, labels={"value":"P2"})
fig.show()


#CREATING AN AR MODEL TO PREDICT PM2.5 READINGS
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg

#wrangle function
def wrangle(collection):
    results = collection.find(
        {"metadata.site": 29, "metadata.measurement": "P2"},
        projection={"P2": 1, "timestamp": 1, "_id": 0},
    )

    # Read data into DataFrame
    df = pd.DataFrame(list(results)).set_index("timestamp")

    # Localize timezone
    df.index = df.index.tz_localize("UTC").tz_convert("Africa/Nairobi")

    # Remove outliers
    df = df[df["P2"] < 500]

    # Resample to 1hr window
    y = df["P2"].resample("1H").mean().fillna(method='ffill')

    return y
  
  #reading the data from the nairobi collection into the Series y
  y = wrangle(nairobi)
  y.head()
  
  #acf plot for y 
  fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(y, ax=ax)
plt.xlabel("Lag [hours]")
plt.ylabel("Correlation Coefficient");

#pacf plot
fig, ax = plt.subplots(figsize=(15, 6))
plot_pacf(y, ax=ax)
plt.xlabel("Lag [hours]")
plt.ylabel("Correlation Coefficient");

#splitting the data
cutoff_test = int(len(y) * 0.95)

y_train = y.iloc[:cutoff_test]
y_test = y.iloc[cutoff_test:]

#baseline mean absolute error for the model.
y_train_mean = y_train.mean()
y_pred_baseline = [y_train_mean] * len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

print("Mean P2 Reading:", round(y_train_mean, 2))
print("Baseline MAE:", round(mae_baseline, 2))

#fitting the data
model = AutoReg(y_train, lags=26 ).fit()

#Generating a list of training predictions for the model and using them to calculate the training mean absolute error
y_pred = model.predict().dropna()
training_mae = mean_absolute_error(y_train.iloc[26:], y_pred)
print("Training MAE:", training_mae)

#calculating residuals
y_train_resid = model.resid
y_train_resid.tail()

#time series plot for the residuals
fig, ax = plt.subplots(figsize=(15, 6))
y_train_resid.plot(ylabel="Residual Value",  ax=ax);

#histogram for residuals
y_train_resid.hist()
plt.xlabel("Residual Value")
plt.ylabel("Frequency")
plt.title("AR(26), Distribution of Residuals");

#residuals acf plot
fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(y_train_resid, ax=ax)

#Calculating the test mean absolute error for the model
y_pred_test = model.predict(y_test.index.min(),y_test.index.max())
test_mae = mean_absolute_error(y_test, y_pred_test)
print("Test MAE:", test_mae)

#Create a DataFrame test_predictions that has two columns: "y_test" and "y_pred"
df_pred_test = pd.DataFrame(
    {"y_test": y_test, "y_pred": y_pred_test}, index=y_test.index
)

# Performing walk-forward validation for the model for the entire test set `y_test`
%%capture

y_pred_wfv = pd.Series()
history = y_train.copy()
for i in range(len(y_test)):
    model = AutoReg(history, lags=26).fit()
    next_pred = model.forecast()
    y_pred_wfv = y_pred_wfv.append(next_pred)
    history=history.append(y_test[next_pred.index])
    pass

#test evaluation
test_mae = mean_absolute_error(y_test, y_pred_wfv)

#model coefficients
print(model.params)

#plot wfv predictions
df_pred_test = pd.DataFrame(
    {"y_test": y_test, "y_pred_wfv": y_pred_wfv}
)
fig = px.line(df_pred_test, labels={"value": "PM2.5"})
fig.show()


#CREATING AN ARMA MODEL TO PREDICT PM2.5 READINGS
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

#wrangle function
def wrangle(collection, resample_rule="1H"):

    results = collection.find(
        {"metadata.site": 29, "metadata.measurement": "P2"},
        projection={"P2": 1, "timestamp": 1, "_id": 0},
    )

    # Read results into DataFrame
    df = pd.DataFrame(list(results)).set_index("timestamp")

    # Localize timezone
    df.index = df.index.tz_localize("UTC").tz_convert("Africa/Nairobi")

    # Remove outliers
    df = df[df["P2"] < 500]

    # Resample and forward-fill
    y = df["P2"].resample(resample_rule).mean().fillna(method="ffill")
    return y
  
  #baseline mae
  y_train_mean= y_train.mean()
y_pred_baseline= [y_train_mean] * len(y_train)
mae_baseline= mean_absolute_error(y_train, y_pred_baseline )
print("Mean P2 Reading:", round(y_train_mean, 2))
print("Baseline MAE:", round(mae_baseline, 2))

#hyperparameter ranges
p_params = range(0,25,8)
q_params = range(0,3,1)

#hyperparameter grid search
# Create dictionary to store MAEs
mae_grid = dict()
# Outer loop: Iterate through possible values for `p`
for p in p_params:
    # Create key-value pair in dict. Key is `p`, value is empty list.
    mae_grid[p] = list()
    # Inner loop: Iterate through possible values for `q`
    for q in q_params:
        # Combination of hyperparameters for model
        order = (p, 0, q)
        # Note start time
        start_time = time.time()
        # Train model
        model = ARIMA(y_train, order=order).fit
        # Calculate model training time
        elapsed_time = round(time.time() - start_time, 2)
        print(f"Trained ARIMA {order} in {elapsed_time} seconds.")
        # Generate in-sample (training) predictions
        y_pred = model.predict()
        # Calculate training MAE
        mae = mean_absolute_error(y_train, y_pred)
        # Append MAE to list in dictionary
        mae_grid[p].append(mae)

print()
print(mae_grid)

mae_df = pd.DataFrame(mae_grid)
mae_df.round(4)

#gridsearch heatmap
sns.heatmap(mae_df, cmap= "Blues")
plt.xlabel("p values")
plt.ylabel("q values")
plt.title("ARMA Grid search (criterion : mae)");

#plot diagnostics
fig, ax = plt.subplots(figsize=(15, 12))
model.plot_diagnostics(fig=fig);

#evaluation
y_pred_wfv = pd.Series()
history = y_train.copy()
for i in range(len(y_test)):
    model = ARIMA(history, order=(8,0,1)).fit()
    next_pred = model.forecast()
    y_pred_wfv = y_pred_wfv.append(next_pred)
    history = history.append(y_test[next_pred.index])
    
test_mae = mean_absolute_error(y_test, y_pred_wfv)
print("Test MAE (walk forward validation):", round(test_mae, 2))

#plotting the wfv predictions
df_predictions = pd.DataFrame({"y_test": y_test, "y_pred_wfv": y_pred_wfv})
fig = px.line(df_predictions, labels={"value":"PM2.5"})
fig.show()
