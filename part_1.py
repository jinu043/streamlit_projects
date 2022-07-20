from email import header
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

@st.cache()
def get_data(filename):
    taxi_data = pd.read_csv(filename)
    return taxi_data

with header:
    st.title("Welcome to My awsome data science project")
    st.text("""
    In this I would like to explore the taxi fares of cars in NYC...""")

with dataset:
    st.header("NYC Taxi data set")
    st.text("""
    I found this data from NYC taxi site and looking for further exploration of
    of each features""")
    taxi_data = get_data("taxi_data.csv")
    st.write(taxi_data.head(5))
    st.subheader("Pick up Location ID distribution on the NYC dataset")
    pulocation_dis = pd.DataFrame(taxi_data["PULocationID"].value_counts()).sort_values(by="PULocationID", ascending=False).head(50)
    st.bar_chart(pulocation_dis)


with features:
    st.header("The Features I created")

    st.markdown("* **First Feature:** I created this feature because of this... I calculated it using this logic...")
    st.markdown("* **Second Feature:** I created this feature because of this... I calculated it using this logic...")


with model_training:
    st.header("Time to Train the Model")
    sel_col, displ_col = st.columns(2)
    max_depth = sel_col.slider("What should be the max_depth of the model?", min_value=10,max_value=100,value=20, step=10)
    n_estimators = sel_col.selectbox("How many tree should there be?", options=[100,200,300, "No Limit"], index=0)
    sel_col.text("Here is a list of features in my data:")
    sel_col.write(taxi_data.columns)
    input_feature = sel_col.text_input("Which feature should be used as input feature?", "PULocationID")

    if n_estimators == "No Limit":
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    x = taxi_data[[input_feature]]
    y = taxi_data["trip_distance"]

    regr.fit(x,y)
    prediction = regr.predict(x)

    displ_col.subheader("Mean absolute error of the model is:")
    displ_col.write(mean_absolute_error(y, prediction))

    displ_col.subheader("Mean squared error of the model is:")
    displ_col.write(mean_squared_error(y, prediction, squared=False))

    displ_col.subheader("R squared error of the model is:")
    displ_col.write(r2_score(y,prediction))


