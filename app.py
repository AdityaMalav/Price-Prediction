import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy
import tensorflow as tfl
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import  datetime, time

st.set_page_config(page_title="Stock Price-Prediction", page_icon="📊", layout="wide")

css = f"""
     <style>
     .stApp {{
         background-image: url("https://cdn.pixabay.com/photo/2020/08/09/14/25/business-5475664_1280.jpg");
         background-attachment: fixed;
         background-size: cover
     }}
     </style>
     """
st.markdown(css, unsafe_allow_html=True)






### function (model)
def stockPredictionModel(name_of_stock, number_of_share, start_date, end_date):
    df = pd.read_csv('stock_nifty50.csv')
    df.head()
    df["Date"] = pd.to_datetime(df["Date"])
    # name_of_stock = input("Enter Name Of Stock")
    # number_of_share = int(input("Number Of Share"))
    Y = df[df["Symbol"] == name_of_stock][["Close"]]
    # start_date = input("Start Date")
    # end_date = input("End Date")
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    scaler = MinMaxScaler(feature_range=(0, 1))
    Y = scaler.fit_transform(np.array(Y).reshape(-1, 1))

    training_size = int(len(Y) * 0.65)
    test_size = len(Y) - training_size
    train_data, test_data = Y[0:training_size, :], Y[training_size:len(Y), :1]

    # convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return numpy.array(dataX), numpy.array(dataY)

    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    ### Create the Stacked LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=5, batch_size=64, verbose=1)

    last_date = pd.to_datetime("2022-12-12")
    lst_output_1 = []
    lst_output_2 = []

    diff_1 = start_date - last_date
    diff_2 = end_date - last_date
    if int(diff_1.days) > 0:
        n_steps = 100
        x_input = test_data[len(test_data) - n_steps:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()
        i = 0
        while (i < int(diff_1.days)):
            if (len(temp_input) > 100):
                # print(temp_input)
                x_input = np.array(temp_input[1:])
                #                 print("{} day input {}".format(i,x_input))
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                # print(x_input)
                yhat = model.predict(x_input, verbose=0)
                #                 print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                # print(temp_input)
                lst_output_1.extend(yhat.tolist())
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                #                 print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                #                 print(len(temp_input))
                lst_output_1.extend(yhat.tolist())
                i = i + 1

    if int(diff_2.days) > 0:
        n_steps = 100
        x_input = test_data[len(test_data) - n_steps:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()
        i = 0
        while (i < int(diff_2.days)):
            if (len(temp_input) > 100):
                # print(temp_input)
                x_input = np.array(temp_input[1:])
                #                 print("{} day input {}".format(i,x_input))
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                # print(x_input)
                yhat = model.predict(x_input, verbose=0)
                #                 print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                # print(temp_input)
                lst_output_2.extend(yhat.tolist())
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                #                 print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                #                 print(len(temp_input))
                lst_output_2.extend(yhat.tolist())
                i = i + 1
    if int(diff_2.days) > 0 and int(diff_1.days) > 0:
        invests = scaler.inverse_transform(lst_output_1)[-1]
        returns = scaler.inverse_transform(lst_output_2)[-1]
        Final_Amount = ((returns - invests) * (int(number_of_share)))
    elif int(diff_2.days) > 0 and int(diff_1.days) < 0:
        invests = (df[(df['Symbol'] == name_of_stock) & (df['Date'] == start_date)]['Close'].values[0]).reshape(1, -1)
        invests = invests[-1] * (int(number_of_share))
        returns = (scaler.inverse_transform(lst_output_2)[-1]) * (int(number_of_share))
        Final_Amount = (returns - invests)
    else:
        invests = (df[(df['Symbol'] == name_of_stock) & (df['Date'] == start_date)]['Close'].values[0]).reshape(1, -1)
        invests = invests[-1] * (int(number_of_share))
        returns = (df[(df['Symbol'] == name_of_stock) & (df['Date'] == end_date)]['Close'].values[0]).reshape(1, -1)
        returns = returns[-1] * (int(number_of_share))

        Final_Amount = (returns - invests)

    # In[2]:

    if Final_Amount[0] > 0:
        print("Profit: ", Final_Amount[0])
        return Final_Amount[0]
    else:
        print("Loss: ", Final_Amount[0])
        return Final_Amount[0]



###----------------------------------------------------------------------------
col1,mid, col2 = st.columns([1, 30, 40])
with col1:
    st.image('pngwing.com (1).png',width=340)
    st.title("""Stock Price Prediction""")
with col2:
    stock_name =  st.selectbox('Select the Stock', ('ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK',
       'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BPCL', 'BHARTIARTL',
       'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'DRREDDY',
       'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HINDALCO',
       'HINDUNILVR', 'HDFC', 'ICICIBANK', 'ITC', 'INDUSINDBK', 'INFY',
       'JSWSTEEL', 'KOTAKBANK', 'LT', 'M&M', 'MARUTI', 'NTPC',
       'NESTLEIND', 'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN',
       'SUNPHARMA', 'TCS', 'TATAGLOBAL', 'TATACONSUM', 'TATAMOTORS',
       'TATASTEEL', 'TECHM', 'TITAN', 'UPL', 'ULTRACEMCO', 'WIPRO'))

    number_of_stocks = st.number_input(f"Enter {stock_name}'s Number of Stocks.", step=1)
    start_date = st.date_input('Select the Starting date.', datetime.date(2020,9,22))
    end_date = st.date_input('Select the End date.', datetime.date(2021,9,22))

    if st.button('🔎'):
        with st.spinner('Operation in progress. Please wait....'):
            time.sleep(0.1)

            result = round(stockPredictionModel(stock_name, number_of_stocks, start_date, end_date), 2)


            if result < 0:
                st.write(f'##### If you pick {stock_name} from {start_date} to {end_date}.\n#### You will get {result} Loss',
                         unsafe_allow_html=True,
                         style={"color": "black"} )

            else:
                st.write(f'##### If you pick {stock_name} from {start_date} to {end_date}.\n### You will get {result} Profit',
                         unsafe_allow_html=True,
                         style={"color": "black"} )

