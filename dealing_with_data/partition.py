import pandas as pd   

# read_csv function which is used to read the required CSV file 
data = pd.read_csv('dealing_with_data\BMW_Data.csv')
#data = data[:1000]
data["Date"] = pd.to_datetime(data["Date"])
print(len(data))

  
# drop function which is used in removing or deleting rows or columns from the CSV files 
data.pop("Adj_Close") 
data.pop("High") 
data.pop("Low") 
data.pop("Open") 
data.pop("Volume") 

data_test = data[6490:]
data_train = data[:6490]
if data_train[0] == data_test[0]:
    print('sth wrong')
