import pandas as pd   
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter
  
# read_csv function which is used to read the required CSV file 
data = pd.read_csv(r'dealing_with_data\rainfall_data.csv')
data = data[:23012]
print(len(data))
data["date"] = pd.to_datetime(data["date"])

  
# drop function which is used in removing or deleting rows or columns from the CSV files 
data.pop("site") 
data.pop("rainfall_units") 
data.pop("lat") 
data.pop("lon") 

plt.plot(data["date"], data["rainfall"])
locator = AutoDateLocator(minticks=1, maxticks=12)  # Adjust minticks/maxticks as needed
formatter = DateFormatter('%b %Y')  # Display month and year, e.g., "Jan 2023"

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.xticks(rotation=45)

plt.xlabel("Data")
plt.ylabel("Opad (mm)")
plt.title("Opady w Auckland")
plt.grid()
plt.tight_layout()
plt.show()


 