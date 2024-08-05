import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('new_sales_data.csv')

# Create a line chart
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Sales'], marker='o', linestyle='-')

# Adding title and labels
plt.title('Date Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show grid
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()

