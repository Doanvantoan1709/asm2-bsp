import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
df = pd.read_csv('ASM1_BPS.csv')

# Loại bỏ dấu phần trăm và chuyển đổi cột 'Market Share' sang kiểu số
df['Market Share'] = df['Market Share'].str.replace('%', '').astype(float)

# Chuyển đổi cột "Average Price Range" thành khoảng giá trị trung bình
def average_price_range(price_range):
    prices = price_range.replace('$', '').split(' - ')
    return (int(prices[0]) + int(prices[1])) / 2

df['Average Price'] = df['Average Price Range'].apply(average_price_range)

# Trực quan hóa dữ liệu
plt.figure(figsize=(15, 10))

# Biểu đồ 1: Phân bố các nhóm sản phẩm theo thị phần
plt.subplot(2, 2, 1)
df.groupby('Product Group Name')['Market Share'].sum().plot(kind='bar')
plt.title('Distribution of product groups by market share')
plt.xlabel('Product Group Name')
plt.ylabel('Market Share (%)')


# Biểu đồ 2: Phân bố giá trung bình của các nhóm sản phẩm
plt.subplot(2, 2, 2)
df.groupby('Product Group Name')['Average Price'].mean().plot(kind='bar', color='orange')
plt.title('Distribution of average prices of product groups')
plt.xlabel('Product Group Name')
plt.ylabel('Average Price ($)')

# Biểu đồ 3: Phân bố các nhóm sản phẩm theo nhóm khách hàng mục tiêu
plt.subplot(2, 2, 3)
df.groupby('Target Market')['Product Group ID'].count().plot(kind='bar', color='green')
plt.title('Distribute product groups according to target customer groups')
plt.xlabel('Target Market')
plt.ylabel('Number of Products')

# Biểu đồ 4: Phân bố các sản phẩm theo ngày ra mắt
plt.subplot(2, 2, 4)
df['Launch Date'] = pd.to_datetime(df['Launch Date'])
df.groupby(df['Launch Date'].dt.year)['Product Group ID'].count().plot(kind='bar', color='purple')
plt.title('Distribution of products by launch date')
plt.xlabel('Year')
plt.ylabel('Number of Products')

plt.tight_layout()
plt.show()


