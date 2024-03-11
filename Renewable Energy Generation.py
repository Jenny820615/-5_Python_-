import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 設置中文字體為 Arial Unicode MS
# 讀取資料
data = pd.read_csv('再生能源發電量.csv')

# 將日期(年)列設置為索引並轉換為 datetime 格式
data['日期(年)'] = pd.to_datetime(data['日期(年)'], format='%Y')
data.set_index('日期(年)', inplace=True)

# 去除部分單位列
data = data.drop(columns=[ '單位','再生能源發電量合計','地熱','風力_小計','生質能_小計','生質能_固態'	,'生質能_氣態','廢棄物'])

# 調整圖表大小
plt.figure(figsize=(12, 6))  # 調整圖表大小為12*6

# 繪製每個欄位的成長曲線
for column in data.columns:
    plt.plot(data.index, data[column], label=column)

# 設置 x 軸的日期格式和間隔
plt.gca().xaxis.set_major_locator(YearLocator())
plt.gca().xaxis.set_major_formatter(DateFormatter("%Y"))


# 添加標籤和圖例
plt.xlabel('年')
plt.ylabel('單位(千度)')
plt.title('再生能源發電量')
plt.legend()
plt.grid(True)

# 顯示圖形
plt.tight_layout()  # 調整子圖之間的間距，確保所有元素都能顯示完整
plt.savefig('再生能源發電量圖表', dpi=300)
plt.show()