import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 讀取資料
data = pd.read_csv('能源供給.csv')

# 定義中英文對照字典
column_mapping = {
    '日期(年)': 'Year',
}
# 替換表頭
data.rename(columns=column_mapping, inplace=True)

# 定義特徵和目標
features = [
            # '自產能源_再生能源_風力', 
            '自產能源_再生能源_小計',
            '進口能源_煙煤-煉焦煤',
            # '進口能源_亞煙煤', 
            '進口能源_原油', 
            '進口能源_液化天然氣', 
            '進口能源_核能']
target = '能源總供給'

# 將資料分為訓練集和測試集
train_data = data[data['Year'] <= 2020]
test_data = data[data['Year'] >= 2021]

# 定義特徵和目標變量
train_X = train_data[features]
train_y = train_data[target]
test_X = test_data[features]

# 建立並訓練線性回歸模型
lm = LinearRegression()
lm.fit(train_X, train_y)

# 使用訓練好的模型預測2021.2022年的能源供給量預測
future_years = [2021, 2022]
future_predictions = lm.predict(test_X)

# 顯示2021.2022的能源供給量預測值
for year, prediction in zip(future_years, future_predictions):
    print(f"{year}年 預測能源總供給量：{int(prediction)}")
    actual_value = test_data[test_data['Year'] == year][target].values[0]
    print(f"{year}年 實際能源總供給量：{int(actual_value)}")
# 計算預測值的準確率
r_squared = lm.score(train_X, train_y)
print("R_squared值=","%.4f" % r_squared)
print()
#%%
# 計算成長比，並儲存結果
growth_rates = []
for column in features:
    growth_rates_column = []
    for i in range(len(data) - 1):
        growth_rate = (data[column][i + 1] - data[column][i]) / data[column][i]
        growth_rates_column.append(growth_rate)
    growth_rates.append(growth_rates_column)
    

# 列印出每個欄位的成長率
print("各特徵欄位平均成長比率：")
for i, column in enumerate(features):
    avg_growth_rate_column = sum(growth_rates[i]) / len(growth_rates[i])
    print(f"{column}:{avg_growth_rate_column}")
print()
#%% 用平均成長比計算2023、2024、2025的特徵數字
future_years = [2023, 2024, 2025]
prediction_data = pd.DataFrame(columns=features)

for year in future_years:
    next_year_predictions = []
    for column, growth_rates_column in zip(features, growth_rates):
        avg_growth_rate_column = sum(growth_rates_column) / len(growth_rates_column)
        
        if year == 2023:
            last_year_data = data[column].iloc[-1] # 最後一年的數字x平均成長比估算下一年預測值
            next_year_prediction = last_year_data * (1 + avg_growth_rate_column)
        else:
            last_year_prediction = prediction_data.loc[year - 1][column]
            next_year_prediction = last_year_prediction * (1 + avg_growth_rate_column)
        
        next_year_predictions.append(next_year_prediction)
    
    # 將下一年的預測结果存入DataFrame中
    prediction_data.loc[year] = next_year_predictions

# print出2023、2024、2025年的預測结果
for year in future_years:
    print(f"{year}各欄位預測數值：")
    for column in features:
        print(f"{column}: {prediction_data.loc[year][column]:.4f}")
    print()
    
#%% 將2021.2022年特徵帶回train_data，預測2023-2025年數值
# 將2021.2022年資料也帶回train_data
train_data2 = data

# 定義特徵和目標變量
train_X2 = train_data2[features]
train_y2 = train_data2[target]

# 建立並訓練線性回歸模型
lm = LinearRegression()
lm.fit(train_X2, train_y2)

# 將2023年到2025年的特徵預測结果帶回訓練模型中預測總供給量
for year in future_years:
    new_row = {'Year': year}
    for column, prediction in zip(features, prediction_data.loc[year]):
        new_row[column] = prediction
    # train_data2 = train_data2.append(new_row, ignore_index=True)
    train_data2 = pd.concat([train_data2, pd.DataFrame(new_row, index=[0])], ignore_index=True)
# 用 pd.concat 將每個新的預測結果，整合到train_data資料集中，
# 讓模型能夠在這些新的特徵和目標值上進行訓練。

# 使用模型預測2023-2025年的能源總供给量
future_predictions_updated = lm.predict(prediction_data)

# print出2023.2024.2025的能源總供给量預測值
for year, prediction in zip(future_years, future_predictions_updated):
    print(f"{year}年 預測能源總供給量：{int(prediction)}")
