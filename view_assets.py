import pandas as pd
import os
import matplotlib.pyplot as plt

asset_files = []
for file in os.listdir("Data"):
    if file.endswith(".xlsx"):
        name = os.path.join("Data", file)
        asset_files.append(name.replace("\\", "/"))

plt.figure
excel = asset_files[0]
df_main = pd.read_excel(excel, header = None, names = ["date", excel[5:-5]])
df_main[excel[5:-5]] = (df_main[excel[5:-5]] - min(df_main[excel[5:-5]]))/max(df_main[excel[5:-5]])
df_main.set_index("date", drop = True, inplace = True)

for excels in asset_files[1:]:
    df = pd.read_excel(excels, header = None, names = ["date", excels[5:-5]])
    df[excels[5:-5]] = (df[excels[5:-5]] - min(df[excels[5:-5]]))/max(df[excels[5:-5]])
    df.set_index("date", drop = True, inplace = True)
    df_main = df_main.join(df, how = "outer")

print(df_main)

df_main.plot()
plt.show()

#     plt.plot(df["date"], df["price"])
#     plt.xlabel("date")
#     plt.ylabel("price")
#     plt.title(excel[:-5])
# plt.show()





