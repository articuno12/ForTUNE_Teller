import pandas as  pd

d1 = pd.read_csv("../dataset/dataset1.csv")
d2 = pd.read_csv("../dataset/dataset2.csv")
d3 = pd.read_csv("../dataset/dataset3.csv")
d4 = pd.read_csv("../dataset/dataset4.csv")
d5 = pd.read_csv("../dataset/dataset5.csv")
#merge datasets
frames = [d1, d2, d3, d4,d5]
result = pd.concat(frames)
print result.shape

#remove songs for which we couldnt find viewcount
valid_result = result[~(result["viewCount"]==0)]
print valid_result.shape

#find out null values
null_values =  valid_result.isnull().sum()
# print type(null_values)

#remove columns with null values
remove = []
for i,row in enumerate(null_values):
    if row > 15:
        remove.append(i)
final_data = valid_result.drop(valid_result.columns[remove],axis=1)
print final_data.shape

#final data in csv
final_data.to_csv("../dataset/final_dataset.csv")
