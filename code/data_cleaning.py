import pandas as  pd
import sys

valid_result = pd.read_csv(sys.argv[1])

#find out null values
null_values =  valid_result.isnull().sum()
print null_values
#remove columns with null values
remove = []
for i,row in enumerate(null_values):
    if row > 15:
        remove.append(i)
final_data = valid_result.drop(valid_result.columns[remove],axis=1)
print final_data.shape

#final data in csv
final_data.to_csv("../dataset/final_dataset_debut_cleaned.csv")

# for cases where we are handling missing values, use this dataset
# remove = []
# for i,row in enumerate(null_values):
#     if row > 150:
#         remove.append(i)
# final_data_missing = valid_result.drop(valid_result.columns[remove],axis=1)
# print final_data_missing.shape
# final_data_missing.to_csv("../dataset/dataset_with_missing_values.csv")
