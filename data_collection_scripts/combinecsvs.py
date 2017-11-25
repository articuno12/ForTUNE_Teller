import pandas as  pd
import sys


d1 = pd.read_csv(sys.argv[1])
d2 = pd.read_csv(sys.argv[2])
# d3 = pd.read_csv(sys.argv[3])
# d4 = pd.read_csv(sys.argv[4])
# d5 = pd.read_csv(sys.argv[5])
# d6 = pd.read_csv(sys.argv[1])
#merge datasets
frames = [d1, d2]
result = pd.concat(frames)
print result.shape

#remove songs for which we couldnt find viewcount
valid_result = result[~(result["viewCount"]==0)]
print valid_result.shape
valid_result.to_csv(sys.argv[3])
