import xml.etree.ElementTree as ET
import lxml
import csv
tree = ET.parse('feature_values_1.xml')
root = tree.getroot()

name2 = []
array_data = []
name = []
value = []
data_set_id = []
intro = "Name of the song"
data_set_id.append(intro)
#root = ET.fromstring(country_data_as_string)
#print root.tag #this gives the output as the first tag which is feature-vector-file
#for child in root: #this gives the output as the first set of children which are comments, data_sets(2) - for two files.
	#print child.tag, child.attrib
	#for child2 in child: #this is accessing the next child, which are the data_set_id and features for each data_set
	#	print child2.tag, child2.attrib

for data in root.findall('data_set'):
	data_id = data.find('data_set_id').text
	feature = data.find('feature')
	data_set_id.append(data_id)
	#print array_data
	#print data_set_id #this is for finding the song name
	for features in data.findall('feature'): #giving the name and values to 
		#name_val = features.find('name').text
		value_val = features.find('v').text
		#print name, value
		#name.append(name_val)
		value.append(value_val)
	#while (data_id == "/home/suma1998/Music/bird.wav"):
	for features in data.findall('feature'):
		name_val = features.find('name').text
		if name_val not in name:
			name.append(name_val)
		name2.append(name_val)
	#for features in data.findall('feature'):
array_data.append(name)
array_data.append(value)
		#print name
		#print value
'''data_id = data.find('data_set_id').text
feature = data.find('feature')
data_set_id.append(data_id)'''
#print array_data
'''with open("output.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(array_data)'''
final_array = []
print array_data
#print data_id
print data_set_id
print name
print value
for i in range (2*len(name)):
	final_array.append(name2[i])
	final_array.append(value[i])

print final_array
with open('output.csv', 'wb') as f:
    writer = csv.writer(f)
    for val in data_set_id:
        writer.writerow([val])

with open('output1.csv','wb') as f:
    w = csv.writer(f)
    w.writerow(name)
    w.writerow(value)

