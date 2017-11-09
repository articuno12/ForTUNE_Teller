import xml.etree.ElementTree as ET
import csv
import lxml
tree = ET.parse("feature_values_1.xml")
root = tree.getroot()

# open a file for writing

songs_data = open('ResidentData.csv', 'w')

# create the csv writer object

csvwriter = csv.writer(songs_data)
resident_head = []

count = 0
for data in root.findall('data_set'):
    resident = []
    if count == 0:
	name = data.find('data_set_id').tag
	resident_head.append(name)
        resident.append(data.find('data_set_id').text)
        for features in data.findall('feature'):
            name_val = features.find('name').text
            resident_head.append(name_val)
            value_val = features.find('v').text
            resident.append(value_val)
        csvwriter.writerow(resident_head)
        csvwriter.writerow(resident)
        count = count + 1
    else:
        resident.append(data.find('data_set_id').text)
        for features in data.findall('feature'):
            name_val = features.find('name').text
            #resident_head.append(name_val)
            value_val = features.find('v').text
            resident.append(value_val)
        csvwriter.writerow(resident)

songs_data.close()
