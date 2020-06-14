import csv

# summary_f = open('corona_dataset/Chest_xray_Corona_dataset_Summary.csv', 'r')
meta_f = open('corona_dataset/Chest_xray_Corona_Metadata.csv')
rdr = csv.reader(meta_f)
count_train = 0
count_test = 0
train_0 = 0
train_1 = 0
test_0 = 0
test_1 = 0
for line in rdr:
    # print("{} _ {} _ {}".format(line[1], line[2], line[3]))
    if(line[3] == 'TEST'):
        count_test +=1
        if(line[2] == 'Pnemonia'):
            test_1 +=1
        else:
            test_0 += 1
    elif(line[3] == 'TRAIN'):
        count_train += 1
        if(line[2] == 'Pnemonia'):
            train_1 += 1
        else:
            train_0 += 1
print("TRAIN: total {}  0: {} 1: {}".format(count_train, train_0, train_1))
print("TEST: total {}  0: {} 1: {}".format(count_test, test_0, test_1))
meta_f.close()