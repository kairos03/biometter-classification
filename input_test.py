# Copyright kairos03 2017. All Right Reserved.

from data import input_data

train, test = input_data.read_train_and_test_data()

print(train)
print(train['image'][:10])
print(type(train['image']))
print(len(train['image']))
print(train['image'][0].shape)
