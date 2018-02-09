# Biometter-Classification

> ***THIS REPO IS NOT GOOD FOR CNN REFERENCE***

## module
### data/input_data
#### read_whole_data
Args: None<br>
Returns: image input data; (660, 2, 640*512); pandas.Dataframe<br>

example: data = input_data.read_whole_data()

#### read_train_and_test_data
Args: None<br>
Returns: image input train, test data; (None, 2, 640*512); pandas.Dataframe<br>

example: train, test = read_train_and_test_data()
