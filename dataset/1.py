prepossessed_dataset.main()
dataset = prepossessed_dataset.main()

print(x_train    , dataset[x_train])
print(y_train    , dataset[y_train])
x_train = dataset[x_train]
y_train = dataset[y_train]
x_test = dataset[x_test]
y_test = dataset[y_test]
x_cv = dataset[x_cv]
y_cv = dataset[y_cv]
print('Number of data points in train data', x_train.shape[0])
print('Number of data points in test data', x_test.shape[0])
print('Number of data points in test data', x_cv.shape[0])