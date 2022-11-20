import loader

#loading the dataset

train =  loader.Cal256Loader('../data', split= 'train')
test = loader.Cal256Loader('../data', split= 'test')

print(train.__getitem__(123))
print(test.__getitem__(123))