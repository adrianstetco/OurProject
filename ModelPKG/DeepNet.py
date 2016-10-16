from BaseModel import abstractmodel
import numpy as np
import tflearn

class model(abstractmodel):
    def train(self, data): 
        print("Deep Net train")
        net = tflearn.input_data(shape=[None, 31])
        net = tflearn.fully_connected(net, 32)
        net = tflearn.fully_connected(net, 32)
        net = tflearn.fully_connected(net, 3, activation='softmax')
        net = tflearn.regression(net)
        # Define model
        model = tflearn.DNN(net)
        # Start training (apply gradient descent algorithm)
        examples=data[:,0:data.shape[1]-2]
        print(examples.shape)
        labels=data[:,data.shape[1]-1]
        print(labels)
        print(type(examples))
        print(type(labels))
        labels = tflearn.data_utils.to_categorical(labels, 3)
        model.fit(examples, labels , n_epoch=10, batch_size=20, show_metric=True)
        return model
    
    def test(self):
        print("Deep Net test")
    
