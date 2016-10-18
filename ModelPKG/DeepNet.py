from ModelPKG.BaseModel import abstractmodel
import numpy as np
import tflearn

class model(abstractmodel):
    m=0
    def train(self, data,data2): 
        print("Deep Net train")
        net = tflearn.input_data(shape=[None, 31])
        net = tflearn.fully_connected(net, 200)
        net = tflearn.fully_connected(net, 200)
        net = tflearn.fully_connected(net, 200)
        net = tflearn.fully_connected(net, 2, activation='softmax')
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
        labels1 = tflearn.data_utils.to_categorical(labels, 2)
        
        model.fit(examples, labels1 , n_epoch=20, batch_size=50, show_metric=True)
        
        examples2=data2[:,0:data2.shape[1]-2]
        labels2=data2[:,data2.shape[1]-1]
        #print(labels1)
        #print(labels)
        #print(examples2.shape)
        #print(labels2.shape)
        print("Accuracy")
        print(model.evaluate(examples2, tflearn.data_utils.to_categorical(labels2, 2)))
        #print(model.predict(examples2))
        #print(tflearn.data_utils.to_categorical(labels2, 2))
        return model
    
    def test(self, data):
        print("Deep Net test")
        #print(m.predict(data))
        
    
