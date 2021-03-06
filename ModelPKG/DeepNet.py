from ModelPKG.BaseModel import abstractmodel
import tflearn

class model(abstractmodel):
    def __init__(self):
        self.model = None # model

    def train(self, data):
        print("Deep Net train")
        net = tflearn.input_data(shape=[None, 31])
        net = tflearn.fully_connected(net, 200)
        net = tflearn.fully_connected(net, 200)
        net = tflearn.fully_connected(net, 200)
        net = tflearn.fully_connected(net, 2, activation='softmax')
        net = tflearn.regression(net)
        # Define model
        self.model = tflearn.DNN(net)
        # Start training (apply gradient descent algorithm)
        examples=data[:,0:data.shape[1]-2]
        print(examples.shape)
        labels=data[:,data.shape[1]-1]
        print(labels)
        print(type(examples))
        print(type(labels))
        labels1 = tflearn.data_utils.to_categorical(labels, 2)
        
        self.model.fit(examples, labels1 , n_epoch=20, batch_size=50, show_metric=True)

    def test(self, data):
        print("Deep Net test")
        examples2 = data[:, 0:data.shape[1] - 2]
        labels2 = data[:, data.shape[1] - 1]
        print("Accuracy")
        print(self.model.evaluate(examples2, tflearn.data_utils.to_categorical(labels2, 2)))

        print(self.model.predict(examples2))
        
    
