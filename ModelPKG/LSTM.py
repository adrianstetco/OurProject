from ModelPKG.BaseModel import abstractmodel
import tflearn

class model(abstractmodel):
    def __init__(self):
        self.model = None # model

    def train(self, data):
        print("LSTM train")
        net = tflearn.input_data(shape=[None, data.shape[1], data.shape[2]])
        net = tflearn.lstm(net, 128, return_seq=True)
        net = tflearn.lstm(net, 128)
        net = tflearn.fully_connected(net, 2, activation='softmax')
        net = tflearn.regression(net, optimizer='adam',
                                 loss='categorical_crossentropy', name="output1")
        self.model = tflearn.DNN(net, tensorboard_verbose=2)

        #examples = data[:, 0:data.shape[1] - 2]
        #print(examples.shape)
        labels = data[:, data.shape[1] - 1]
        #print(labels)
        #print(type(examples))
        #print(type(labels))
        labels = tflearn.data_utils.to_categorical(labels, 2)

        self.model.fit(data, labels, n_epoch=5, validation_set=0.1, show_metric=True,
                  snapshot_step=100)

