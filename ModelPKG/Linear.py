from BaseModel import abstractmodel
import tflearn

#this is just a test class
#not working yet

class model(abstractmodel):
    def train(self, data):
        print("Linear Regression train")

        # Linear Regression graph
        input_ = tflearn.input_data(shape=[None,31])
        linear = tflearn.single_unit(input_)
        regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
                                        metric='R2', learning_rate=0.01)
        m = tflearn.DNN(regression)
        examples = data[:, 0:data.shape[1] - 2]
        labels = data[:, data.shape[1] - 1]

        m.fit(examples, labels, n_epoch=1000, show_metric=True, snapshot_epoch=False)

        print("\nRegression result:")
        print("Y = " + str(m.get_weights(linear.W)) +
              "*X + " + str(m.get_weights(linear.b)))

        print("\nTest prediction for x = 3.2, 3.3, 3.4:")
        print(m.predict([3.2, 3.3, 3.4]))