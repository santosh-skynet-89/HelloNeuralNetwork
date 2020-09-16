import numpy as np
import scipy.special

class NeuralNet:

    def __init__(self,input_nodes,output_nodes,hidden_nodes,lr):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_nodes = hidden_nodes
        self.lr = lr
        self.Winput_hidden = np.random.normal(0.0,pow(self.input_nodes,-0.5),(self.hidden_nodes,self.input_nodes))
        self.Whidden_output = np.random.normal(0.0,pow(self.hidden_nodes,-0.5),(self.output_nodes,self.hidden_nodes))
        self.activation_function = lambda x:scipy.special.expit(x)
        pass

    def train(self,input_list,target_list):
        self.inputs = np.array(input_list, ndmin=2).T
        self.target = np.array(target_list,ndmin=2).T
        self.hidden_inputs = np.dot(self.Winput_hidden, self.inputs)
        self.hidden_outputs = self.activation_function(self.hidden_inputs)
        self.final_inputs = np.dot(self.Whidden_output, self.hidden_outputs)
        self.final_outputs = self.activation_function(self.final_inputs)

        #calculate the error made
        self.output_error = self.target - self.final_outputs

        #calculate hidden layer error  "Backpropagation"
        self.hidden_error = np.dot(self.Whidden_output.T,self.output_error)

        #update the weight b/w the output and the hidden layer "Gradient Descent"
        self.Whidden_output += self.lr * np.dot((self.output_error * self.final_outputs * (1.0 - self.final_outputs)),np.transpose(self.hidden_outputs))

        # update the weight b/w the input and the hidden layer
        self.Winput_hidden += self.lr * np.dot((self.hidden_error * self.hidden_outputs * (1.0 - self.hidden_outputs)),np.transpose(self.inputs))

    def predict(self,input_list):
        self.inputs = np.array(input_list,ndmin=2).T
        self.hidden_inputs = np.dot(self.Winput_hidden,self.input_nodes)
        self.hidden_outputs = self.activation_function(self.hidden_inputs)
        self.final_inputs = np.dot(self.Whidden_output,self.hidden_outputs)
        self.final_outputs = self.activation_function(self.final_inputs)

        return  self.final_outputs

#parameters
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.3

#NeuralNetwork
sk = NeuralNet(input_nodes,output_nodes,hidden_nodes,learning_rate)

#loading Data
data_file = open("mnist_train.csv", 'r')
data_list = data_file.readlines()
data_file.close()

print("Started Training")

epochs = 7

for e in range(epochs):
    for records in data_list:

        #split the commas
        array = records.split(",")

        #convert to numpy array and scaling it to 0.01 to 0.99
        inputs = (np.asfarray(array[1:]) / 255.0 * 0.99) + 0.01

        #create an target array [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
        targets = np.zeros(output_nodes) + 0.01

        #set the tageted value to 0.99
        targets[int(array[0])] = 0.99

        #train the network
        sk.train(inputs,targets)

test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

print("Started Testing")
for records in test_data_list:

    test_array = records.split(",")
    correct_label = test_array[0]
    test_inputs = (np.asfarray(array[1:]) / 255.0 * 0.99) + 0.01
    output = sk.predict(test_inputs)
    if output == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_array = np.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)
