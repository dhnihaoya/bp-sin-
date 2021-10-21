import math
import random
import matplotlib.pyplot as plt
import numpy


def sigmoid(x):
    return 1.0 / (1 + numpy.exp(-x))


def tanh(x):
    exp = numpy.exp(x)
    return (exp - 1 / exp) / (exp + 1 / exp)


class Layer(object):
    def __init__(self, input_size, output_size, random_scale, is_last):
        self.bias = numpy.random.normal(scale=random_scale, size=(output_size, 1))
        self.weight = numpy.random.normal(scale=random_scale, size=(input_size, output_size))

        self.input = numpy.zeros((input_size, 1))
        self.sums = numpy.zeros((output_size, 1))
        self.result = numpy.zeros((output_size, 1))
        # 暂时存储一个batch内的改变
        self.batch_delta_weight = numpy.zeros(self.weight.shape)
        self.batch_delta_bias = numpy.zeros(self.bias.shape)
        self.crt_batch_cnt = 0
        self.is_last = is_last

    # 计算输入的加权和
    def calc_sum(self, crt_input):
        self.input = crt_input
        input_sum = self.weight.T.dot(self.input) + self.bias
        self.sums = input_sum
        return self.sums

    def activate(self):
        if not self.is_last:
            self.result = tanh(self.sums)
        else:
            self.result = self.sums
        return self.result

    def activate_derive(self):
        tan = tanh(self.sums)
        return 1 - tan ** 2

    def save_batch(self, weight, bias):
        self.batch_delta_weight += weight
        self.batch_delta_bias += bias
        self.crt_batch_cnt += 1

    def change_weight(self, learning_rate):
        delta_weight = learning_rate * self.batch_delta_weight / self.crt_batch_cnt
        delta_bias = learning_rate * self.batch_delta_bias / self.crt_batch_cnt

        self.crt_batch_cnt = 0
        self.batch_delta_bias = numpy.zeros(self.bias.shape)
        self.batch_delta_weight = numpy.zeros(self.weight.shape)

        self.weight += delta_weight
        self.bias += delta_bias

    def forward(self, crt_input):
        self.calc_sum(crt_input)
        self.activate()
        return self.result

    def backward(self, loss):
        bias_gradient = loss * self.activate_derive()
        weight_gradient = self.input.dot(bias_gradient.T)
        self.save_batch(-weight_gradient, -bias_gradient)
        loss_next = self.weight.dot(bias_gradient)
        return loss_next


class NeuralNetwork(object):
    def __init__(self, neuron_nums, learning_rate, scale=0.15, training_size=3000):
        if len(neuron_nums) < 2:
            print("网络至少需要两层")
            exit(1)
        self.learning_rate = learning_rate
        self.neuron_nums = neuron_nums
        self.training_data = numpy.linspace(-math.pi, math.pi, training_size)
        numpy.random.shuffle(self.training_data)

        # init layers
        self.layers = []
        for i in range(0, len(neuron_nums) - 1):
            if i == len(neuron_nums) - 2:
                self.layers.append(Layer(neuron_nums[i], neuron_nums[i + 1], scale, True))
            else:
                self.layers.append(Layer(neuron_nums[i], neuron_nums[i + 1], scale, False))

        # init test data
        self.test_data = []

    def forward(self, crt_input):
        for layer in self.layers:
            crt_input = layer.forward(crt_input)
        return crt_input

    def backward(self, loss):
        for layer in reversed(self.layers):
            loss = layer.backward(loss)

    def change_weight(self, learning_rate):
        for layer in self.layers:
            layer.change_weight(learning_rate)

    def train(self, batch_size=20):
        numpy.random.shuffle(self.training_data)
        expected_result = numpy.sin(self.training_data)
        batch_num = 0
        for i in range(0, len(self.training_data)):
            cur_predict = self.forward(numpy.array([[self.training_data[i]]]))
            cur_loss = cur_predict - numpy.array([[expected_result[i]]])
            self.backward(cur_loss)
            batch_num += 1
            if batch_num == batch_size:
                batch_num = 0
                self.change_weight(self.learning_rate)

    def draw(self, test_result, epoch_num):
        test_data = self.test_data
        plt.plot(test_data, numpy.sin(test_data))
        plt.plot(test_data, test_result)
        plt.title("epoch" + str(epoch_num))
        plt.show()

    def get_test(self):
        self.test_data = []
        for i in range(0, 200):
            self.test_data.append(random.uniform(-math.pi, math.pi))
        self.test_data.sort()
        return self.test_data

    def check_accuracy(self, epoch_num, draw=False):
        accumulated_loss = 0
        test = self.get_test()
        test_results = []
        for i in range(0, len(self.test_data)):
            cur_predict = self.forward(numpy.array([[test[i]]]))
            test_results.append(cur_predict[0][0])
            accumulated_loss += abs(numpy.sin(test[i]) - cur_predict)
        if draw:
            self.draw(test_results, epoch_num)
        print("test accuracy,在 %d 次中产生了 %f 误差" % (len(test_results), accumulated_loss))
        avg_loss = accumulated_loss / len(test_results)
        print("平均误差是" + str(avg_loss))
        return avg_loss


if __name__ == '__main__':
    bp_sin = NeuralNetwork(neuron_nums=[1, 64, 64, 1], learning_rate=0.01)
    bp_sin.check_accuracy(True)
    x = []
    y = []
    for epoch in range(0, 3001):
        bp_sin.train()
        if epoch % 200 == 0:
            print("it is epoch" + str(epoch))
            x.append(epoch)
            y.append(bp_sin.check_accuracy(epoch, True)[0][0])
    plt.plot(x, y)
    plt.show()




