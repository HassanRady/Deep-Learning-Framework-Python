import unittest
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter
import NeuralNetwork
import matplotlib.pyplot as plt
import tabulate

from DeepStorm.model import Model
from DeepStorm.Layers.conv import Conv2d
from DeepStorm.Layers.batch_normalization import BatchNorm2d
from DeepStorm.Layers.dropout import Dropout
from DeepStorm.Layers.pooling import MaxPool2d
from DeepStorm.Layers.flatten import Flatten
from DeepStorm.Layers.linear import Linear
from DeepStorm.Initializers.xavier import Xavier
from DeepStorm.Initializers.he import He
from DeepStorm.Initializers.uniform_random import UniformRandom
from DeepStorm.Initializers.constant import Constant
from DeepStorm.Activations.relu import ReLU
from DeepStorm.Activations.sigmoid import Sigmoid
from DeepStorm.Activations.softmax import SoftMax
from DeepStorm.Optimizers.adam import Adam
from DeepStorm.Optimizers.sgd import Sgd, SgdWithMomentum
from DeepStorm.Losses.cross_entropy import CrossEntropyLoss

from tests import helpers

ID = 2  # identifier for dispatcher
BATCH_SIZE = 32
METRICS = ['accuracy']









class TestModel(unittest.TestCase):
    plot = False
    directory = 'plots/'
    log = 'log.txt'
    iterations = 100

    def test_append_layer(self):
        net = Model()
        fcl_1 = Linear(1, 1, weights_initializer=Constant(
            0.123), bias_initializer=Constant(0.123))
        net.append_layer(fcl_1)
        fcl_2 = Linear(1, 1, weights_initializer=Constant(
            0.23), bias_initializer=Constant(0.23))
        net.append_layer(fcl_2)

        net.compile(Sgd(1e-4), CrossEntropyLoss(), BATCH_SIZE, METRICS)

        self.assertEqual(len(net.layers), 2)
        self.assertFalse(net.layers[0].optimizer is net.layers[1].optimizer)
        self.assertTrue(np.all(net.layers[0].weights == 0.123))

    def test_data_access(self):
        np.random.seed(None)
        net = NeuralNetwork.NeuralNetwork(Sgd(1e-4),
                                          )
        categories = 3
        input_size = 4
        net.data_layer = helpers.IrisData(50)
        net.loss_layer = CrossEntropyLoss()
        fcl_1 = Linear(input_size, categories,
                       weights_initializer=UniformRandom())
        net.append_layer(fcl_1)
        net.append_layer(ReLU())
        fcl_2 = Linear(categories, categories, UniformRandom())
        net.append_layer(fcl_2)
        net.append_layer(SoftMax())

        out = net.forward()
        out2 = net.forward()

        self.assertNotEqual(out, out2)

    def test_iris_data(self):
        np.random.seed(None)
        net = NeuralNetwork.NeuralNetwork(Sgd(1e-3),
                                          )
        categories = 3
        input_size = 4
        net.data_layer = helpers.IrisData(100)
        net.loss_layer = CrossEntropyLoss()
        fcl_1 = Linear(input_size, categories, UniformRandom())
        net.append_layer(fcl_1)
        net.append_layer(ReLU())
        fcl_2 = Linear(categories, categories, UniformRandom())
        net.append_layer(fcl_2)
        net.append_layer(SoftMax())

        net.train(4000)
        if TestModel.plot:
            fig = plt.figure(
                'Loss function for a Neural Net on the Iris dataset using SGD')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestModel3.pdf"),
                        transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset, we achieve an accuracy of: ' +
                  str(accuracy * 100) + '%', file=f)
        self.assertGreater(accuracy, 0.9)

    def _test_regularization_loss(self):
        '''
        This test checks if the regularization loss is calculated for the fc and rnn layer and tracked in the
        NeuralNetwork class
        '''
        import random
        fcl = FullyConnected(4, 3)
        rnn = RNN.RNN(4, 4, 3)

        for layer in [fcl, rnn]:
            loss = []
            for reg in [False, True]:
                opt = Sgd(1e-3)
                if reg:
                    opt.add_regularizer(Constraints.L1_Regularizer(8e-2))
                net = NeuralNetwork.NeuralNetwork(opt, Constant(0.5),
                                                  Constant(0.1))

                net.data_layer = helpers.IrisData(100, random=False)
                net.loss_layer = CrossEntropyLoss()
                net.append_layer(layer)
                net.append_layer(SoftMax())
                net.train(1)
                loss.append(np.sum(net.loss))

            self.assertNotEqual(loss[0], loss[1], "Regularization Loss is not calculated and added to the overall loss "
                                                  "for " + layer.__class__.__name__)

    def test_iris_data_with_momentum(self):
        np.random.seed(None)
        net = NeuralNetwork.NeuralNetwork(SgdWithMomentum(1e-3, 0.8),)
        categories = 3
        input_size = 4
        net.data_layer = helpers.IrisData(100)
        net.loss_layer = CrossEntropyLoss()
        fcl_1 = Linear(input_size, categories, UniformRandom())
        net.append_layer(fcl_1)
        net.append_layer(ReLU())
        fcl_2 = Linear(categories, categories, UniformRandom())
        net.append_layer(fcl_2)
        net.append_layer(SoftMax())

        net.train(2000)
        if TestModel.plot:
            fig = plt.figure(
                'Loss function for a Neural Net on the Iris dataset using Momentum')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestModel3_Momentum.pdf"),
                        transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset, we achieve an accuracy of: ' +
                  str(accuracy * 100) + '%', file=f)
        self.assertGreater(accuracy, 0.9)

    def test_iris_data_with_adam(self):
        np.random.seed(None)
        net = NeuralNetwork.NeuralNetwork(Adam(1e-3, 0.9, 0.999),)
        categories = 3
        input_size = 4
        net.data_layer = helpers.IrisData(100)
        net.loss_layer = CrossEntropyLoss()
        fcl_1 = Linear(input_size, categories, UniformRandom())
        net.append_layer(fcl_1)
        net.append_layer(ReLU())
        fcl_2 = Linear(categories, categories, UniformRandom())
        net.append_layer(fcl_2)
        net.append_layer(SoftMax())

        net.train(3000)
        if TestModel.plot:
            fig = plt.figure(
                'Loss function for a Neural Net on the Iris dataset using ADAM')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestModel3_ADAM.pdf"),
                        transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset, we achieve an accuracy of: ' +
                  str(accuracy * 100) + '%', file=f)
        self.assertGreater(accuracy, 0.9)

    def test_iris_data_with_batchnorm(self):
        np.random.seed(None)
        net = NeuralNetwork.NeuralNetwork(Adam(1e-2, 0.9, 0.999),)
        categories = 3
        input_size = 4
        net.data_layer = helpers.IrisData(50)
        net.loss_layer = CrossEntropyLoss()
        net.append_layer(BatchNorm2d(input_size))
        fcl_1 = Linear(input_size, categories, UniformRandom())
        net.append_layer(fcl_1)
        net.append_layer(ReLU())
        fcl_2 = Linear(categories, categories, UniformRandom())
        net.append_layer(fcl_2)
        net.append_layer(SoftMax())

        net.train(2000)
        if TestModel.plot:
            fig = plt.figure(
                'Loss function for a Neural Net on the Iris dataset using Batchnorm')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestModel3_Batchnorm.pdf"),
                        transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        results_next_run = net.test(data)

        accuracy = helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset using Batchnorm, we achieve an accuracy of: ' +
                  str(accuracy * 100.) + '%', file=f)
        self.assertGreater(accuracy, 0.8)
        self.assertEqual(np.mean(np.square(results - results_next_run)), 0)

    def test_iris_data_with_dropout(self):
        np.random.seed(None)
        net = NeuralNetwork.NeuralNetwork(Adam(1e-2, 0.9, 0.999),
                                          UniformRandom(),
                                          Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = helpers.IrisData(50)
        net.loss_layer = CrossEntropyLoss()
        fcl_1 = Linear(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU())
        fcl_2 = Linear(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(Dropout(0.3))
        net.append_layer(SoftMax())

        net.train(2000)
        if TestModel.plot:
            fig = plt.figure(
                'Loss function for a Neural Net on the Iris dataset using Dropout')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestModel3_pdf"),
                        transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = helpers.calculate_accuracy(results, labels)

        results_next_run = net.test(data)

        with open(self.log, 'a') as f:
            print('On the Iris dataset using Dropout, we achieve an accuracy of: ' +
                  str(accuracy * 100.) + '%', file=f)
        self.assertEqual(np.mean(np.square(results - results_next_run)), 0)

    def test_layer_phases(self):
        np.random.seed(None)
        net = Model()
        categories = 3
        input_size = 4
        net.compile(Adam(1e-2, 0.9, 0.999),
                    CrossEntropyLoss(), input_size, METRICS)
        net.data_layer = helpers.IrisData(50)
        net.append_layer(BatchNorm2d(input_size))
        fcl_1 = Linear(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU())
        fcl_2 = Linear(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(Dropout(0.3))
        net.append_layer(SoftMax())

        net.train(100)

        data, labels = net.data_layer.get_test_set()
        results = net.test(data)

        bn_phase = net.layers[0].testing_phase
        drop_phase = net.layers[4].testing_phase

        self.assertTrue(bn_phase)
        self.assertTrue(drop_phase)

    def test_digit_data2(self):
        adam = Adam(5e-3, 0.98, 0.999)
        self._perform_test(
            adam, TestModel.iterations, 'ADAM', False, False)

    def test_digit_data(self):
        net = NeuralNetwork.NeuralNetwork(Adam(5e-3, 0.98, 0.999),
                                          )
        input_image_shape = (1, 8, 8)
        conv_stride_shape = (1, 1)
        convolution_shape = (1, 3, 3)
        categories = 10
        batch_size = 200
        num_kernels = 4

        net.data_layer = helpers.DigitData(batch_size)
        net.loss_layer = CrossEntropyLoss()

        cl_1 = Conv2d(stride=conv_stride_shape, kernel_size=3,
                      in_channels=1, out_channels=num_kernels, padding="same")
        net.append_layer(cl_1)
        cl_1_output_shape = (*input_image_shape[1:], num_kernels)
        net.append_layer(ReLU())

        pool = MaxPool2d((2, 2), (2, 2))
        pool_output_shape = (4, 4, 4)
        net.append_layer(pool)
        fcl_1_input_size = np.prod(pool_output_shape)

        net.append_layer(Flatten())

        fcl_1 = Linear(fcl_1_input_size, int(fcl_1_input_size/2.))
        net.append_layer(fcl_1)

        net.append_layer(ReLU())

        fcl_2 = Linear(int(fcl_1_input_size/2.), categories)
        net.append_layer(fcl_2)

        net.append_layer(SoftMax())

        net.train(200)

        if TestModel.plot:
            description = 'on_digit_data'
            fig = plt.figure(
                'Loss function for training a Convnet on the Digit dataset')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestConvNet_" + description +
                        ".pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the UCI ML hand-written digits dataset, we achieve an accuracy of: ' +
                  str(accuracy * 100) + '%', file=f)
        print('\nOn the UCI ML hand-written digits dataset, we achieve an accuracy of: ' +
              str(accuracy * 100) + '%')
        self.assertGreater(accuracy, 0.5)

    def _test_digit_data_L2_Regularizer(self):
        sgd_with_l2 = Adam(5e-3, 0.98, 0.999)
        sgd_with_l2.add_regularizer(Constraints.L2_Regularizer(8e-2))
        self._perform_test(
            sgd_with_l2, TestModel.iterations, 'L2_regularizer', False, False)

    def _test_digit_data_L1_Regularizer(self):
        sgd_with_l1 = Adam(5e-3, 0.98, 0.999)
        sgd_with_l1.add_regularizer(Constraints.L1_Regularizer(8e-2))
        self._perform_test(
            sgd_with_l1, TestModel.iterations, 'L1_regularizer', False, False)

    def test_digit_data_dropout(self):
        sgd_with_l2 = Adam(5e-3, 0.98, 0.999)
        # sgd_with_l2.add_regularizer(Constraints.L2_Regularizer(4e-4))
        self._perform_test(
            sgd_with_l2, TestModel.iterations, 'Dropout', True, False)

    def test_digit_batch_norm(self):
        adam = Adam(1e-2, 0.98, 0.999)
        self._perform_test(adam, TestModel.iterations,
                           'Batch_norm', False, True)

    def _test_all(self):
        sgd_with_l2 = Adam(1e-2, 0.98, 0.999)
        sgd_with_l2.add_regularizer(Constraints.L2_Regularizer(8e-2))
        self._perform_test(
            sgd_with_l2, TestModel.iterations, 'Batch_norm and L2', False, True)

    def _perform_test(self, optimizer, iterations, description, dropout, batch_norm):
        np.random.seed(None)
        net = NeuralNetwork.NeuralNetwork(optimizer,
                                          )
        input_image_shape = (1, 8, 8)
        conv_stride_shape = (1, 1)
        convolution_shape = (1, 3, 3)
        categories = 10
        batch_size = 200
        num_kernels = 4

        net.data_layer = helpers.DigitData(batch_size)
        net.loss_layer = CrossEntropyLoss()

        if batch_norm:
            net.append_layer(BatchNorm2d(1))

        cl_1 = Conv2d(stride=conv_stride_shape, in_channels=1,
                      kernel_size=3, out_channels=num_kernels, padding="same")
        net.append_layer(cl_1)
        cl_1_output_shape = (num_kernels, *input_image_shape[1:])

        if batch_norm:
            net.append_layer(BatchNorm2d(num_kernels))

        net.append_layer(ReLU())

        fcl_1_input_size = np.prod(cl_1_output_shape)

        net.append_layer(Flatten())

        fcl_1 = Linear(
            fcl_1_input_size, int(fcl_1_input_size/2.))
        net.append_layer(fcl_1)

        if dropout:
            net.append_layer(Dropout(0.3))

        net.append_layer(ReLU())

        fcl_2 = Linear(
            int(fcl_1_input_size / 2), int(fcl_1_input_size / 3))
        net.append_layer(fcl_2)

        net.append_layer(ReLU())

        fcl_3 = Linear(int(fcl_1_input_size / 3), categories)
        net.append_layer(fcl_3)

        net.append_layer(SoftMax())

        net.train(iterations)
        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the UCI ML hand-written digits dataset using {} we achieve an accuracy of: {}%'.format(
                description, accuracy * 100.), file=f)
        print('\nOn the UCI ML hand-written digits dataset using {} we achieve an accuracy of: {}%'.format(description, accuracy * 100.))
        self.assertGreater(accuracy, 0.3)





if __name__ == "__main__":

    import sys
    if sys.argv[-1] == "Bonus":
        # sys.argv.pop()
        loader = unittest.TestLoader()
        bonus_points = {}
        tests = [TestModel]
        percentages = [8, 5, 2, 45, 15, 2, 23]
        total_points = 0
        for t, p in zip(tests, percentages):
            if unittest.TextTestRunner().run(loader.loadTestsFromTestCase(t)).wasSuccessful():
                bonus_points.update({t.__name__: ["OK", p]})
                total_points += p
            else:
                bonus_points.update({t.__name__: ["FAIL", p]})

        import time
        time.sleep(1)
        print("=========================== Statistics ===============================")
        exam_percentage = 3
        table = []
        for i, (k, (outcome, p)) in enumerate(bonus_points.items()):
            table.append([i, k, outcome, "0 / {} (%)".format(p) if outcome == "FAIL" else "{} / {} (%)".format(p, p),
                          "{:.3f} / 10 (%)".format(p / 100 * exam_percentage)])
        table.append([])
        table.append(["Ex2", "Total Achieved", "", "{} / 100 (%)".format(total_points),
                      "{:.3f} / 10 (%)".format(total_points * exam_percentage / 100)])

        print(tabulate.tabulate(table, headers=[
              'Pos', 'Test', "Result", 'Percent in Exercise', 'Percent in Exam'], tablefmt="github"))
    else:
        unittest.main()
