from neural import *

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

xor_training_data = [([0, 0], [0]), ([0 , 1], [1]), ([1,0], [1]), ([1, 1], [1])]

xorn = NeuralNet(2,2,1)
xorn.train(xor_training_data, iters= 5000, print_interval = 100 )
print(xorn.test_with_expected(xor_training_data))

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

xor_training_data = [([0, 0], [0]), ([0 , 1], [1]), ([1,0], [1]), ([1, 1], [1])]

xorn = NeuralNet(2,8,1)
xorn.train(xor_training_data, iters= 5000, print_interval = 100 )
print(xorn.test_with_expected(xor_training_data))

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

xor_training_data = [([0, 0], [0]), ([0 , 1], [1]), ([1,0], [1]), ([1, 1], [1])]

xorn = NeuralNet(2,1,1)
xorn.train(xor_training_data, iters= 5000, print_interval = 100 )
print(xorn.test_with_expected(xor_training_data))

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

pol_training_data = [([0.9, 0.6,0.8,0.3,0.1], [1.0]), ([0.8 , 0.8, 0.4, 0.6, 0.4], [1.0]), ([0.7, 0.2, 0.4, 0.6, 0.3], [1.0]), ([0.5, 0.5, 0.8, 0.4, 0.8], [0.0]), ([0.3, 0.1, 0.6, 0.8, 0.8], [0.0]), ([0.6, 0.3, 0.4, 0.3, 0.6], [0.0])]

pol_testing_data = [([1.0, 1.0, 1.0, 0.1, 0.1], []), ([0.5, 0.2, 0.1, 0.7, 0.7], []), ([0.8,0.3,0.3,0.3,0.8], []), ([0.8,0.3,0.3,0.8,0.3], []), ([0.9,0.8,0.8,0.3,0.6], [])]

polNet = NeuralNet(5,2,1)
polNet.train(pol_training_data, iters= 5000, print_interval = 100 )
print(polNet.test_with_expected(pol_testing_data))