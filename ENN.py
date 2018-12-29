import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(17)
np.random.seed(117)

def readData1():
    datafile1 = open('data1.txt')
    datalines1 = datafile1.read().split('\n') # reading lines
    dataio1 = [[0 for x in range(2)]for y in range(32)]
    for i in range(1, 33): # seperationg into inputs and outputs
        dataio1[i-1] = datalines1[i].split(' ')
    datainputs1 = [[0 for x in range(5)]for y in range(32)]
    for i in range(32):
        for j in range(5): # taking inputs
            datainputs1[i][j] = int(dataio1[i][0][j])
    dataoutputs1 = [[0 for x in range(1)] for y in range(32)]
    for i in range(32): # taking outputs
        dataoutputs1[i][0] = int(dataio1[i][1])
    datafile1.close()

    return np.array(datainputs1), np.array(dataoutputs1)

def readData2():
    datafile2 = open('data2.txt')
    datalines2 = datafile2.read().split('\n') # reading lines
    dataio2 = [[0 for x in range(2)]for y in range(64)]
    for i in range(1, 65): # seperationg into inputs and outputs
        dataio2[i-1] = datalines2[i].split(' ')
    datainputs2 = [[0 for x in range(6)]for y in range(64)]
    for i in range(64):
        for j in range(6): # taking inputs
            datainputs2[i][j] = int(dataio2[i][0][j])
    dataoutputs2 = [[0 for x in range(1)] for y in range(64)]
    for i in range(64): # taking outputs
        dataoutputs2[i][0] = int(dataio2[i][1])
    datafile2.close()

    return np.array(datainputs2), np.array(dataoutputs2)

def readData3():
    datafile3 = open('data3.txt')
    datalines3 = datafile3.read().split('\n') # reading lines
    dataio3 = [[0 for x in range(7)]for y in range(2000)]
    for i in range(1, 2001): # seperationg into inputs and outputs
        dataio3[i-1] = datalines3[i].split(' ')
    datainputs3 = [[0 for x in range(6)]for y in range(2000)]
    for i in range(2000):
        for j in range(6): # taking inputs
            datainputs3[i][j] = float(dataio3[i][j])
    dataoutputs3 = [[0 for x in range(1)] for y in range(2000)]
    for i in range(2000): # taking outputs
        dataoutputs3[i][0] = float(dataio3[i][6])
    datafile3.close()

    return np.array(datainputs3), np.array(dataoutputs3)

def sig(x): #sigmoid function
    return 1.0 / (1 + np.exp(-x))

def sig_d(x): # sigmoid derivative function
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, i, o, n = 8, l = 0.3):
        self.hiddenLayer    = n # number of neurons in the hidden layer
        self.input          = i
        self.biases         = np.ones((self.input.shape[0], 1)) # setting up biases
        self.input          = np.concatenate((self.input, self.biases), axis = 1)
        self.weights1       = np.random.rand(self.input.shape[1], self.hiddenLayer) # input weights
        self.weights2       = np.random.rand(self.hiddenLayer, 1) # layer 1 weights
        self.expected       = o
        self.learningRate   = l
        self.output         = np.zeros(self.expected.shape)
        self.error          = np.array([])

    def feedForward(self):
        self.layer1         = sig(np.dot(self.input, self.weights1))
        self.output         = sig(np.dot(self.layer1, self.weights2))
        self.error          = self.expected - self.output

    def backProp(self): #for two layer network
        # back propagation of error using chain rule
        loss           = 2*(self.error) * sig_d(self.output)
        delta_w2       = np.multiply(self.learningRate, np.dot(self.layer1.T, loss))
        loss           = np.dot(loss, self.weights2.T) * sig_d(self.layer1)
        delta_w1       = np.multiply(self.learningRate, np.dot(self.input.T, loss))

        self.weights1 += delta_w1
        self.weights2 += delta_w2

    def assignWeights(self, w1, w2):
        self.weights1 = w1
        self.weights2 = w2

    def setInput(self, i):
        self.input = np.concatenate((i, np.ones((i.shape[0], 1))), axis = 1)
    
    def setOutput(self, o):
        self.expected = o


class GeneticAlgorithm:
    def __init__(self, c, f, cx = 0.2, mx = 0.07, d = 1):
        self.chromosomes    = c
        self.population     = self.chromosomes.shape[0] # population size
        self.fitness        = f
        self.crossoverX      = cx
        self.mutationX       = mx

    def setChromosomes(self, c):
        self.chromosomes = c

    def setFitness(self, f):
        self.fitness = f

    def rouletteWheel(self):
        temp = self.chromosomes[:]
        for i in range(self.population): # selection for population size
            rouletteWheel = random.uniform(0, np.sum(self.fitness)) # roulette wheel
            cumulativeFitness = 0
            j = 0
            while cumulativeFitness < rouletteWheel: # checking the one selected
                cumulativeFitness += self.fitness[j]
                self.chromosomes[i] = temp[j]
                j += 1

    def crossover(self):
        crossoverPosition = 0
        parent1 = np.array([])
        parent2 = np.array([])
        for i in range(0, int(self.population), 2):
            if random.random() < self.crossoverX:
                crossoverPosition = random.randint(0, self.chromosomes.shape[1])
                parent1 = self.chromosomes[i]
                parent2 = self.chromosomes[i+1]
                self.chromosomes[i] = np.append(parent1[:crossoverPosition], parent2[crossoverPosition:]) # offspring 1
                self.chromosomes[i+1] = np.append(parent2[:crossoverPosition], parent1[crossoverPosition:]) # offsprinf 2

    def mutate(self):
        chromosomeLength = self.chromosomes.shape[1]
        for i in range(self.population):
            if random.random() < self.mutationX:
                self.chromosomes[i][int(random.uniform(0,chromosomeLength))] = random.uniform(-25, 25) # change a random gene


if __name__ == "__main__":

    # reading data
    inputs, outputs = readData3()
    validationInputs = inputs[1200:]
    validationOutputs = outputs[1200:]
    inputs = inputs[:1200]
    outputs = outputs[:1200]
    batch = 1
    batchSize = 4
    finalBatch = 20

    # Initializing neural network
    hiddenLayerSize     = 11
    learningRate        = 0.2 # required only for neural net with backpropafation for testing
    nn                  = NeuralNetwork(inputs[:batchSize*batch], outputs[:batchSize*batch], hiddenLayerSize, learningRate)

    #initializing genetic algorithm
    populationSize  = 100
    maxGeneration   = 5000
    generation = 0
    crossoverProbability = 0.0
    mutationProbability = 0.2
    targetError = 0.05
    chromosomeArray = np.random.uniform(-25, 25, (populationSize, nn.input.shape[1] * nn.hiddenLayer + nn.hiddenLayer))
    fitnessArray    = np.zeros(populationSize)
    ga              = GeneticAlgorithm(chromosomeArray, fitnessArray, crossoverProbability, mutationProbability)

    mean_error_list = []
    max_error_list  = []
    min_error_list  = []


    while (generation < maxGeneration and batch <= finalBatch):
        error_list = np.zeros(populationSize)
        for j in range(populationSize):
            tempWeights1 = chromosomeArray[j][:nn.input.shape[1]*nn.hiddenLayer].reshape(nn.input.shape[1], nn.hiddenLayer)
            tempWeights2 = chromosomeArray[j][nn.input.shape[1]*nn.hiddenLayer:].reshape(nn.hiddenLayer, 1)
            nn.assignWeights(tempWeights1, tempWeights2)
            nn.feedForward()
            chromosomeArray[j] = np.append(nn.weights1.flatten(), nn.weights2.flatten())
            error_list[j] = np.sqrt(np.average(np.square(nn.error)))
        ga.setChromosomes(chromosomeArray)
        fitnessArray = np.reciprocal(error_list)
        ga.setFitness(fitnessArray)
        ga.rouletteWheel()
        ga.crossover()
        ga.mutate()
        chromosomeArray = ga.chromosomes[:]
            
        mean_error_list += [np.average(error_list)]
        max_error_list += [np.max(error_list)]
        min_error_list += [np.min(error_list)]

        # neuralnet with backpropagation for testing
        # nn.feedForward()    
        # nn.backProp()

        # mean_error_list += [np.sqrt(np.average(np.square(nn.error)))]
        # max_error_list  += [np.max(np.abs(nn.error))]
        # min_error_list  += [np.min(np.abs(nn.error))]

        generation += 1

        if (mean_error_list[-1] < targetError):
            print('Training batch ', batch, '...')
            batch += 1
            if (batchSize*batch > 64):
                nn.setInput(inputs[batchSize*batch-64:batchSize*batch])
                nn.setOutput(outputs[batchSize*batch-64:batchSize*batch])
            else:
                nn.setInput(inputs[:batchSize*batch])
                nn.setOutput(outputs[:batchSize*batch])

    print('')

    # taking the values with best fitness
    tempWeights1 = chromosomeArray[np.argmax(fitnessArray)][:nn.input.shape[1]*nn.hiddenLayer].reshape(nn.input.shape[1], nn.hiddenLayer)
    tempWeights2 = chromosomeArray[np.argmax(fitnessArray)][nn.input.shape[1]*nn.hiddenLayer:].reshape(nn.hiddenLayer, 1)
    nn.assignWeights(tempWeights1, tempWeights2)

    # VALIDATION

    # nn.setInput(inputs) # for data set 1 and 2
    # nn.setOutput(outputs) # for data set 1 and 2
    nn.setInput(validationInputs) # for data set 3
    nn.setOutput(validationOutputs) # for data set 3
    nn.feedForward()

    print('VALIDATION')
    print('Root Mean Squared Error:\t', np.round(np.sqrt(np.average(np.square(nn.error))), 2))
    print('Number of hits:\t\t\t', nn.error.shape[0] - np.count_nonzero(np.round(nn.error)))
    print('Number of misses:\t\t', np.count_nonzero(np.round(nn.error)))
    print('Percentage of hits:\t\t', np.round(((nn.error.shape[0] - np.count_nonzero(np.round(nn.error)))/nn.error.shape[0])*100, 2))

    # Plotting error graph

    # plt.plot(range(generation), max_error_list, label = 'Max Error')
    plt.plot(range(generation), mean_error_list, 'red', label = 'Mean Error')
    plt.plot(range(generation), min_error_list, 'green', label = 'Min Error')

    plt.xlabel('Generation')
    plt.ylabel('Mean Squared Error')
    plt.title('Error Values Over Generations')
    plt.xlim(left = 0, right = generation)
    plt.ylim(bottom = 0)
    plt.legend()
    plt.show()