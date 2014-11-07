from random import *


class RandomVariable(object):
    def __init__(self):
        self.lower = None
        self.upper = None
        self.delta = None
        self.cdf = []


class DiscreteRandomVariable(RandomVariable):
    def __init__(self, array, n_classes=2):
        """
        takes a list of positive ints and turns it into
        a discrete distribution
        i.e. array([0,0,1,1,2]) => [0.4, 0.4, 0.2]
        """
        super(self.__class__, self).__init__()
        counts = [0] * n_classes
        for element in array:
            counts[element] += 1
        self.distribution = [float(count)/len(array) for count in counts]


def average_distributions(distributions):
    result = []
    for p_values in zip(*distributions):
        p = float(sum(p_values)) / len(distributions)
        result.append(p)
    return result


class ContinuousRandomVariable(RandomVariable):
    def __init__(self, array, n_parts):
        super(self.__class__, self).__init__()
        # Compute Bounds
        self.lower = min(array)
        self.upper = max(array)
        self.delta = float(self.upper - self.lower) / (n_parts-1)
        if (self.delta == 0):
            self.delta = 0.001
        # Compute Counts
        self.distribution = [0 for _ in range(n_parts)]
        for value in array:
            index = int(float(value-self.lower)/self.delta)
            self.distribution[index] += 1
        # Compute CDF
        total_count = 0
        for i,count in enumerate(self.distribution):
            value = i*self.delta+self.lower
            total_count += count
            p = float(total_count) / len(array)
            self.cdf.append(p)

    def sample(self):
        r = random()
        rand = (random())*self.delta
        for i in range(len(self.cdf)-1):
            value = self.cdf[i]
            if value >= r:
                lower = i*self.delta + self.lower
                return lower + rand
        return self.upper - rand


def rv_test():
    a = [0.0,0.01,0.02,0.1,0.2,0.3,0.5,0.7,0.9,1.0]
    A = ContinuousRandomVariable(a,100)
    print(A.cdf)
    N = 1000000
    print(A.lower, A.upper, A.delta)
    print(sum([A.sample() for _ in range(N)]) / N)
    b = [0,0,1,1,2]
    B = DiscreteRandomVariable(b)

if __name__ == '__main__':
    rv_test()
