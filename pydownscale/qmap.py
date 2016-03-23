import numpy

class QMap():
    def __init__(self, step=0.01):
        self.step = step

    def fit(self, x, y):
        steps = numpy.arange(0, 100, self.step)
        self.x_map = numpy.percentile(x, steps)
        self.y_map = numpy.percentile(y, steps)
        return self

    def predict(self, y):
        idx = [numpy.abs(val - self.y_map).argmin() for val in y]
        return self.x_map[idx]

def test_qmap():
    numpy.random.seed(0)
    x = numpy.random.normal(10, size=(10,20))
    y = numpy.random.normal(100, size=(10, 20))
    mapped = numpy.zeros(x.shape)
    for j in range(x.shape[1]):
        qmap = QMap()
        qmap.fit(x[:,j], y[:,j])
        mapped[:, j] = qmap.predict(y[:,j])

test_qmap()
