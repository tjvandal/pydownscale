from sklearn.linear_model import LinearRegression
import numpy

class BackwardStepwiseRegression():
    def __init__(self):
        self.linearmodel = LinearRegression(fit_intercept=True)

    def fit(self, X, y):
        variables = range(X.shape[1])
        self.linearmodel.fit(X, y)
        bestbic = self.BIC(X, y)
        for j in range(X.shape[1]):
            tempvariables = list(variables)
            tempvariables.remove(j)
            self.linearmodel.fit(X[:, tempvariables], y)
            bic = self.BIC(X[:, tempvariables], y)

            if bestbic > bic:
                variables.remove(j)
                bestbic = bic

        self.linearmodel.fit(X[:, variables], y)
        self.variables = variables

    def loglikelihood(self, X, y): # copied from statsmodels
        n2 = len(y)/2.0

        residuals = (y - self.linearmodel.predict(X))
        l = -n2 * numpy.log(2 * numpy.pi) -\
            n2 * numpy.log(1/(2*n2) * residuals.T.dot(residuals)) - n2
        return l


    def BIC(self, X, y):
        n = len(y)*1.
        df = X.shape[1]-1
        l = self.loglikelihood(X, y)
        bic = -2 * l + numpy.log(n) * df  #close enough
        return bic

    def predict(self, X):
        return self.linearmodel.predict(X[:, self.variables])

if __name__ == "__main__":
    from sklearn import datasets
    data = datasets.load_boston()
    X = data["data"]
    y = data["target"]
    model = BackwardStepwiseRegression()
    model.fit(X, y)
