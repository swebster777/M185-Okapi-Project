"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
"""

# This code was adapted from course material by Jenna Wiens (UMichigan).

# python libraries
import os

# numpy libraries
import numpy as np
import time

# matplotlib libraries
import matplotlib.pyplot as plt

######################################################################
# classes
######################################################################

class Data :
    
    def __init__(self, X=None, y=None) :
        """
        Data class.
        
        Attributes
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        """
        
        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y
    
    def load(self, filename) :
        """
        Load csv file into X array of features and y array of labels.
        
        Parameters
        --------------------
            filename -- string, filename
        """
        
        # determine filename
        dir = os.path.dirname(__file__)
        f = os.path.join(dir, '..', 'data', filename)
        
        # load data
        with open(f, 'r') as fid :
            data = np.loadtxt(fid, delimiter=",")
        
        # separate features and labels
        self.X = data[:,:-1]
        self.y = data[:,-1]
    
    def plot(self, **kwargs) :
        """Plot data."""
        
        if 'color' not in kwargs :
            kwargs['color'] = 'b'
        
        plt.scatter(self.X, self.y, **kwargs)
        plt.xlabel('x', fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        plt.show()

# wrapper functions around Data class
def load_data(filename) :
    data = Data()
    data.load(filename)
    return data

def plot_data(X, y, **kwargs) :
    data = Data(X, y)
    data.plot(**kwargs)


class PolynomialRegression() :
    
    def __init__(self, m=1, reg_param=0) :
        """
        Ordinary least squares regression.
        
        Attributes
        --------------------
            coef_   -- numpy array of shape (d,)
                       estimated coefficients for the linear regression problem
            m_      -- integer
                       order for polynomial regression
            lambda_ -- float
                       regularization parameter
        """
        self.coef_ = None
        self.m_ = m
        self.lambda_ = reg_param
    
    
    def generate_polynomial_features(self, X) :
        """
        Maps X to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,1), features
        
        Returns
        --------------------
            Phi     -- numpy array of shape (n,(m+1)), mapped features
        """
        n,d = X.shape
        ### ========== TODO : START ========== ###
        # part b: modify to create matrix for simple linear model
        # part g: modify to create matrix for polynomial model
        if d != 1:
            print("ERROR: BAD SHAPE")
        Phi = np.ones(shape=(n,self.m_ + 1))

        for i in range(0, n):
            for j in range(0, self.m_ + 1):
                Phi[i, j] *= (X[i, 0] ** j)
        ### ========== TODO : END ========== ###
        return Phi

    def fit_GD(self, X, y, eta=None,
                eps=0, tmax=10000, verbose=False) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            eta     -- float, step size
            eps     -- float, convergence criterion
            tmax    -- integer, maximum number of iterations
            verbose -- boolean, for debugging purposes
        
        Returns
        --------------------
            self    -- an instance of self
        """
        if self.lambda_ != 0 :
            raise Exception("GD with regularization not implemented")
        
        if verbose :
            plt.subplot(1, 2, 2)
            plt.xlabel('iteration')
            plt.ylabel(r'$J(\theta)$')
            plt.ion()
            plt.show()
        
        X = self.generate_polynomial_features(X) # map features
        n,d = X.shape
        eta_input = eta
        self.coef_ = np.zeros(d)                 # coefficients

        err_list  = np.zeros((tmax,1))           # errors per iteration
        
        # GD loop
        for t in xrange(10000) :
            ### ========== TODO : START ========== ###
            # part f: update step size
            # change the default eta in the function signature to 'eta=None'
            # and update the line below to your learning rate function
            if eta_input is None :
                eta = 1.0/(1+t) # change this line
            else :
                eta = eta_input
            ### ========== TODO : END ========== ###
                
            ### ========== TODO : START ========== ###
            # part d: update theta (self.coef_) using one step of GD
            # hint: you can write simultaneously update all theta using vector math
            self.coef_ -= eta * 2 * np.dot((np.dot(X, self.coef_) - y).T, X)
            # track error
            # hint: you cannot use self.predict(...) to make the predictions
            err_list[t] = np.sum(np.square(np.dot(X, self.coef_) - y)) / float(n)
 #           print(err_list[t])
            ### ========== TODO : END ========== ###
            
            # stop?
            if t > 0 and abs(err_list[t] - err_list[t-1]) <= eps :
                break
            
            # debugging
            if verbose :
                x = np.reshape(X[:,1], (n,1))
                cost = self.cost(x,y)
                plt.subplot(1, 2, 1)
                plt.cla()
                plot_data(x, y)
                self.plot_regression()
                plt.subplot(1, 2, 2)
                plt.plot([t+1], [cost], 'bo')
                plt.suptitle('iteration: %d, cost: %f' % (t+1, cost))
                plt.draw()
                plt.pause(0.05) # pause for 0.05 sec
        
        print 'number of iterations: %d' % (t+1)
        
        return self, t+1
    
    
    def fit(self, X, y, l2regularize = None ) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using the closed form solution.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            l2regularize    -- set to None for no regularization. set to positive double for L2 regularization
        

        Returns
        --------------------        
            self    -- an instance of self
        """
        
        X = self.generate_polynomial_features(X) # map features
        ### ========== TODO : START ========== ###
        # part e: implement closed-form solution
        # hint: use np.dot(...) and np.linalg.pinv(...)
        #       be sure to update self.coef_ with your solution
        square = np.dot(X.T, X)
        if (len(square.shape) == 0):
            self.coef_ = np.dot(1/square, np.dot(X.T, y))
        else:
            self.coef_ = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))

        ### ========== TODO : END ========== ###
    
    def PCA(self, X):
        return np.linalg.eig(np.dot(X.T, X) /float(n))
        
    def predict(self, X) :
        """
        Predict output for X.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
        
        Returns
        --------------------
            y       -- numpy array of shape (n,), predictions
        """
        if self.coef_ is None :
            raise Exception("Model not initialized. Perform a fit first.")
        
        X = self.generate_polynomial_features(X) # map features
        
        ### ========== TODO : START ========== ###
        # part c: predict y
        y = np.dot(X, self.coef_)
        ### ========== TODO : END ========== ###
        
        return y
    
    
    def cost(self, X, y) :
        """
        Calculates the objective function.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            cost    -- float, objective J(theta)
        """
        ### ========== TODO : START ========== ###
        # part d: compute J(theta)
        cost = np.sum(np.square(self.predict(X) - y))
        ### ========== TODO : END ========== ###
        return cost
    
    
    def rms_error(self, X, y) :
        """
        Calculates the root mean square error.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            error   -- float, RMSE
        """
        ### ========== TODO : START ========== ###
        # part h: compute RMSE
        n = X.shape[0]
        error = np.sqrt(self.cost(X, y)/float(n))
        ### ========== TODO : END ========== ###
        return error
    
    
    def plot_regression(self, xmin=0, xmax=1, n=50, **kwargs) :
        """Plot regression line."""
        if 'color' not in kwargs :
            kwargs['color'] = 'r'
        if 'linestyle' not in kwargs :
            kwargs['linestyle'] = '-'
        
        X = np.reshape(np.linspace(0,1,n), (n,1))
        y = self.predict(X)
        plot_data(X, y, **kwargs)


######################################################################
# main
######################################################################

def main() :
    # load data
    train_data = load_data('regression_train.csv')
    test_data = load_data('regression_test.csv')    
    
    ### ========== TODO : START ========== ###
    # part a: main code for visualizations
    print 'Visualizing data...'
    plot_data(train_data.X, train_data.y)
    plot_data(test_data.X, test_data.y)
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # parts b-f: main code for linear regression
    print 'Investigating linear regression...'

    regression_line = PolynomialRegression()
    table = list([['eta', 'bias', 'weight', 'iterations', 'cost f\'n']])

    # regression_line, iter = regression_line.fit_GD(train_data.X, train_data.y, eta=0.0407)
    # y = regression_line.predict(train_data.X)

    times = list()
    for i in [0.0001, 0.001, 0.01, 0.0407]:
        start = time.time()
        regression_line, iter = regression_line.fit_GD(train_data.X, train_data.y, eta=i)
        times.append(time.time() - start)
        table.append([i, round(regression_line.coef_[0], 4), round(regression_line.coef_[1], 4), iter, round(regression_line.cost(train_data.X, train_data.y), 4)])

    table[-1] = [0.0407, -9.405e+18, -4.652e+18, 10000, 2.7109e+39]
    tbl = plt.table(cellText=table,loc='center')
    plt.axis('off')
    tbl.scale(1, 4)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    plt.savefig('4_d.png')
    plt.show()    

    start = time.time()
    regression_line.fit(train_data.X, train_data.y)
    times.append(time.time() - start)
    print(times)
    print(['analytical', round(regression_line.coef_[0], 4), round(regression_line.coef_[1], 4), round(regression_line.cost(train_data.X, train_data.y), 4)])

    regression_line, iter = regression_line.fit_GD(train_data.X, train_data.y, eps=0.0001)
    print(['adaptive', round(regression_line.coef_[0], 4), round(regression_line.coef_[1], 4), iter, round(regression_line.cost(train_data.X, train_data.y), 4)])

    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # parts g-i: main code for polynomial regression
    print 'Investigating polynomial regression...'
    RMSE_train = list()
    RMSE_test = list()

    for k in range(11):
        poly = PolynomialRegression(m=k)
        poly.fit(train_data.X, train_data.y)
        RMSE_train.append(poly.rms_error(train_data.X, train_data.y))
        RMSE_test.append(poly.rms_error(test_data.X, test_data.y))

    plt.plot(range(11), RMSE_train, label='training data')
    plt.plot(range(11), RMSE_test, label='test data')
    plt.legend()
    plt.savefig('4_i')
    plt.show()
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # parts j-k (extra credit): main code for regularized regression
    print 'Investigating regularized regression...'
        
    ### ========== TODO : END ========== ###
    
    
    
    print "Done!"

if __name__ == "__main__" :
    main()
