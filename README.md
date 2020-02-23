# M185-Okapi-Project

regression.py:
  regression_line = PolynomialRegression() -- creates an object that saves the model
  
  regression_line.fit_GD(training_data.X, training_data.Y, eta=None, eps=0, tmax=10000, verbose=False) 
    
    creates numpy array of coefficients using linear regression
        
        --assigns coefficients to regression_line.coef_
        
        --uses gradient descent
    eta -- step size
    eps -- min err diff between subsequent iterations before the program terminates early
    tmax -- max number of iterations
    verbose -- for debugging
    returns (self, number of iterations)
    
  regression_line.fit(training_data.X, training_data.Y) -- linear regression using closed form solution - may be much slower, but is guaranteed to return optimal solution
    
    returns self
  
  regression_line.PCA(training_data.X) -- returns axes associated with 1, 2, . . ., n-1, PCA components
    
    first axis in first column, and so on
    untested lol
