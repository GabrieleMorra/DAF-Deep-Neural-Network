import numpy as np

class ActivationFunctions(object): 

    def SoftPlus(self, x):
        """
        SoftPlus Activation Function
        """
        return np.log1p(np.exp(x))
    
    def Derivative_SoftPlus(self, dA, x):
        """
        Compute the derivative of the SoftPlus activation function.
        """
        return dA * 1 / (1 + np.exp(x))

    def ReLU(self, x):
        """
        Rectified Linear Unit (ReLU) Activation Function
        """
        return np.where(x > 0, x, 0)
    
    def Derivative_ReLU(self, dA, x):
        """
        Compute the derivative of the Rectified Linear Unit (ReLU) activation function.
        """
        return np.where(x > 0, dA, 0)
    
    def Leaky_ReLU(self, x, alpha = 0.01):
        """
        Leaky Rectified Linear Unit (Leaky ReLU) Activation Function
        """
        return np.where(x > 0, x, alpha * x)
    
    def Derivative_Leaky_ReLU(self, dA, x, alpha = 0.01):
        """
        Compute the derivative of the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
        """
        return np.where(x > 0, dA, alpha * dA)

    def Sigmoid(self, x):
        """
        Sigmoid Activation Function
        """
        return 1 / (1 + np.exp(-x)) 

    def Derivative_Sigmoid(self, dA, x):
        """ 
        Compute the derivative of the Sigmoid activation function.  
        """
        sig = 1 / (1 + np.exp(-x))
        return dA * sig * (1 - sig)

    def Tanh(self, x):
        """
        Hyperbolic Tangent Activation Function
        """
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def Derivative_Tanh(self, dA, x):
        """
        Compute the derivative of the Hyperbolic Tangent activation function.
        """
        return dA * ( (-np.exp(x) + np.exp(-x)) / (np.exp(x) + np.exp(-x))**2 +1)

    def Softmax(self, x):
        """
        Softmax Activation Function
        """
        return np.exp(x)/np.sum(np.exp(x))

    def Derivative_Softmax(self, dA, x):
        """
        Compute the derivative of the Softmax activation function.
        """
        return dA * np.exp(x)/np.sum(np.exp(x)) * (1 - np.exp(x)/np.sum(np.exp(x)))
    
    def Elu(self, x, alpha = 1):
        """
        Exponential Linear Unit (ELU) Activation Function
        """
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def Derivative_Elu(self, dA, x, alpha = 1):
        """
        Compute the derivative of the Exponential Linear Unit (ELU) activation function.
        """
        return dA * np.where(x > 0, 1, alpha * np.exp(x))
