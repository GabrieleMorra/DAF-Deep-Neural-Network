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
        exp_neg_x = np.exp(-x)
        sig = 1 / (1 + exp_neg_x)
        return dA * sig * (1 - sig)

    def Tanh(self, x):
        """
        Hyperbolic Tangent Activation Function
        """
        exp_x = np.exp(x)
        exp_neg_x = np.exp(-x)
        return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
    
    def Derivative_Tanh(self, dA, x):
        """
        Compute the derivative of the Hyperbolic Tangent activation function.
        """
        exp_x = np.exp(x)
        exp_neg_x = np.exp(-x)
        denominator = exp_x + exp_neg_x
        return dA * ((-exp_x + exp_neg_x) / (denominator**2) + 1)

    def Softmax(self, x):
        """
        Softmax Activation Function
        """
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def Derivative_Softmax(self, dA, x):
        """
        Compute the derivative of the Softmax activation function.
        """
        exp_x = np.exp(x)
        sum_exp_x = np.sum(exp_x)
        softmax_val = exp_x / sum_exp_x
        return dA * softmax_val * (1 - softmax_val)
    
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