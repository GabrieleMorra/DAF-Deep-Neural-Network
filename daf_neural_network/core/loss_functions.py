import numpy as np

def get_mean_loss(Y_hat, Y, nn):
    Y = np.array(Y).astype(float)
    Y_hat = np.array(Y_hat).astype(float)

    LossFun = nn["NeuralNetworkModel"]["loss"]
    GLV = GetLossValue()

    if LossFun == "MSE":
        return GLV.squared_error(Y, Y_hat)
    elif LossFun == "LC":
        return GLV.log_cosh_error(Y, Y_hat)
    elif LossFun == "RMSE":
        return GLV.root_squared_error(Y, Y_hat)
    elif LossFun == "MSLE":
        return GLV.squared_logarithmic_error(Y, Y_hat)
    elif LossFun == "MAPE":
        return GLV.mean_absolute_percentage_error(Y, Y_hat)
    elif LossFun == "SSE":
        return GLV.sum_squared_error(Y, Y_hat)
    elif LossFun == "custom_loss":
        return GLV.custom_loss(Y, Y_hat)
    else:
        raise Exception(f" Non-supported loss function: {LossFun} ")
    
def get_loss_derivative(Y_hat, Y, nn):
    Y = np.array(Y).astype(float)
    Y_hat = np.array(Y_hat).astype(float)

    LossFun =  nn["NeuralNetworkModel"]["loss"]
    GLD = GetLossDerivative()

    if LossFun == "MSE":
        return GLD.squared_error_derivative(Y, Y_hat)
    elif LossFun == "LC":
        return GLD.log_cosh_error_derivative(Y, Y_hat)
    elif LossFun == "RMSE":
        return GLD.root_squared_error_derivative(Y, Y_hat)
    elif LossFun == "MSLE":
        return GLD.squared_logarithmic_error_derivative(Y, Y_hat)
    elif LossFun == "MAPE":
        return GLD.mean_absolute_percentage_error_derivative(Y, Y_hat)
    elif LossFun == "SSE":
        return GLD.sum_squared_error_derivative(Y, Y_hat)
    elif LossFun == "custom_loss":
        return GLD.custom_loss_derivative(Y, Y_hat)
    else:
        raise Exception(f" Non-supported loss function: {LossFun} ")


class GetLossValue(object): 

    def squared_error(self, Y, Y_hat):
        return np.mean(0.5 * (Y - Y_hat)**2)

    def log_cosh_error(self, Y, Y_hat):
        return np.mean(np.log(np.cosh(Y - Y_hat)))

    def root_squared_error(self, Y, Y_hat):
        return np.mean(np.sqrt((Y - Y_hat)**2))

    def squared_logarithmic_error(self, Y, Y_hat):
        return np.mean(0.5 * (np.log1p(Y) - np.log1p(Y_hat))**2)

    def mean_absolute_percentage_error(self, Y, Y_hat):
        return np.mean(np.abs(np.where(Y!=0, (Y - Y_hat) / Y, 0)))
    
    def sum_squared_error(self, Y, Y_hat):
        return np.mean(0.5 * np.sum((Y - Y_hat)**2))
    
    def custom_loss(self, Y, Y_hat):
        diff = np.abs((Y - Y_hat) / (Y + 1))
        loss = np.where(diff > 0.03, 10*np.mean(np.abs(np.where(Y!=0, (Y - Y_hat) / Y, 0))), np.mean(np.abs(np.where(Y!=0, (Y - Y_hat) / Y, 0))))
        return np.mean(loss)
    
    
class GetLossDerivative(object): 

    def squared_error_derivative(self, Y, Y_hat):
        return - (Y - Y_hat)
    
    def log_cosh_error_derivative(self, Y, Y_hat):
        return - np.tanh(Y - Y_hat)
    
    def root_squared_error_derivative(self, Y, Y_hat):
        return np.where(Y!=Y_hat, - (Y - Y_hat) / np.sqrt((Y - Y_hat)**2), 0)
    
    def squared_logarithmic_error_derivative(self, Y, Y_hat):
        return - (np.log1p(Y) - np.log1p(Y_hat) ) / (Y_hat + 1)
    
    def mean_absolute_percentage_error_derivative(self, Y, Y_hat):
        return - np.sign(np.where(Y!=0, (Y - Y_hat) / Y, 0))
    
    def sum_squared_error_derivative(self, Y, Y_hat):
        return - (Y - Y_hat)
    
    def custom_loss_derivative(self, Y, Y_hat):
        diff = np.abs((Y - Y_hat) / Y)
        return np.where(diff > 0.03, - 10 * np.sign(np.where(Y!=0, (Y - Y_hat) / Y, 0)), - np.sign(np.where(Y!=0, (Y - Y_hat) / Y, 0)))