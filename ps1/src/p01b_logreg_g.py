import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    logreg = LogisticRegression()
    logreg.fit(x_train,y_train)
    print(logreg.theta_0)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to pred_path
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    # global x_train,y_train
    # global theta_G
    # m = y_train.size

    def sigmoidFunction(z):
        return 1/(1+math.exp(z))

    def lossFunction(theta,x_train,y_train):
        m = y_train.size
        sum = 0
        for i in range(m):
            sum += y_train[i]*math.log(sigmoidFunction(theta.dot(x_train[i]))) + (1-y_train[i])*(math.log(1-sigmoidFunction(theta.dot(x_train[i]))))

        return sum*(-1/m)

    def kthGradient(k,theta,x_train,y_train):
        m = y_train.size
        sum = 0
        for i in range(m):
            sum += (y_train[i]-sigmoidFunction(theta.dot(x_train[i])))*x_train[i][k]

        return sum*(-1/m)
        
    def gradient(theta,x,y):
        gradient = np.zeros(theta.size)
        for i in range(theta.size):
            gradient[i] = kthGradient(i,theta,x,y)

        return gradient
        
    def hessianAt(self, theta,j,k,x_train,y_train):
        sum = 0
        m=y_train.size
        for i in range(m):
            sum += sigmoidFunction(theta.dot(x_train[i]))*(1-sigmoidFunction(theta.dot(x_train[i])))*x_train[i][j]*x_train[i][k]

        return  sum*(1/m)

    def hessian(self, theta,x,y):
        m = theta.shape
        print(m)
        hessian = np.zeros(m)
        for i in range(theta.size):
            for j in range(theta.size):
                hessian[i][j] = self.hessianAt(theta,i,j,x,y)
                
        return hessian   

    def norm(theta_o,theta_n):
        sum = 0
        for i in range(theta_n.size):
            sum += abs(theta_o[i] - theta_n[i])

        return sum     
                      
    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***
        self.theta = np.zeros(x[[0],:].size)
        count = 0
        alpha = self.step_size
        N = self.max_iter
        epsilon = self.eps
        theta_i = self.theta
        # grad = gradient(zero_v)
        hess = self.hessian(theta_i,x,y)
        # grad_iter = grad - np.linalg.inv(hessian(zero_v)).dot()
        temp = theta_i
        theta_i = temp - alpha*(np.linalg.inv(hessian(temp,x,y)).dot(self.gradient(temp,x,y)))
        count += 1
        while count < N and norm(temp,theta_i) >= epsilon:
            temp = theta_i
            theta_i = temp - np.linalg.inv(hessian(temp,x,y)).dot(gradient(temp,x,y))
            count += 1

        util.plot(x,y,theta_i,correction=1.0)
        self.theta = theta_i



    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        output = np.zeros(x[:,[0]].size)
        for i in range(output.size):
            probability = sigmoidFunction(self.theta_0.dot(x[i]))
            if probability > 0.5:
                output[i] = 1

        return output
        # *** END CODE HERE ***
