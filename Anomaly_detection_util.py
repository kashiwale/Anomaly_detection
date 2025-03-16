import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal as mvnorm
from matplotlib.colors import ListedColormap


class AnomalyDetector:
    def __init__(self):
        """ Initializes the Anomaly Detector class. """
        self.mu = None
        self.sigma2 = None
        self.epsilon = None
        self.F1 = None
        self.p = None

    def estimateGaussian(self, X):
        """
        Estimates the parameters (mean, variance) of a Gaussian distribution for the dataset X.
        """
        self.mu = np.mean(X, axis=0)
        self.sigma2 = np.var(X, axis=0)  # Variance instead of standard deviation squared
        return self.mu, self.sigma2

    def multivariateGaussian(self, X):
        """
        Computes the probability density function of the multivariate normal distribution.
        """
        return mvnorm.pdf(X, mean=self.mu, cov=np.diag(self.sigma2))

    def selectThreshold(self, yval, pval):
        """
        Selects the best threshold epsilon for detecting anomalies using the F1 score.
        """
        bestEpsilon = 0
        bestF1 = 0
        stepsize = (np.max(pval) - np.min(pval)) / 1000

        for epsilon in np.arange(np.min(pval), np.max(pval), stepsize):
            predictions = (pval < epsilon).astype(int)

            tp = np.sum((predictions == 1) & (yval == 1))  # True positives
            fp = np.sum((predictions == 1) & (yval == 0))  # False positives
            fn = np.sum((predictions == 0) & (yval == 1))  # False negatives

            # Prevent division by zero
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            F1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            if F1 > bestF1:
                bestF1 = F1
                bestEpsilon = epsilon

        self.epsilon = bestEpsilon
        self.F1 = bestF1
        return bestEpsilon, bestF1

    def visualizeFit(self, X):
        """
        Plots the dataset and the estimated Gaussian distribution.
        """
        levels = np.linspace(0, 0.0005, 10)
        cmap = ListedColormap(['red', 'blue', 'lightgreen', 'gray', 'cyan'])

        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))

        Z = self.multivariateGaussian(np.c_[xx1.ravel(), xx2.ravel()])
        Z = Z.reshape(xx1.shape)

        plt.figure(figsize=(10, 8))
        plt.contourf(xx1, xx2, Z, levels, cmap=cmap, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c='blue', edgecolors='k', label="Data Points")
        plt.xlabel('Latency (ms)')
        plt.ylabel('Throughput (mb/s)')
        plt.title('Gaussian Fit of the Data')
        plt.legend()
        plt.show()

    def plotAnomalies(self, X):
        """
        Plots the dataset and highlights the detected anomalies.
        """
        outliers = np.where(self.p < self.epsilon)

        plt.figure(figsize=(10, 8))
        plt.scatter(X[:, 0], X[:, 1], c='blue', edgecolors='k', label="Normal Data")
        plt.scatter(X[outliers][:, 0], X[outliers][:, 1], c='red', marker='o', s=100, label="Anomalies")
        plt.xlabel('Latency (ms)')
        plt.ylabel('Throughput (mb/s)')
        plt.title('Anomaly Detection')
        plt.legend()
        plt.show()

    def detect_anomalies(self, X, Xval, yval):
        """
        Runs the full anomaly detection process.
        """
        print('🔹 Estimating Gaussian parameters...')
        self.estimateGaussian(X)

        print('🔹 Computing probability densities...')
        self.p = self.multivariateGaussian(X)

        print('🔹 Visualizing dataset fit...')
        self.visualizeFit(X)

        print('🔹 Selecting best anomaly threshold...')
        pval = self.multivariateGaussian(Xval)
        self.selectThreshold(yval, pval)

        print(f'✅ Best Epsilon: {self.epsilon:.6e}')
        print(f'✅ Best F1 Score: {self.F1:.6f}')

        print('🔹 Plotting detected anomalies...')
        self.plotAnomalies(X)


