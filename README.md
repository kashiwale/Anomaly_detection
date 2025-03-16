# 🚀 Anomaly Detection using Gaussian Distribution (OOP Implementation)
This repository contains an **Object-Oriented Programming (OOP) implementation** of **Anomaly Detection** using **Gaussian Distribution**.

The implementation:
- **Estimates Gaussian parameters (mean & variance)** for a given dataset.
- **Computes probability densities** using a **multivariate normal distribution**.
- **Detects anomalies (outliers)** using **cross-validation & F1-score optimization**.
- **Visualizes results**, highlighting **normal data vs. anomalies**.

---

## 📌 **Features**
✅ **Encapsulated in a `AnomalyDetector` class** for modularity.  
✅ **Robust anomaly detection with Gaussian probability densities**.  
✅ **Auto-adjusting `epsilon` threshold using F1-score**.  
✅ **Prevents division by zero for reliable calculations**.  
✅ **Detailed visualizations of dataset & detected anomalies**.  

---

## 📦 **Installation**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/Anomaly-Detection.git
cd Anomaly-Detection
2️⃣ Install Dependencies
Ensure you have Python 3.8+ and install the required libraries:

bash
Copy
pip install numpy matplotlib scipy
🚀 Usage Example
1️⃣ Running Anomaly Detection
python
Copy
import scipy.io as sio
from anomaly_detection import AnomalyDetector

# Load dataset
data = sio.loadmat('test_data.mat')  # Ensure file is in the working directory
X = data['X']
Xval = data['Xval']
yval = data['yval'].flatten()

# Initialize and run anomaly detection
detector = AnomalyDetector()
detector.detect_anomalies(X, Xval, yval)
2️⃣ Expected Output
✅ Best Epsilon (Threshold): ~8.99e-05
✅ Best F1 Score: ~0.875
✅ Plots:

Gaussian Fit - Contour visualization of normal distribution.
Anomaly Detection - Outliers (in red) vs. normal data points (in blue).
🛠️ Class Methods
Method	Description
__init__()	Initializes the detector.
estimateGaussian(X)	Computes mean & variance of the dataset.
multivariateGaussian(X)	Computes Gaussian probability density.
selectThreshold(yval, pval)	Finds best epsilon using F1-score optimization.
visualizeFit(X)	Plots dataset fit using Gaussian distribution.
plotAnomalies(X)	Highlights outliers in red.
detect_anomalies(X, Xval, yval)	Runs full anomaly detection workflow.
📊 Anomaly Visualization
✅ Gaussian Fit (Normal Data)

✅ Anomalies (Outliers in Red)

📌 Note: Replace the above links with actual images after running the script.

📝 Dataset Information
ex8data1.mat: Small dataset with 2 features (Latency & Throughput).
ex8data2.mat: Larger dataset with multiple features.
Both datasets contain normal examples and some anomalies (outliers).

💡 Future Improvements
🔹 Implement K-Means clustering for anomaly detection.
🔹 Add Deep Learning (Autoencoders) for complex anomaly detection.
🔹 Extend support for real-world datasets (network traffic, fraud detection, etc.).

📜 License
This project is licensed under the MIT License.

👨‍💻 Author
Developed by Dhirendra Kashiwale 🚀
📧 Email: kashiwale@yahoo.com
🔗 LinkedIn: https://www.linkedin.com/in/kashiwale
🔗 GitHub: kashiwale
