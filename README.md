# Weibull Estimator

## Description
This project provides a Python implementation of the `WeibullEstimator` class, designed to fit Weibull distributions to datasets using various statistical methods. It is particularly tailored for determining the parameters of a diametric distribution in the forestry sector. The methods implemented in this class are based on research, including "Evaluation of Methods to Predict Weibull Parameters for Characterizing Diameter Distributions" (DOI: 10.5849/forsci.12-001) and "Simplified Method-of-Moments Estimation for the Weibull Distribution" by Oscar Garcia, 1981. 

## Features
- Parameter estimation of the Weibull distribution using numerical and empirical methods.
- Fit evaluation using statistical tests such as Kolmogorov-Smirnov and Anderson-Darling.
- Visualization of fitted distributions and comparison among methods.

## Installation

```bash
git clone https://github.com/pmenaq-new/fitweibull.git
cd weibull-estimator
pip install -r requirements.txt
```

## Usage

### Importing the Class
```python
from weibull import WeibullEstimator
```

### Create Instance and Fit Models
```python
import numpy as np

# Generate sample data
data = np.random.weibull(5, 1000) * 3 + 2
weibull_estimator = WeibullEstimator(data)
weibull_estimator.fit_all_methods()
```

### Visualize Results
```python
weibull_estimator.plot_distributions()
```

### Get the Best Model
```python
best_model = weibull_estimator.get_best_model()
print(best_model)
```

## Examples
In the `examples` folder, you will find detailed scripts demonstrating practical use of `WeibullEstimator` in various scenarios.

## Contributions
Contributions are welcome! Please read `CONTRIBUTING.md` for more details on how to contribute to the project.

## License
This project is licensed under the MIT License - see the `LICENSE.md` file for details.