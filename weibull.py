# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, newton
from scipy.special import gamma
from math import log, sqrt

class WeibullEstimator:
    
    def __init__(self, data):
        
        # pdb.set_trace()
        self.data   = data
        if len(data) < 3:
            self.params = {
                'insufficient_data': {
                    'a': 0,
                    'b': 0,
                    'c': 0,
                    'ks': 0,
                    'ad': 0,
                    'loglike': 0,
                    'rank': 0,
                    'method': 'n insuficiente'
                }
            }
            return
    
        self.d0     = np.min(data)
        self.dsd    = np.std(data)
        self.dm     = np.mean(data)
        self.dq     = np.sqrt(np.mean(data**2))
        self.d25    = np.quantile(data, 0.25)
        self.d31    = np.quantile(data, 0.31)
        self.d50    = np.quantile(data, 0.50)
        self.d63    = np.quantile(data, 0.63)
        self.d95    = np.quantile(data, 0.95)
        self.n      = data.shape[0]
        self.wa     = 0.5 * self.d0
        self.params = {}
        self.methods = ['numerical', 'garcia', 'cao02',
                        'cao03', 'cao04', 'cao05', 'cao06',
                        'cao07', 'cao08', 'cao09', 'cao10']
        self.m  = len(self.methods)
        self.fit_all_methods()

    @staticmethod
    def weibull_pdf(x, a, b, c):
        """ Función de densidad de probabilidad de Weibull """
        if a>1 or b>1 or c>1:
            return (c/b) * ((x - a)/b)**(c-1) * np.exp(-((x - a)/b)**c)
        else:
            return 0

    @staticmethod
    def weibull_cdf(x, a, b, c):
        """ Función de densidad de probabilidad acumulada de Weibull """
        if a>1.01 or b>1.01 or c>1.01:
            return 1 - np.exp(-((x - a) / b)**c)
        else:
            return 0

    @staticmethod
    def create_bins(data, bin_size):
        min_data = np.min(data)
        max_data = np.max(data)
        if np.isnan(min_data) or np.isnan(max_data) or np.isnan(bin_size):
            raise ValueError("Los datos o el tamaño del bin tienen valores NaN.")
        if bin_size == 0:
            raise ValueError("El tamaño del bin no puede ser 0.")
        bins = np.arange(min_data, max_data + bin_size, bin_size)
        return bins
    
    @staticmethod
    def get_cumulative_probs(data, bins):
        counts, _ = np.histogram(data, bins)
        cdf = np.cumsum(counts) / len(data)
        return cdf

    @staticmethod
    def fcao2(c, wa, dm, dsd):
        # pdb.set_trace()
        g1 = gamma(1+(1/c))
        g2 = gamma(1+(2/c))
        b  = (dm - wa)/g1
        return (b**2) * (g2-g1**2) - (dsd**2)
    
    @staticmethod
    def fcao3(c, wa, dq,  dsd):
        g1 = gamma(1+(1/c))
        g2 = gamma(1+(2/c))
        b  = -(wa*g1/g2) + (((wa/g2)**2)*(g1*g1-g2) + (dq*dq/g2))**0.5
        return b*b*(g2-g1*g1) - dsd*dsd
    
    @staticmethod
    def fcao8(c, wa, d95, dm):
        g1 = gamma(1+(1/c))
        b  = (d95 - wa)/(-log(1 - 0.95))**(1/c)
        return (wa + b*g1) - dm
    
    @staticmethod
    def fcao9(c, wa, d95, dq):
        g1 = gamma(1+(1/c))
        g2 = gamma(1+(2/c))
        b  = (d95 - wa)/(-log(1 - 0.95))**(1/c)
        return (b*b*g2) + (2*wa*b*g1) + (wa**2) - (dq**2)
    
    def loglike_function(self, params, data):
        a, b, c          = params
        ll = 0
        for d in data:
            ll += np.log(WeibullEstimator.weibull_pdf(d, a, b, c))
        return -ll

    def ks_test(self, a, b, c):
        """
        Compute the Kolmogorov-Smirnov statistic for a sample and a theoretical CDF.
        Parameters:
        - data: array-like, the sample data
        Returns:
        - a, b y c : parametros de weibull
        - D: float, the KS statistic
        """
        # pdb.set_trace()
        n = len(self.data)
        data_sorted = np.sort(self.data)
        empirical_cdf = np.arange(1, n + 1) / n
        theoretical_cdf = np.array([WeibullEstimator.weibull_cdf(x, a, b, c) for x in data_sorted])
        D = np.max(np.abs(empirical_cdf - theoretical_cdf))
        return D

    def ad_test(self, a, b, c):
        """Realiza una versión simplificada del test de Anderson-Darling."""
        n = len(self.data)
        data_sorted = np.sort(self.data)
        cdfi = np.array([WeibullEstimator.weibull_cdf(x, a, b, c) for x in data_sorted])
        cdfj = np.array([WeibullEstimator.weibull_cdf(x, a, b, c) for x in data_sorted[::-1]])
        ad_statistic = -n - np.sum((2*np.arange(1, n+1)-1) * (np.log(cdfi ) + np.log(1-cdfj))) / n
        return ad_statistic

    def mloglike(self, a, b, c):
        """Calcula el Root Mean Square Error."""
        try:
            mlogL = sum([log(b) - log(c) + (1 + c)*log((xi-a)/b) + ((xi-a)/b)**c  for xi in self.data])
        except:
            mlogL = np.inf
        
        return mlogL

    def fit(self, method=None):
    
        if method == 'numerical':
            initial_guess = [self.wa, 0.1, 0.1]
            result = minimize(self.loglike_function,initial_guess, method='CG', tol=1e-5, args=(self.data))
            a, b, c = result.x
        else:
            try:
                a, b, c = self.recovery_methods(method)
            except:
                a, b, c = 1, 1, 1
    
        ks_statistic = self.ks_test(a, b, c)
        ad_statistic = self.ad_test( a, b, c)
        mloglike_val = self.mloglike(a, b, c)
    
        if method not in self.params:
            self.params[method] = {}
            self.params[method].update({
                'a': a,
                'b': b,
                'c': c,
                'ks': ks_statistic,
                'ad': ad_statistic,
                'loglike': mloglike_val
            })
        
    def recovery_methods(self, method):
        # Moment-based parameter recovery
        if method ==   'cao02':
            wc     = newton(self.fcao2, 2.01, tol=1e-5, maxiter=100, args=(self.wa, self.dm, self.dsd))
            g1     = gamma(1+(1/wc))
            wb     = (self.dm - self.wa)/gamma(1+(1/wc))
            retval = [self.wa, wb, wc]
            
        elif method == 'cao03':
            wc     = newton(self.fcao3, 2.01, tol=1e-6, maxiter=100, args=(self.wa, self.dq, self.dsd))
            g1,g2  = gamma(1+(1/wc)),gamma(1+(2/wc))
            wb     = -self.wa*g1/g2+(((self.wa/g2)**2)*(g1*g1-g2) + (self.dq*self.dq/g2))**0.5
            retval = [self.wa, wb, wc]
            
        #percentile-based parameter recovery
        elif method == 'cao04':
            wc     = log(log(1 - 0.63)/log(1 - 0.31))/(log(self.d63 - self.wa) - log(self.d31-self.wa))
            wb     = (self.d63 - self.wa)/(-log(1-0.63))**(1/wc)
            retval = [self.wa, wb, wc]
            
        elif method == 'cao05':
            wc     = log(log(1 - 0.95)/log(1 - 0.50))/(log(self.d95 - self.wa) - log(self.d50-self.wa))
            wb     = (self.d50 - self.wa)/(-log(1-0.50))**(1/wc)
            retval = [self.wa, wb, wc]
            
        #hibrids methods    
        elif method == 'cao06':
            wc     = log(log(1 - 0.95)/log(1 - 0.25))/(log(self.d95 - self.wa) - log(self.d25-self.wa))
            g1,g2  = gamma(1+(1/wc)),gamma(1+(2/wc))
            wb     = ((-self.wa*g1)/g2)+((((self.wa/g2)**2)*(g1*g1-g2)+(self.dq*self.dq/g2))**0.5)
            retval = [self.wa, wb, wc]
            
        elif method == 'cao07':
            wc     = log(log(1 - 0.63)/log(1 - 0.31))/(log(self.d63 - self.wa) - log(self.d31-self.wa))
            g1,g2  = gamma(1+(1/wc)),gamma(1+(2/wc))
            wb     = ((-self.wa*g1)/g2)+((((self.wa/g2)**2)*(g1*g1-g2)+(self.dq*self.dq/g2))**0.5)
            retval = [self.wa, wb, wc]
            
        elif method == 'cao08':
            wc     = newton(self.fcao8, 2.01, tol=1e-5, maxiter=100, args=(self.wa, self.d95, self.dm))
            wb     = (self.d95 - self.wa)/(-log(1 - 0.95))**(1/wc)
            retval = [self.wa, wb, wc]
            
        elif method == 'cao09':
            wc     = newton(self.fcao9, 2.01, tol=1e-5, maxiter=100, args=(self.wa, self.d95, self.dq))
            wb     = (self.d95 - self.wa)/(-log(1 - 0.95))**(1/wc)
            retval = [self.wa, wb, wc]
            
        elif method == 'cao10':
            wa     =   ( self.d0 * (self.n**(1/3)) - self.d50)/((self.n**(1/3)) - 1)
            wc     = log(log(1 - 0.95)/log(1 - 0.25))/(log(self.d95 - wa) - log(self.d25-wa))
            g1, g2 = gamma(1 + (1/wc)), gamma(1 + (2/wc))
            wb     = -wa*(g1/g2) + sqrt((wa/g2)*(wa/g2)*(g1*g1 - g2) + self.dq*self.dq/g2)
            retval = [wa, wb, wc]
        
        elif method == '':
            
            retval = [wa, wb, wc]
        elif method =='garcia':
            b6, b7, b8, b9, b10, b11, b12, b13 = 0.03587, 0.19353, 0.4822, 0.7567, 0.91821, 0.89706, 0.98821, 0.57719
            cv = self.dsd/self.dm
            Gamma = ((((((((b6*cv-b7)*cv + b8) * cv-b9) * cv + b10)* cv-b11) * cv + b12) * cv-b13) * cv + 1)
            wa = 0.5*self.d0
            wb = abs(self.dm - self.d0)/Gamma
            wc = 1 / cv
            retval = [wa, wb, wc]
        else:
            raise ValueError("Method not recognized.")
            retval = [1., 1., 1.] 
        
        wa, wb, wc = retval
        return wa, wb, wc   

    def combined_score(self, statistics_values, weights):
        """Compute a combined score for ranking methods based on a dictionary of statistics."""
        
        if set(statistics_values.keys()) != set(weights.keys()):
            raise ValueError("Ensure that weights are provided for all statistics.")
        
        # Normalize each statistic to [0, 1]
        normalized_values = {}
        for stat, value in statistics_values.items():
            if stat == 'ks' or stat == 'ad':
                normalized_values[stat] = 1 - value  # Assuming smaller values are better
            elif stat == 'loglike':
                normalized_values[stat] = (value - abs(min(statistics_values.values()))) / (abs(min(statistics_values.values())) + max(statistics_values.values()))
            else:
                normalized_values[stat] = value  # Assuming values are already in [0, 1]

        score = sum(weights[stat] * normalized_values[stat] for stat in statistics_values.keys())
        return score
    
    def fit_all_methods(self):
        for method in self.methods:
            self.fit(method)
            statistics_values = {'ks': self.params[method]['ks'] ,
                                 'ad': self.params[method]['ad'] ,
                                 'loglike': self.params[method]['loglike'] }

            weights = {'ks': 1,'ad': 1,'loglike': 1}
            self.params[method]['rank'] = self.combined_score(statistics_values, weights)
            
    def plot_distributions(self):
        # pdb.set_trace()
        x = np.linspace(min(self.data), max(self.data), 1000)
        plt.figure(figsize=(10, 6))
        plt.hist(self.data, bins=20, density=True, alpha=0.5, label='Data', cumulative=True)
        for method in self.params:
            a = self.params[method]['a']
            b = self.params[method]['b']
            c = self.params[method]['c']
            y = np.vectorize(self.weibull_cdf)(x, a, b, c)
            plt.plot(x, y, label=method) 
            plt.legend()
        plt.title('Weibull Distributions by Method')
        plt.xlabel('x')
        plt.ylabel('Probability')
        plt.show()
        
    def get_best_model(self):
        best_method = None  # Inicializar a None
        best_loglike = float('inf')  # Inicializar a un valor muy grande
        
        for method, metrics in self.params.items():
            if metrics['loglike'] < best_loglike:
                best_method = method
                best_loglike = metrics['loglike']
        
        best_parameters = {
            'method': best_method,
            'a': self.params[best_method]['a'],
            'b': self.params[best_method]['b'],
            'c': self.params[best_method]['c'],
            'ks': self.params[best_method]['ks'],
            'ad': self.params[best_method]['ad'],
            'loglike': self.params[best_method]['loglike'],
            'rank': self.params[best_method]['rank']
        }
        
        return best_parameters


    def get_summary_table(self):
        # Crear una lista vacía para almacenar las filas de la tabla
        rows = []

        # Iterar sobre cada método y sus métricas
        for method, metrics in self.params.items():
            row = {
                'Method': method,
                'a': self.params[method]['a'],
                'b': self.params[method]['b'],
                'c': self.params[method]['c'],
                'ks': self.params[method]['ks'],
                'ad': self.params[method]['ad'],
                'loglike': self.params[method]['loglike'],
                'rank': self.params[method]['rank']
            }
            rows.append(row)

        # Convertir la lista de filas en un dataframe de pandas
        df = pd.DataFrame(rows)
        
        # Ordenar el dataframe por el valor p del test KS en orden descendente
        df = df.sort_values(by='loglike', ascending=False)

        return df