import numpy as np
import matplotlib.pyplot as plt

T60_med = np.array([3.14, 3.15, 3.92, 3.65, 3.35, 3.89, 3.79, 3.32])
T60_uten = np.array([4.28, 4.28, 4.13, 3.76, 4.14, 4.33, 4.10, 4.21])

V=240 #volum i m^3
S_absorbent=10 #absorbsjonsareal i m^2 
c=343.4 #lydhastighet i m/s

def alpha_absorbent(T60_med, T60_uten, V, S_absorbent, c):
    return (24*V*np.log(10))/(c*S_absorbent)*((1/np.mean(T60_med))-(1/np.mean(T60_uten)))


# Funksjon for å beregne et estimat for standardavviket for alpha_absorbent
def estimated_std_error_alpha_absorbent(T60_med, T60_uten, V, S_absorbent, c):
    return 24*V*np.log(10)/(c*S_absorbent)*(np.sqrt(1/(np.mean(T60_med)**4)*np.std(T60_med)**2+1/(np.mean(T60_uten)**4)*np.std(T60_uten)**2))

print("Estimatet for standardavviket er: ", estimated_std_error_alpha_absorbent(T60_med, T60_uten, V, S_absorbent, c))

# Funksjon for å beregne et 95% konfidensintervall for alpha_absorbent basert på estimatet for standardavviket
def confidence_interval_alpha_absorbent(T60_med, T60_uten, V, S_absorbent, c):
    std_error = estimated_std_error_alpha_absorbent(T60_med, T60_uten, V, S_absorbent, c)
    alpha = alpha_absorbent(T60_med, T60_uten, V, S_absorbent, c)
    lower_bound = alpha - 2.365 * (std_error / np.sqrt(len(T60_med)))
    upper_bound = alpha + 2.365 * (std_error / np.sqrt(len(T60_med)))
    return lower_bound, upper_bound

print("95% konfidensintervall for alpha_absorbent: ", confidence_interval_alpha_absorbent(T60_med, T60_uten, V, S_absorbent, c))