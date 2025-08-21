"""
Modelado Estocástico - 2025
Maestría en Inteligencia Artificial
Universidad de San Andrés
Trabajo Grupal N°1
Grupo IBM

Alumnos: 
- Aguirre Felipe
- Caccia Matias
- Casiraghi Tadeo 
- Carrizo Oscar
- Marin Del Boca José 
- Mesch Henriques Sebastian

Fecha de entrega: 19 de Julio del 2025

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#%% 1 carga de datos
df=pd.read_excel("_IBM.xlsx")

# Columna de datos de IBM
adjprice=df['Adj Price'].values
# Armado de la columna para los días con su formato
df['Date'] = pd.to_datetime(df['Date'])
dates = df['Date']
log_return_dates = dates.iloc[1:].reset_index(drop=True)

# Visualización de los datos
df.head()

#%% 2 calcular retornos logarítmicos diarios
log_returns = np.diff(np.log(adjprice))  # len = n - 1
log_return_dates = dates.iloc[1:].reset_index(drop=True)

#%% 3 graficar los datos 
# Retornos diarios
plt.figure(figsize=(12, 4))
plt.plot(dates, adjprice, label='Precio ajustado')

# Ticks mensuales
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.title("Precio Ajustado - IBM")
plt.xlabel("Fecha")
plt.ylabel("Precio")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Retornos logarítmicos diarios
plt.figure(figsize=(12, 4))
plt.plot(log_return_dates, log_returns, label='Retorno logarítmico diario', color='orange')

# Ticks mensuales
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.title("Retornos Logarítmicos Diarios - IBM")
plt.xlabel("Fecha")
plt.ylabel("log(Retorno)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

#%% 4 cantidad de observaciones logarítmicas finales
n_obs = len(log_returns)
print(f"Cantidad de retornos logarítmicos diarios: {n_obs}")

#%% 6 simulación de 1000 retornos logarítmicos
n_simulations = 1000
n_dias_mes = 20

np.random.seed(100)

# Simulaciones: cada fila tiene 20 índices aleatorios
random_indices = np.random.randint(0, n_obs, size=(n_simulations, n_dias_mes))
retorno_simulado = log_returns[random_indices].sum(axis=1)

#%% 9 estadísticos de los retornos logarítmicos diarios REALES
print("\n--- Estadísticas de Retornos Logarítmicos Diarios ---")
print(f"Promedio: {np.mean(log_returns):.6f}")
print(f"Desvío estándar (Volatilidad diaria): {np.std(log_returns):.6f}")
print(f"Percentil 1%: {np.percentile(log_returns, 1):.6f}")
print(f"Percentil 2.5%: {np.percentile(log_returns, 2.5):.6f}")
print(f"Percentil 5%: {np.percentile(log_returns, 5):.6f}")
print(f"Percentil 97.5%: {np.percentile(log_returns, 97.5):.6f}")
print(f"Primer cuartil (Q1): {np.percentile(log_returns, 25):.6f}")

# 10 estadísticos de los retornos logarítmicos mensuales SIMULADOS
print("\n--- Estadísticas de Retornos Mensuales Simulados ---")
print(f"Promedio: {np.mean(retorno_simulado):.6f}")
print(f"Desvío estándar (Volatilidad mensual): {np.std(retorno_simulado):.6f}")
print(f"Percentil 1%: {np.percentile(retorno_simulado, 1):.6f}")
print(f"Percentil 2.5%: {np.percentile(retorno_simulado, 2.5):.6f}")
print(f"Percentil 5%: {np.percentile(retorno_simulado, 5):.6f}")
print(f"Percentil 97.5%: {np.percentile(retorno_simulado, 97.5):.6f}")
print(f"Primer cuartil (Q1): {np.percentile(retorno_simulado, 25):.6f}")

# 11 verificación de la relación entre volatilidades
print("\n--- Relación entre las volatilidades ---")
vol_diaria = np.std(log_returns)
vol_mensual = np.std(retorno_simulado)
print("\nVerificación: Vol mensual = √n * Vol diaria")
print(f"Vol mensual simulada: {vol_mensual:.6f}")
print(f"Vol diaria * √20: {vol_diaria * np.sqrt(20):.6f}")

#%% 12 histograma de retornos mensuales simulados
print("\n--- Histograma de retornos mensuales simulados ---")
plt.figure(figsize=(12, 4))
plt.hist(retorno_simulado, bins=100, edgecolor='gray', color='lightblue')
plt.title("Histograma de Retornos Logarítmicos Mensuales Simulados")
plt.xlabel("Retorno mensual")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.show()