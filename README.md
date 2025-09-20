

# Regresión Lineal usando NumPy
Técnicas de Machine Learning con NumPy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/)

Este proyecto implementa un modelo de regresión lineal desde cero utilizando la biblioteca NumPy de Python, explicando los fundamentos matemáticos detrás de este algoritmo de Machine Learning.

## Descripción

La regresión lineal es una técnica de aprendizaje automático supervisado que modela la relación entre una variable dependiente (y) y una o más variables independientes (X) mediante un enfoque lineal. 

Matemáticamente, el modelo se representa como:

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

Donde:
- y es la variable dependiente
- x₁, x₂, ..., xₙ son las variables independientes
- β₀ es el término de intersección (bias)
- β₁, β₂, ..., βₙ son los coeficientes de regresión
- ε es el término de error

## Funcionalidades

- Implementación de regresión lineal simple y múltiple
- Entrenamiento del modelo mediante el método de mínimos cuadrados ordinarios (OLS)
- Entrenamiento mediante gradiente descendente
- Cálculo de métricas de evaluación (R², MSE, RMSE)
- Predicción de valores continuos
- Visualización de resultados
- Validación del modelo

## Fundamentos Matemáticos

### Mínimos Cuadrados Ordinarios (OLS)

El método de mínimos cuadrados busca minimizar la suma de los residuos al cuadrado:

```
min Σ(yᵢ - ŷᵢ)²
```

La solución analítica para los coeficientes es:

```
β = (XᵀX)⁻¹Xᵀy
```

### Gradiente Descendente

Para optimizar la función de coste:

```
J(β) = (1/2m) Σ(hβ(xᵢ) - yᵢ)²
```

Donde:
- m es el número de muestras
- hβ(xᵢ) es la hipótesis del modelo

El algoritmo actualiza los parámetros mediante:

```
βⱼ := βⱼ - α(∂J(β)/∂βⱼ)
```

Donde α es la tasa de aprendizaje.

### Métricas de Evaluación

- **Error Cuadrático Medio (MSE)**: `MSE = (1/n)Σ(yᵢ - ŷᵢ)²`
- **Raíz del Error Cuadrático Medio (RMSE)**: `RMSE = √MSE`
- **Coeficiente de Determinación (R²)**: `R² = 1 - (SSres/SStot)`

## Instalación

Clona el repositorio e instala las dependencias necesarias:

```bash
git clone https://github.com/tu-usuario/Regresion_Linear_usando_Numpy.git
cd Regresion_Linear_usando_Numpy
pip install -r requirements.txt
```

## Uso

### Ejemplo básico

```python
import numpy as np
from linear_regression import LinearRegression

# Cargar datos
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X, y)

# Realizar predicciones
predictions = model.predict(np.array([[3, 5]]))
print(f"Predicción: {predictions}")
```

### Ejecutar el notebook

Para ejecutar el notebook con ejemplos detallados:

```bash
jupyter notebook
```

## Estructura del Proyecto

```
├── README.md
├── requirements.txt
├── linear_regression.py    # Implementación del modelo
├── utils.py                # Funciones auxiliares y visualización
├── tests/                  # Pruebas unitarias
│   └── test_linear_regression.py
└── examples/               # Notebooks con ejemplos
    └── regresion_lineal_ejemplos.ipynb
```

## Requisitos

- Python 3.8+
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0
- Jupyter Notebook (opcional, para ejecutar los ejemplos)

## Contribuciones

Las contribuciones son bienvenidas. Si deseas contribuir a este proyecto, por favor:

1. Realiza un fork del repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crea un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT - consulta el archivo [LICENSE](LICENSE) para más detalles.

## Contacto

[Nombre] - [Email] - [Twitter/GitHub]
