# Simulación y Modelos

Aplicación de escritorio desarrollada en Python para la generación y análisis de variables aleatorias discretas y continuas.

El proyecto está compuesto por dos simuladores que permiten trabajar con distintas distribuciones de probabilidad, realizar pruebas estadísticas y visualizar los resultados mediante gráficos.

## Contenido

### Variables discretas
Permite generar y analizar las siguientes distribuciones:
- Poisson
- Binomial
- Geométrica

### Variables continuas
Permite generar y analizar las siguientes distribuciones:
- Exponencial
- Normal
- Uniforme

## Tecnologías utilizadas

- Python
- CustomTkinter
- SciPy
- Matplotlib
- Tkinter

## Estructura del proyecto

```text
simulacion-y-modelos/
├── variables-discretas/
│   └── simulador_discretas.py
├── variables-continuas/
│   └── simulador_continuas.py
├── README.md
└── requirements.txt
```

## Instalación y ejecución

1. Clonar o descargar el repositorio.
2. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

3. Ejecutar el simulador que se desea utilizar:

   **Variables discretas:**
   ```bash
   python variables-discretas/simulador_discretas.py
   ```

   **Variables continuas:**
   ```bash
   python variables-continuas/simulador_continuas.py
   ```
