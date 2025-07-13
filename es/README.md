
# Recursos para el Aprendizaje y Desarrollo en Machine Learning

Este repositorio reúne conceptos, prácticas y ejemplos de código para entender y aplicar técnicas de Machine Learning en diversos contextos académicos y de investigación. Está diseñado para profesores, investigadores y estudiantes que desean profundizar en los principios y herramientas fundamentales que impulsan el campo de Machine Learning, desde algoritmos básicos hasta modelos avanzados de aprendizaje profundo.

---

## **Visualización de una Red Neuronal en Entrenamiento**

![Red Neuronal en Entrenamiento](https://github.com/sgevatschnaider/machine-learning/blob/904ce259b00f331c9d0d550d0870ff4771e73033/recursos/red%20neuronal.gif)

La animación anterior muestra una red neuronal artificial entrenándose en tiempo real. Las capas densas (fully connected) reciben y propagan señales a través de sus nodos, y los pesos se ajustan continuamente durante el proceso de retropropagación. A la derecha se observan métricas clave del entrenamiento, incluyendo la precisión (`accuracy`), la función de pérdida (`loss`) y el número de épocas (`epochs`) ya completadas. Esta representación visual es ideal para comprender intuitivamente el aprendizaje profundo y cómo evolucionan los modelos a lo largo del tiempo.

---
## **Contenido**
1. [Fundamentos de Machine Learning](#1-fundamentos-de-machine-learning)
2. [Modelos Clásicos de Machine Learning](#2-modelos-clásicos-de-machine-learning)
3. [Redes Neuronales y Aprendizaje Profundo](#3-redes-neuronales-y-aprendizaje-profundo)
4. [Aprendizaje por Refuerzo (Reinforcement Learning)](#4-aprendizaje-por-refuerzo-reinforcement-learning)
5. [Procesamiento de Lenguaje Natural (NLP)](#5-procesamiento-de-lenguaje-natural-nlp)
6. [Visualización y Análisis de Datos](#6-visualización-y-análisis-de-datos)
7. [Ejemplos y Casos de Estudio](#7-ejemplos-y-casos-de-estudio)

---

## **1. Fundamentos de Machine Learning**
Conceptos básicos y fundamentos necesarios para comenzar en Machine Learning.

- **Tipos de Aprendizaje**: Introducción a los tipos de aprendizaje supervisado, no supervisado, semi-supervisado y por refuerzo.
- **Procesamiento de Datos**: Técnicas para limpieza y preparación de datos, incluyendo manejo de datos faltantes, normalización y transformación de datos.
- **Evaluación de Modelos**: Métricas de evaluación para clasificaciones, regresiones y clustering (precisión, recall, F1-score, MSE, etc.).

---

## **2. Modelos Clásicos de Machine Learning**
Implementación de algoritmos tradicionales de Machine Learning y sus aplicaciones.

- **Regresión Lineal y Logística**: Modelos básicos para tareas de predicción y clasificación.
- **Árboles de Decisión y Bosques Aleatorios**: Algoritmos de árbol para clasificación y regresión.
- **Máquinas de Soporte Vectorial (SVM)**: Modelos para clasificación con márgenes óptimos.
- **K-Means y Clustering**: Técnicas de agrupamiento para análisis no supervisado.

---

## **3. Redes Neuronales y Aprendizaje Profundo**
Introducción y práctica con redes neuronales y modelos avanzados de deep learning.

- **Redes Neuronales Artificiales (ANNs)**: Conceptos básicos y ejemplos de redes de perceptrones multicapa.
- **Redes Convolucionales (CNNs)**: Aplicaciones en visión por computadora.
- **Redes Recurrentes (RNNs)**: Modelos para secuencias de datos, como series temporales y procesamiento de texto.
- **Transfer Learning**: Cómo aprovechar modelos preentrenados para mejorar la eficiencia y precisión.

---

## **4. Aprendizaje por Refuerzo (Reinforcement Learning)**
Exploración de algoritmos y conceptos fundamentales del aprendizaje por refuerzo.

- **Fundamentos del Aprendizaje por Refuerzo**: Introducción a conceptos clave como agentes, estados, acciones, recompensas y entorno.
- **Métodos de Control Basados en Políticas**: Estrategias como SARSA y Q-learning.
- **Aprendizaje Profundo en RL**: Implementaciones avanzadas como Deep Q-Networks (DQN) y Aprendizaje A3C.
- **Casos de Uso**: Aplicaciones en videojuegos, robótica y optimización de procesos.

---

## **5. Procesamiento de Lenguaje Natural (NLP)**
Técnicas y modelos para trabajar con datos textuales.

- **Representación de Textos**: Métodos como Bag of Words, TF-IDF y embeddings (Word2Vec, GloVe).
- **Modelos de NLP**: Implementación de algoritmos de clasificación de texto, análisis de sentimiento y generación de texto.
- **Transformers y BERT**: Introducción a los modelos avanzados de NLP, incluyendo ejemplos con BERT y GPT.

---

## **6. Visualización y Análisis de Datos**
Herramientas y técnicas para explorar y visualizar datos.

- **Visualización de Datos**: Gráficas y diagramas para el análisis de datos, utilizando librerías como Matplotlib y Seaborn.
- **Análisis Exploratorio de Datos (EDA)**: Técnicas para entender la distribución y patrones de los datos antes de aplicar modelos.

---

## **7. Ejemplos y Casos de Estudio**
Aplicaciones prácticas de Machine Learning en diferentes áreas.

- **Predicción de Precios**: Aplicación de modelos de regresión para la predicción de precios de bienes.
- **Clasificación de Imágenes**: Uso de CNNs para clasificar imágenes en diferentes categorías.
- **Análisis de Sentimientos en Redes Sociales**: Clasificación de texto para identificar sentimientos en comentarios de redes sociales.

---

## **Estructura del Repositorio**
```plaintext
machine-learning-research/
├── data/                  # Datos de entrada (datasets para entrenar modelos)
├── notebooks/             # Notebooks de Jupyter para experimentos y demostraciones
│   ├── es/                # Notebooks en español
│   │   ├── Estadística_Paradigma_EDA_y_p.ipynb
│   │   ├── Aprendizaje_por_Refuerzo_en_Lenguaje_Natural_(NLRL).ipynb
│   ├── en/                # Notebooks en inglés
│       ├── Statistics_EDA.ipynb
│       ├── Natural_Language_Reinforcement_Learning_(NLRL).ipynb
├── scripts/               # Scripts de Python para procesamiento y entrenamiento
├── models/                # Modelos entrenados para pruebas y evaluaciones
├── results/               # Resultados, gráficas y visualizaciones
├── references/            # Artículos, papers y material de referencia
└── README.md              # Documentación del repositorio
```

---
## Fundamentos de Inteligencia Artificial y Machine Learning

Esta sección contiene notebooks teóricos y conceptuales que establecen las bases para entender el campo de la IA, sus subcategorías y los diferentes enfoques de aprendizaje.

| 📄 Recurso | 📥 Acceso |
| :--- | :--- |
| **Introducción a la IA y Machine Learning <br><br> <details><summary><strong>Resumen:</strong> <em>(haz clic para expandir/colapsar)</em></summary><p>Este notebook es una guía introductoria y puramente conceptual que desglosa y clarifica los términos fundamentales del mundo de la Inteligencia Artificial. Es el punto de partida ideal para cualquiera que desee entender el "qué" y el "porqué" antes de sumergirse en el "cómo".<br><br><h4>Jerarquía de Conceptos</h4><p>Explica la relación entre los campos principales, a menudo representados como círculos concéntricos:</p><ul><li><strong>Inteligencia Artificial (IA):</strong> El campo más amplio, enfocado en crear máquinas que puedan simular la inteligencia humana (razonar, aprender, resolver problemas).</li><li><strong>Machine Learning (ML):</strong> Un subconjunto de la IA que se centra en algoritmos que permiten a las máquinas aprender de los datos para identificar patrones y tomar decisiones sin ser programadas explícitamente para cada tarea.</li><li><strong>Deep Learning (DL):</strong> Un subconjunto del ML que utiliza redes neuronales artificiales con múltiples capas ("profundas") para resolver problemas complejos, especialmente en áreas como el reconocimiento de imágenes y el procesamiento del lenguaje natural.</li></ul><h4>Tipos de Aprendizaje Automático (Machine Learning)</h4><p>El notebook detalla los tres paradigmas principales del ML:</p><ol><li><strong>Aprendizaje Supervisado:</strong> Entrenar un modelo con datos "etiquetados" (donde ya conocemos la respuesta correcta). Se divide en:<ul><li><strong>Regresión:</strong> Predecir un valor numérico continuo (ej: el precio de una casa).</li><li><strong>Clasificación:</strong> Predecir una categoría o clase (ej: si un correo es spam o no).</li></ul></li><li><strong>Aprendizaje No Supervisado:</strong> Entrenar un modelo con datos "no etiquetados" para que encuentre patrones o estructuras ocultas por sí mismo. Incluye:<ul><li><strong>Clustering:</strong> Agrupar datos similares (ej: segmentación de clientes).</li></ul></li><li><strong>Aprendizaje por Refuerzo:</strong> Un "agente" aprende a tomar decisiones interactuando con un entorno. Recibe "recompensas" o "castigos" por sus acciones, con el objetivo de maximizar la recompensa total. Es la base de muchos sistemas de juegos (AlphaGo) y robótica.</li></ol></p></details> | [![Ver en GitHub](https://img.shields.io/badge/Ver%20en-GitHub-blue?style=for-the-badge&logo=github)](https://github.com/sgevatschnaider/machine-learning/blob/1c135db89f4acc380588fb81996dd7320f766a56/notebooks/Machine_Learning_Inteligencia_artificial.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sgevatschnaider/machine-learning/blob/main/notebooks/Machine_Learning_Inteligencia_artificial.ipynb) |




##  Aprendizaje Automático (Machine Learning)

| 📄 Recurso | 📥 Acceso |
| :--- | :--- |
| **Introducción al Machine Learning** <br><br> <details><summary><strong>Resumen:</strong> <em>(haz clic para expandir/colapsar)</em></summary><p>Este notebook proporciona una introducción conceptual al campo del Machine Learning. Define qué es, explora sus aplicaciones prácticas y distingue entre los principales tipos de aprendizaje (supervisado, no supervisado y por refuerzo). Además, introduce la terminología fundamental como modelos, características (features) y etiquetas (labels), sentando las bases teóricas para el estudio de algoritmos más complejos.</p></details> | <a href="https://github.com/sgevatschnaider/machine-learning/blob/1e243455df8027e527fc903191626af477132439/notebooks/es/Machine_Learning_Introducci%C3%B3n_a_la_materia_.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Ver%20en-GitHub-blue?style=for-the-badge&logo=github" alt="Ver en GitHub"></a> <a href="https://colab.research.google.com/github/sgevatschnaider/machine-learning/blob/main/notebooks/es/Machine_Learning_Introducci%C3%B3n_a_la_materia_.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> |

---

## **Instalación y Configuración**
Para ejecutar los ejemplos de este repositorio, asegúrate de tener instalado Python (3.7 o superior) y los paquetes necesarios listados en `requirements.txt`.

### Clona el repositorio:
```bash
git clone https://github.com/tu-usuario/machine-learning-research.git
cd machine-learning-research
```

### Instala las dependencias necesarias:
```bash
pip install -r requirements.txt
```

---

## **Ejecución de Ejemplos**
Cada carpeta incluye instrucciones específicas para ejecutar los ejemplos. Puedes ejecutar los notebooks de Jupyter directamente para seguir los pasos de cada proyecto o utilizar los scripts para realizar pruebas y evaluaciones automáticas.

---

## **Contribuciones**
Las contribuciones son bienvenidas. Si deseas agregar nuevos ejemplos o mejorar los existentes, sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama para tus cambios:
   ```bash
   git checkout -b feature/nueva-funcionalidad
   ```
3. Realiza un pull request con una descripción detallada de tus cambios.

Consulta el archivo `CONTRIBUTING.md` para más detalles sobre cómo contribuir.

---

## **Licencia**
Este proyecto está licenciado bajo la licencia MIT. Para más detalles, consulta el archivo `LICENSE`.


