
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
| **Introducción a la IA y Machine Learning** <br><br> <details><summary><strong>Resumen:</strong> <em>(haz clic para expandir/colapsar)</em></summary><p>Este notebook es una guía introductoria y puramente conceptual que desglosa y clarifica los términos fundamentales del mundo de la Inteligencia Artificial. Es el punto de partida ideal para cualquiera que desee entender el "qué" y el "porqué" antes de sumergirse en el "cómo".<br><br><h4>Jerarquía de Conceptos</h4><p>Explica la relación entre los campos principales, a menudo representados como círculos concéntricos:</p><ul><li><strong>Inteligencia Artificial (IA):</strong> El campo más amplio, enfocado en crear máquinas que puedan simular la inteligencia humana (razonar, aprender, resolver problemas).</li><li><strong>Machine Learning (ML):</strong> Un subconjunto de la IA que se centra en algoritmos que permiten a las máquinas aprender de los datos para identificar patrones y tomar decisiones sin ser programadas explícitamente para cada tarea.</li><li><strong>Deep Learning (DL):</strong> Un subconjunto del ML que utiliza redes neuronales artificiales con múltiples capas ("profundas") para resolver problemas complejos, especialmente en áreas como el reconocimiento de imágenes y el procesamiento del lenguaje natural.</li></ul><h4>Tipos de Aprendizaje Automático (Machine Learning)</h4><p>El notebook detalla los tres paradigmas principales del ML:</p><ol><li><strong>Aprendizaje Supervisado:</strong> Entrenar un modelo con datos "etiquetados" (donde ya conocemos la respuesta correcta). Se divide en:<ul><li><strong>Regresión:</strong> Predecir un valor numérico continuo (ej: el precio de una casa).</li><li><strong>Clasificación:</strong> Predecir una categoría o clase (ej: si un correo es spam o no).</li></ul></li><li><strong>Aprendizaje No Supervisado:</strong> Entrenar un modelo con datos "no etiquetados" para que encuentre patrones o estructuras ocultas por sí mismo. Incluye:<ul><li><strong>Clustering:</strong> Agrupar datos similares (ej: segmentación de clientes).</li></ul></li><li><strong>Aprendizaje por Refuerzo:</strong> Un "agente" aprende a tomar decisiones interactuando con un entorno. Recibe "recompensas" o "castigos" por sus acciones, con el objetivo de maximizar la recompensa total. Es la base de muchos sistemas de juegos (AlphaGo) y robótica.</li></ol></p></details> | [![Ver en GitHub](https://img.shields.io/badge/Ver%20en-GitHub-blue?style=for-the-badge&logo=github)](https://github.com/sgevatschnaider/machine-learning/blob/1c135db89f4acc380588fb81996dd7320f766a56/notebooks/Machine_Learning_Inteligencia_artificial.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sgevatschnaider/machine-learning/blob/main/notebooks/Machine_Learning_Inteligencia_artificial.ipynb) |




##  Aprendizaje Automático (Machine Learning)

| 📄 Recurso | 📥 Acceso |
| :--- | :--- |
| **Introducción al Machine Learning** <br><br> <details><summary><strong>Resumen:</strong> <em>(haz clic para expandir/colapsar)</em></summary><p>Este notebook proporciona una introducción conceptual al campo del Machine Learning. Define qué es, explora sus aplicaciones prácticas y distingue entre los principales tipos de aprendizaje (supervisado, no supervisado y por refuerzo). Además, introduce la terminología fundamental como modelos, características (features) y etiquetas (labels), sentando las bases teóricas para el estudio de algoritmos más complejos.</p></details> | <a href="https://github.com/sgevatschnaider/machine-learning/blob/1e243455df8027e527fc903191626af477132439/notebooks/es/Machine_Learning_Introducci%C3%B3n_a_la_materia_.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Ver%20en-GitHub-blue?style=for-the-badge&logo=github" alt="Ver en GitHub"></a> <a href="https://colab.research.google.com/github/sgevatschnaider/machine-learning/blob/main/notebooks/es/Machine_Learning_Introducci%C3%B3n_a_la_materia_.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> |

##  Probabilidad y Estadística

| 📄 Recurso | 📥 Acceso |
| :--- | :--- |
| **Notebook: Fundamentos de Probabilidad y Estadística** <br><br> <details><summary><strong>Resumen:</strong> <em>(haz clic para expandir/colapsar)</em></summary><p>Un notebook esencial que cubre los pilares de la probabilidad y la estadística, conceptos indispensables para el Machine Learning. Explora temas como estadística descriptiva (media, varianza), distribuciones de probabilidad y teoremas clave. Este material es crucial para entender cómo se analizan los datos, se evalúan los modelos y se cuantifica la incertidumbre en las predicciones.</p></details> | <a href="https://github.com/sgevatschnaider/machine-learning/blob/a4f6115bbf6382254703d03864ca163415f9edc2/notebooks/es/Probabilidad_y_estadistica_.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Ver%20en-GitHub-blue?style=for-the-badge&logo=github" alt="Ver en GitHub"></a> <a href="https://colab.research.google.com/github/sgevatschnaider/machine-learning/blob/main/notebooks/es/Probabilidad_y_estadistica_.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> |


---

| 📄 Recurso | Enlaces |
|---|---|
| **Probabilidad, Estadística y Funciones Hash** <br><br> <details><summary><strong>Resumen:</strong> <em>(haz clic para expandir/colapsar)</em></summary><p>Notebook que repasa nociones clave de probabilidad y estadística y las conecta con el comportamiento de las funciones hash: uniformidad, colisiones, integridad de datos y aplicaciones en seguridad. Incluye ejemplos en Python.</p></details> | [![Ver en GitHub](https://img.shields.io/badge/Ver%20en-GitHub-blue?style=for-the-badge&logo=github)](https://github.com/sgevatschnaider/machine-learning/blob/9f86d0721e8cd2810e9e8b579f00722460ae4c7f/notebooks/es/Probabilidad_%2C_estad%C3%ADstica__Funciones_hash.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sgevatschnaider/machine-learning/blob/9f86d0721e8cd2810e9e8b579f00722460ae4c7f/notebooks/es/Probabilidad_%2C_estad%C3%ADstica__Funciones_hash.ipynb) |
| **Entropía (Información e Incertidumbre)** <br><br> <details><summary><strong>Resumen:</strong> <em>(haz clic para expandir/colapsar)</em></summary><p>Recurso sobre entropía como medida de incertidumbre/información en distribuciones de probabilidad. Discute intuiciones, ejemplos numéricos y usos en machine learning, compresión y teoría de la información.</p></details> | [![Abrir Página Interactiva](https://img.shields.io/badge/Abrir%20Página-Interactiva-brightgreen?style=for-the-badge&logo=html5)](https://sgevatschnaider.github.io/machine-learning/recursos/entropia.html) |

| 📄 Recurso | Enlaces |
|---|---|
| **Problema de Fermi y la Sabiduría de Masas** <br><br> <details><summary><strong>Resumen:</strong> <em>(haz clic para expandir/colapsar)</em></summary><p>Este notebook explora el clásico "Problema de Fermi" y el concepto de sabiduría de masas, mostrando cómo las estimaciones colectivas pueden acercarse sorprendentemente a la realidad. Incluye teoría, ejemplos y aplicaciones en inteligencia colectiva y análisis de datos.</p></details> | [![Ver en GitHub](https://img.shields.io/badge/Ver%20en-GitHub-blue?style=for-the-badge&logo=github)](https://github.com/sgevatschnaider/IA-Teoria-Practica/blob/1d71e15637c6a3f2fa32698ab8bb420e5135ad3f/notebooks/Problema_de_Fermi_y_la_Sabidur%C3%ADa_de_masas.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sgevatschnaider/IA-Teoria-Practica/blob/1d71e15637c6a3f2fa32698ab8bb420e5135ad3f/notebooks/Problema_de_Fermi_y_la_Sabidur%C3%ADa_de_masas.ipynb) |


### Distribuciones Discretas y Continuas

| 📄 Recurso | 📥 Acceso |
| :--- | :--- |
| **Distribuciones Discretas y Continuas** <br><br> <details><summary><strong>Resumen:</strong> <em>(haz clic para expandir/colapsar)</em></summary><p>Notebook dedicado al estudio de las distribuciones de probabilidad discretas y continuas, esenciales en estadística y machine learning. Incluye explicaciones teóricas, ejemplos prácticos, visualizaciones y casos de uso en la modelización y el análisis de datos.</p></details> | [![Ver Notebook](https://img.shields.io/badge/Ver%20Notebook-en%20GitHub-orange?style=for-the-badge&logo=github)](https://github.com/sgevatschnaider/machine-learning/blob/a33017fcb85f0a2f4656518ce0db8eab7d072890/notebooks/es/Distribuciones_discretas_y_continuas.ipynb) |

### Regresión Lineal, Outliers y Random Forest e Hiperparámetros 

| 📄 Notebook | 📥 Acceso |
| :--- | :--- |
| **Regresión Lineal** <br><br> <details><summary><strong>Resumen:</strong> <em>(haz clic para expandir/colapsar)</em></summary><p>Notebook enfocado en la regresión lineal, uno de los modelos fundamentales en estadística y machine learning. Incluye fundamentos teóricos, ejemplos prácticos, análisis de resultados y visualizaciones para comprender la relación entre variables.</p></details> | [![Ver Notebook](https://img.shields.io/badge/Ver%20Notebook-en%20GitHub-orange?style=for-the-badge&logo=github)](https://github.com/sgevatschnaider/machine-learning/blob/8d5368e5d9e37830efe7641fc6642d5c9622b6b3/notebooks/es/Regresion_Lineal.ipynb) |
| **Outliers** <br><br> <details><summary><strong>Resumen:</strong> <em>(haz clic para expandir/colapsar)</em></summary><p>Notebook dedicado a la detección y análisis de outliers (valores atípicos) en conjuntos de datos. Presenta métodos estadísticos, ejemplos prácticos y visualizaciones para identificar, tratar e interpretar outliers en análisis de datos y modelos predictivos.</p></details> | [![Ver Notebook](https://img.shields.io/badge/Ver%20Notebook-en%20GitHub-orange?style=for-the-badge&logo=github)](https://github.com/sgevatschnaider/machine-learning/blob/76504b3ca948a777ef8bb2eb77115a33add738b7/notebooks/es/Outliers.ipynb) |
| **Random Forest e Hiperparámetros** <br><br> <details><summary><strong>Resumen:</strong> <em>(haz clic para expandir/colapsar)</em></summary><p>Notebook dedicado a la técnica de Random Forest, un modelo de ensamble ampliamente utilizado en machine learning. Incluye teoría, ejemplos prácticos y estrategias para la selección de hiperparámetros.</p></details> | [![Ver Notebook](https://img.shields.io/badge/Ver%20Notebook-en%20GitHub-orange?style=for-the-badge&logo=github)](https://github.com/sgevatschnaider/machine-learning/blob/64d5d3e366332b38952f444d5666c07d725302c5/notebooks/es/Random_forest_e_hiperpar%C3%A1metros_.ipynb) |

| 📄 Recurso | Enlaces |
|---|---|
| **Sesgo, Varianza y Entropía** <br><br> <details><summary><strong>Resumen:</strong> <em>(haz clic para expandir/colapsar)</em></summary><p>Notebook que explica los conceptos fundamentales de sesgo, varianza y entropía en el contexto de modelos de machine learning. Incluye ejemplos, visualizaciones y aplicaciones prácticas para comprender el equilibrio entre estos conceptos en la construcción de modelos predictivos.</p></details> | [![Ver en GitHub](https://img.shields.io/badge/Ver%20en-GitHub-blue?style=for-the-badge&logo=github)](https://github.com/sgevatschnaider/machine-learning/blob/0d8037f01f56a159a530ab1e8c255d5433b2ee23/notebooks/es/Sesgo%2C_Varianza_y_entropia.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sgevatschnaider/machine-learning/blob/0d8037f01f56a159a530ab1e8c255d5433b2ee23/notebooks/es/Sesgo%2C_Varianza_y_entropia.ipynb) |


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


