# Predicción de Abandono Estudiantil con Árboles de Decisión

**Instituto Superior de Formación Técnica N° 197**  
**Alumno:** Lucas Nahuel Llanes  
**Materia:** Procesamiento de Aprendizaje Automático  
**Docente:** Scudero, Yanina Ximena  
**Carrera:** Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial  
**Año:** 2026

---

## 🎯 PUNTO B) - Predicción de abandono en estudiantes terciarios

Modelo predictivo desarrollado en Python para identificar el riesgo de deserción académica y evaluar qué variables influyen en la continuidad de los estudiantes.

## 📊 Dataset REAL de ENTRENAMIENTO (con `EstadoFinal`)

- **100 registros** con variables académicas, demográficas y contextuales.
- Distribución: 54 Abandonó (54%) / 46 Continúa (46%).
- Entropía original: 0.9954 (incertidumbre casi máxima).
- **Hallazgo clave:** las medias entre clases son casi idénticas, lo que indica que el dataset **no tiene señal predictiva**.

## 🚀 Introducción al Proyecto

El objetivo de este trabajo es automatizar el análisis de datos educativos mediante un enfoque basado en Machine Learning. El script procesa un dataset histórico de entrenamiento, evalúa la precisión del modelo mediante un conjunto de prueba, extrae las variables con mayor peso analítico y genera predicciones sobre nuevos registros para exportarlas a un reporte final (`predicciones_DatosFinal.xlsx`).

## 🌲 ¿Qué es un Árbol de Decisión?

Es un algoritmo de aprendizaje supervisado que clasifica datos imitando un diagrama de flujo jerárquico:

- **Impureza de Gini:** Mide el grado de mezcla de clases en un nodo. El árbol elige los cortes que más reducen esta impureza.
- **Importancia de variables:** Cuánto contribuyó cada variable a reducir la impureza en los cortes. Los valores suman 1.
- **Hojas:** Representan la decisión final (Continúa / Abandonó).

## 💻 Implementación en el Código

1. **Carga y preparación:** Lee los Excel de entrenamiento y predicción. Corrige el error de tipeo "TÉCNICATURA" → "TECNICATURA", codifica binarias (0/1) y aplica One-Hot Encoding a `carrera`.
2. **División:** 70% entrenamiento / 30% prueba, estratificado por clase.
3. **Entrenamiento:** `DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=5)`.
4. **Evaluación:** Accuracy, matriz de confusión y reporte por clase.
5. **Interpretación (Punto 4):** Importancia de variables + reglas del árbol + **prueba de estabilidad** con 5 divisiones distintas.
6. **Predicción:** Aplica el modelo a `DatosFinal` y exporta `predicciones_DatosFinal.xlsx` ordenado por probabilidad.

## 📁 Archivos del Proyecto

| Archivo | Descripción |
|---------|-------------|
| `AD.py` | Script Python unificado con todo el flujo |
| `AD2.ipynb` | Cuaderno Jupyter con el mismo flujo |
| `AD.html` | Informe en HTML con estilo institucional |
| `TablaPrediccionAbandono-Entrenamiento_OG.xlsx` | Dataset REAL de entrenamiento |
| `TablaPrediccionAbandono-DatosFinal.xlsx` | Nuevos estudiantes a predecir |
| `predicciones_DatosFinal.xlsx` | Salida con predicciones (se genera al ejecutar) |
| `arbol_decision.png` | Gráfico del árbol entrenado |
| `matriz_confusion.png` | Matriz de confusión |
| `importancia_variables.png` | Ranking de importancia de variables |

## 🚀 Cómo Ejecutar

### Opción 1: Script Python

```bash
pip install pandas scikit-learn matplotlib openpyxl
python AD.py