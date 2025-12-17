# Modelado Predictivo del Rendimiento Financiero en Unidades de Negocio mediante Machine Learning

![Análisis financiero y control de gestión](img/controller_financiero.jpg)


## 🧭 Descripción

Este proyecto desarrolla un sistema de **análisis y modelado predictivo** basado en *Machine Learning* para **anticipar ineficiencias y desviaciones en el rendimiento financiero** de unidades de negocio.

Integra información financiera, ratios y variables operativas con el objetivo de **apoyar la toma de decisiones estratégicas**, especialmente en contextos de control de costes, optimización de recursos y seguimiento del desempeño.

El sistema se concibe como una **herramienta de alerta temprana**, no como un motor de decisión automático.

---
## 🎯 Objetivo del proyecto

Desarrollar un modelo de Machine Learning capaz de identificar de forma temprana unidades de negocio con riesgo de ineficiencia financiera, a partir de información financiera, ratios y variables operativas, con un enfoque de apoyo a la toma de decisiones.

---
## 💼 Problema de negocio

En organizaciones con múltiples unidades, los problemas de eficiencia suelen detectarse cuando:
- los márgenes ya se han deteriorado,
- los costes se han desviado,
- o la rentabilidad y el cumplimiento de objetivos están comprometidos.

Este proyecto aborda la siguiente cuestión:

> **¿Qué patrones financieros y operativos permiten anticipar pérdidas de eficiencia antes de que su impacto sea visible en los resultados consolidados?**

---

## 🗂️ Datos

Se integran **tres fuentes internas** con un total de **2.500 observaciones**, representativas de distintos sistemas de información corporativos:

- **Datos financieros**
- **Ratios financieros**
- **Datos operativos**

Los datasets presentan **problemas reales de calidad** (tipos inconsistentes, valores faltantes y necesidad de integración), tratados explícitamente durante el proceso de preparación.

### Variable objetivo

Clasificación binaria del rendimiento financiero de cada unidad de negocio (*eficiente* / *en riesgo*), definida a partir de umbrales económicos construidos sobre indicadores financieros y operativos.

---

## 🧠 Enfoque analítico

- Limpieza, validación e integración de datos  
- *Feature engineering* con criterio económico
  
### Aprendizaje supervisado
Modelos utilizados:
- Regresión logística (baseline e interpretabilidad)
- Árbol de decisión
- Random Forest
- Support Vector Machine (SVM)

### Aprendizaje no supervisado
- Reducción de dimensionalidad mediante **PCA**
- Segmentación de unidades de negocio mediante **K-Means**
 
El análisis no supervisado se utiliza como herramienta exploratoria para identificar patrones latentes y segmentaciones estructurales complementarias al modelado predictivo supervisado.

Evaluación centrada en **detección temprana, robustez y generalización**, no solo en accuracy  

---

## ⚖️ Decisiones analíticas clave

- **Priorizar recall frente a precisión**
- **Equilibrio entre interpretabilidad y rendimiento**
- **Control del sobreajuste**
- **Apoyo a la decisión, no automatización**

---

## 🎯 Resultado

La solución permite:

- identificar ineficiencias financieras de forma anticipada  
- priorizar unidades que requieren análisis o intervención  
- optimizar la asignación de recursos  
- reforzar el control de costes y la ejecución de objetivos estratégicos  

---

## 🖥️ Despliegue

El modelo final se expone mediante una **aplicación interactiva en Streamlit**, orientada a usuarios no técnicos, que permite introducir indicadores financieros y obtener una **clasificación de riesgo clara y accionable**.
La aplicación está pensada como apoyo al análisis financiero, no como sistema de decisión automática.

---

## 📁 Estructura del proyecto

Modelado_Predictivo_Rendimiento_Financiero_Unidades_Negocio_ML/
│
├── img
├── data
│ ├── raw
│ ├── processed
│ ├── train
│ └── test
│
├── Notebooks
├── src
├── Models
├── app_streamlit
├── docs
└── README.md


---

## 🔮 Posibles mejoras futuras

- Incorporar **series temporales más largas**
- Añadir **variables externas** (macroeconómicas o sectoriales)
- Reentrenamiento periódico del modelo
- Mayor énfasis en **explicabilidad**
- Integración directa en procesos de control de gestión, FP&A o reporting corporativo

---

## ✅ Conclusión

Este proyecto muestra cómo aplicar *Machine Learning* en un contexto de **análisis financiero avanzado**, combinando rigor analítico, datos imperfectos y orientación a decisión.

El foco no está en el modelo aislado, sino en **convertir información financiera y operativa en capacidad de anticipación y mejora**.
