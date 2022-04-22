# Notas para reu: 30-03-22

**Temas**:

- Lectura de "Towards Class-Imbalance Aware Multi-Label Learning"

- Lectura de "Integration of deep learning model and feature selection for multi-label classification"

- Data augmentation 

- Errores de clasificación: ¿son los mismos al variar el threshold?

  

**Towards Class-Imbalance Aware Multi-Label Learning**

- El paper incorpora conceptos como Label Density y Label Cardinality. 
- El enfoque propuesto, COCOA, se compara con los enfoques de transformación del problema que ya probamos (algoritmos lineales, no lineales y de ensemble).
- Se usa como principal métrica el F1-Score. Esto es razonable en general, pues es una media harmónica entre precision y recall. Sin embargo, a nosotros podría interesarnos más el recall, caso en el que podría convenir usar F0.5-Score.
- Se realizan experimentos sobre 18 datasets distintos, con distinto tipo de entrada. De ellos, el con la menor label density alcanza un valor de 0.036, poco más del doble de nuestro caso (0.015). Se trata de Bibtex, un dataset de texto. El mejor F-Score es de 0.326 por RML y el segundo 0.318 por COCOA.
- También se incluyen dos datasets de imagenes, Corel5k y Scene. COCOA resulta ser el mejor y segundo mejor algoritmo para cada caso, con F-score de 0.728 y 0.196, respectivamente.
- Me llama la atencion que se mencionan los **árboles de decisión** como un buen clasificador base para problemas de class imbalance. **Debería realizar experimentos con ellos para ver si mejora nuestro estado actual.**
- No creo que COCOA sirva para nuestro caso. El algoritmo trata de abarcar el desbalance de clases pero no en una misma clase si no en pares de clases, para asi contribuir a la forma en que se descubren las relaciones entre labels.
  - En este escenario, frecuentemente hay un alto desbalance pues es dificil tener muchos patrones con ambas etiquetas (tenemos algunos con una, otros con la otra y solo algunos con ambas).
  - Lo que propone es, para una etiqueta A, generar K predictores multiclase que predigan a su vez A y alguna otra etiqueta B_i, transformando el problema de multilabel en multiclase: 0 si no aparece ni A ni B, 1 si solo aparece A, 2 si solo aparece B, 3 si aparecen ambas.  Y la mejora que propone es reducir las cuatro clases a solo 3, combinando el caso 1 y 3 (es decir: cada vez que aparezca el label que me interesa predecir, asumo que viene también con su label correlacionado).
  - Qué tendría que pasar para que este enfoque nos sirva: **que un label facil de predecir (probablemente aquellos mas frecuentes) arrastre siempre labels especificos (mas dificiles)**
- No encontré implementaciones de COCOA, pero podría codearlo si lo consideramos necesario.



**Integration of deep learning model and feature selection for multi-label classification**

- Enfoque basado en deep learning

- Graph-based feature selection para reducir dimensiones

- Reducir las dimensiones de los datos como una forma de reducir también la cantidad de datos necesarios para entrenar (razonable)

  

**Data Augmentation**

- Pasamos de 775 patrones a 3285
- La frecuencia de cada etiqueta aumentó muchísimo. Ahora con t=15 tenemos 144 labels, mientras que con t=60 tenemos 57.
- Para el caso desbalanceado, los resultados no mejoran, pero cambia levemente el ranking de algoritmos
  - Precision alrededor de 0.7, recall alrededor de 0.4
- Para el caso balanceado (solo BR) los resultados empeoran levemente.
- 

**Errores de clasificación: ¿son los mismos al variar el threshold?**

- Se prueba con Binary Relevance sobre Logistic Regression, en t=40 (HL: 0.82) y t=60 (HL: 0.944). 

- Caso t=40:

  <img src="D:\Escritorio\Tesis-git\notas_reunion\Captura de pantalla 2022-03-27 a las 20.18.37.png" alt="Captura de pantalla 2022-03-27 a las 20.18.37" style="zoom: 67%;" />

- Caso t=60:

  <img src="D:\Escritorio\Tesis-git\notas_reunion\Captura de pantalla 2022-03-27 a las 20.22.44.png" alt="Captura de pantalla 2022-03-27 a las 20.22.44" style="zoom:67%;" />

  

En general, los resultados no cambian de un threshold a otro. Algunas excepciones - tales como el caso de la etiqueta hatched - se asume pueden deberse a la reconstrucción de train y test size de forma aleatoria.

