# Charla Tesis

Se realizará el día Miércoles 03 de Agosto. Duración aproximada de 1 hora de exposición más 30 minutos de preguntas/trabajo.

#### Objetivo

El objetivo principal es:

- Realizar una charla donde Matías presente a sus profesores guía los resultados de su investigación sobre Aprendizaje Multietiqueta a través de Deep Learning, a  saber, los distintos métodos que encontró en el paper, sus principales características y modo de funcionamiento, a fin de reconocer los posibles caminos a seguir para el trabajo del segundo semestre.

Algunos objetivos "secundarios" podrían ser:

- Dar una charla de un tema relacionado al Deep Learning, que podría llamar la atención de eventuales oyentes para estudiar esta área o incluso para abordar problemas similares como temas de memoria.

---

## Categorías de MLC

- **MLC Tradicional**
  
  - Etiquetas completas y correctas
  
  - Cantidad baja de etiquetas con respecto a cantidad de ejemplos (por ejemplo, 10k ejemplos vs 10 etiquetas).

- **Extreme Multi-Label Learning (XMLC)**
  
  - Caso en el cual el espacio de las etiquetas es enorme (cientos de miles o hasta millones de etiquetas posibles).
  
  - Por ejemplo, Wikipedia incluye más de un millón de categorías, y uno podría estar interesado en construir un clasificador que, dado un nuevo artículo, indique el subconjunto de categorías más relevantes.

- **Multi-label learning with limited supervision**
  
  - En la práctica, conseguir datos completamente etiquetados y sin error es sumamente costoso.
  
  - Este fenómeno da lugar a distintos escenarios de etiquetados parciales o parcialmente correctos.
    
    - **Multi-label with missing labels (MLML),** que asume que solamente un subset de las etiquetas es dado,
    
    - **Semi-supervised MLC (SS-MLC),** que asume un gran conjunto de datos etiquetados y otro no etiquetado,
    
    - **Partial multi-label learning (PML),** que permite a los anotadores entregar un superconjunto de etiquetas canddatas.
    
    - **MLC with Noisy Labels (Noisy-MLC),** que permite entradas faltantes/incorrectas tanto en las etiquetas relevantes como irrelevantes,
    
    - **MLC with Unseen Labels,** que asume que el espacio de etiquetas puede crecer dinámicamente,
    
    - **Multi-Label Active Learning (MLAL),** en donde se seleccionan las instancias más informativas para ser entregadas como ejemplos a los modelos y que así estos aprendan con menos esfuerzo.
    
    - **Label Distribution Learning (LDL),** que busca reducir la ambigüedad de etiquetado (i.e. corregir etiquetados imperfectos) mediante el estudio de la distribución de las etiquetas en los datos.
    
    - **MLC with Multiple Instances (MIML),** que asume que cada ejemplo está descrito por múltiples instancias que a su vez están asociadas a múltiples etiquetas binarias. Se centra principalmente en detectar ambigüedades a nivel de features.

- **Online Multi-label Learning**
  
  - Datos en streaming. 
  
  - A diferencia del caso offline, no toda la data está disponible previamente para el aprendizaje.
  
  - En el caso offline, normalmente toda la data cae en memoria para el aprendizaje. En el caso online, rara vez.
  
  - En el caso online, la data podría ir "expirando" con el paso del tiempo. No se puede esperar a tener grandes conjuntos para generar el aprendizaje, si no más bien se requiere aprendizaje continuo. 
  
  - En la actualidad, el problema se aborda principalmente desde DL.

- **Statistical Multi-label Learning:**
  
  - Estudios probabilisticos de MLC.
  
  - No tratan problemas directamente, si no más bien la teoría detrás de los mismos.
  
  - Converge, por ejemplo, la función de loss de un clasificador a la pérdida de Bayes cuando el tamaño de ejemplos en entrenamiento aumenta? (Consistencia del aprendizaje: qué tan "aprendible" es el problema si se tienen tantos ejemplos como se desee)

## Enfoques clásicos para XMLC

#### Embedding-based

Los enfoques de embedding buscan resolver el problema al reducir el número efectivo de etiquetas al proyectar los vectores de etiquetas a un espacio de baja dimensión, basados en la asunción de que la matriz de labels es low-rank (muchas columnas dependen de unas pocas). 

Los vectores de etiquetas se proyectan entonces a un espacio de baja dimensión, y luego se entrenan regresores para predecir la proyección a partir de la entrada.  Las etiquetas para un punto nuevo se producen finalmente al multiplicar la proyección por una matriz de decompresión que devuelve el embedded label vector al espacio original.

Los distintos métodos varian principalmente en la elección de su técnica de compresión y decompresión (compressed sensing, Bloom filters, etc) 

**Ventajas:** fuerte base teórica, fácil de implementar, maneja correlaciones entre etiquetas, se adapta a escenarios online e incrementales

**Desventajas:** es lento tanto en entrenamiento como en predicción incluso para dimensiones de embedding bajas, hace asunciones demasiado grandes. En particular, el tema de la low-rank label matrix se rompe en aplicaciones reales donde existen muchas etiquetas "cola" (que aparecen en muy pocos ejemplos) y que por ende no pueden ser bien aproximados por ninguna base lineal de baja dimensión.

Método más popular: SLEEC

### Tree-based

Enfoque basado en dividir el problema principal en una secuencia de problemas de escala pequeña al particionar jerárquicamente el conjunto de instancias o el de etiquetas.

Su funcionamiento más básico es el siguiente: la jerarquía se aprende al optimizar una ranking loss. Una instancia se pasa hacia abajo en el árbol hasta que alcance una hoja (en el caso de un árbol de instancias) o varias hojas (arbol de etiquetas). 

Para un árbol de etiquetas, las hojas alcanzadas contienen las predicciones

Para un árbol de instancias, la predicción se realiza por un clasificador entrenado en las instancias del nodo hoja.

**Ventajas:** es que los costos de predicción son sub lineales o incluso logaritmicos si el árbol está balanceado. No es en general más eficaz, solo más eficiente.

**Desventajas:** El entrenamiento es complejo pues involucra problemas de optimización sobre espacios no convexos en cada nodo.

Método más popular: FastXML

### One-vs-all

Los métodos One-vs-all (OVA) son aquellos en donde se entrena un clasificador binario para cada etiqueta (al estilo Binary Relevance). Sin embargo, esta técnica sufre de dos mayores limitaciones para XMLC: Altos costos computacionales y tamaños de modelo extremadamente largos, lo cual genera una predicción lenta. 

Los métodos más novedoss buscan explotar la esparcidad de los datos para proponer una adaptación sublineal de los métodos OvA al caso XMLC. El ejemplo seminal es PD-Sparse, el cual propone minimizar la separation ranking loss y el l1 penalty en un enfoque de Empirical Risk Minimization (ERM). El ranking de separación penaliza la predicción de una instancia por la respuesta más alta del set de etiquetas negativas menos la respuesta más baja del set de respuestas positivas. El método obtiene una solución extremadamente esparsa en primal y dual con costo de tiempo sublineal, entregando una accuracy más alta que SLEEC, FastXML y otros métodos XMLC. 

**Ventajas:** Mejor accuracy que SLEEC

**Desventajas:** más lento en general, no explotan correlaciones entre labels

Método más popular: Binary Relevance

---

## MLC a través de Deep Learning

### Deep Embedding Methods

Estos métodos se basan en buscar un nuevo espacio de features mediante redes profundas (dejar en la red la tarea de aprender descriptores) y emplear un clasificador multilabel al final de la red. 

El método seminal es BP-MLL:  BackPropagation for Multi-Label Learning. Su principal aporte es el uso de una función de pérdida especialmente diseñada para el MLC que penaliza fuertemente la ausencia de un label positivo y debilmente un label negativo, y que estaría fuertemente relacionada con la Ranking Loss. 

Los métodos de embedding a través de Deep Learning han demostrado ser efectivos para capturar las dependencias entre etiquetas. Aún más, el estado del arte evidencia una relación entre la profundidad de los modelos y el orden de las dependencias: mientras modelos shallow encuentran dependencias más simples, los modelos más profundos (como C2AE) encuentran dependencias de alto nivel, dando incluso lugar al feature-aware label embedding y label-correlation aware prediction.

Método más popular: C2AE, RankAE (versión de C2AE con una función de pérdida especialmente pensada para XMLC con noisy labels)

### Advanced Deep Learning for MLC

Recientemente se han diseñado algunas arquitecturas específicamente pensadas para problemas de MLC. Estas se basan principalmente en el uso de celdas recurrentes y/o de memoria aplicadas sobre redes convolucionales para la codificación de relaciones entre etiquetas.

Algunos ejemplos son:

- CNN-RNN, 

- LSTM,

- Graph Convolutional Network (GCN)

- Deep Forest (MultiLabel Deep Forest, MLDF)

El problema de estos métodos es que en general incorporan millones de parámetros y presentan alta complejidad en términos de entrenamiento y predicción.

**CNN-RNN** aprende un embedding conjunto de imagen-etiquetas que permite captar la semántica detrás de una etiqueta así como la relevancia imagen-etiqueta (grado de relación). 

----

## Métodos seminales

#### SLEEC

- Trabajo pionero de XMLC, publicado el 2015. 

- Los enfoques de embedding no han logrado entregar altas accuracies en predicción, o escalar a problemas grandes debido a que la asunción del low rank se viola en la mayoría de las aplicaciones al mundo real.

- La principal contribución técnica de SLEEC es una formulación para aprender un conjunto pequeño de embeddings que preservan las distancias locales, los cuales permiten predecir etiquetas poco frecuentes. Esto permite a SLEEC romper con la asunción de low-rank y aumentar el accuracy de clasificacion al aprender embeddings que preservan las distancias pairwise solo entre los vectores más cercanos.

- SLEEC escala eficientemente a datasets con un millon de etiquetas.

- En primer lugar, en lugar de proyectar todas las etiquetas a un mismo espacio lineal de baja dimensión, SLEEC aprende embeddings en un espacio no lineal, pero que conserva las distancias lineales entre etiquetas cercanas.

- En predicción, en lugar de usar una matriz de decompresión, SLEEC usa un clasificador KNN en el embedding space (lo cual resulta en que, al entrenar, las distancias entre etiquetas cercanas se preserven).

- Una versión mejorada de SLEEC (y presentada en el mismo paper) usa clustering en lugar de KNN para la predicción.

### C2AE

- Deep Neural Network (DNN)

- Su objetivo es auntar a relacionar de mejor manera los dominios de feature y de etiquetas para obtener una mejor clasificación

- Se construye un embedding conjunto de features y labels a través de un deep latent space, seguido de la introducción de una función de perdida sensible a la correlación de etiquetas para recuperar labels de predicción.

- En particular, la arquitectura en juego es un autoencoder para etiquetas y otro para features, los cuales se reunen en una sola gran codificación a través de Deep Canonical Correlation Analysis (DCCA), lo cual permite un aprendizaje y predicción end-to-end   con la habilidad de explotar las dependencias entre etiquetas. 
  
  - Tres funciones de mapeo: Fx, feature mapping, Fe, encoding function, Fd, decoding function
  
  - En entrenamiento, el componente de DCCA se encarga de determinar el espacio latente L conformado por los encodings de X e Y, el cual posteriormente es forzado a sacar como output el mismo Y (arquitectura de autoencoder).

- Se extiende facilmente al problema de los missing labels

- Efectivo y robusto en multipls datasets con distintas escalas y densidades

<img title="" src="file:///C:/Users/m_jvs/AppData/Roaming/marktext/images/2022-07-19-19-31-14-image.png" alt="" width="331" data-align="center">

<img title="" src="file:///C:/Users/m_jvs/AppData/Roaming/marktext/images/2022-07-19-19-40-14-image.png" alt="" width="319" data-align="center">

### Binary Relevance

- Entrenar un clasificador binario por cada etiqueta. La predicción es la concatenación de las N predicciones individuales.

---

## MLC Datasets

![](C:\Users\m_jvs\AppData\Roaming\marktext\images\2022-07-18-15-35-40-image.png)

Destacan:

- `genbase`, con una cantidad de instancias y de etiquetas muy similar pero con una densidad de 3 veces la densidad de nuestro dataset.

- `medical`, con una cantidad de instancias razonablemente similar a la de nosotros, 45 etiquetas, y una densidad del doble. 

- `delicious`, con 16k instancias y 1k labels, pero con una densidad muy similar a nuestro caso

## Extractos utiles

**SLEEC**

- Challenges: XML is a hard problem that involves learning with hundreds of thousands, or even millions, of labels, features and training points. Although, some of these problems can be ameliorated using a label hierarchy, such hierarchies are unavailable in many applications [1, 2]. In this setting, an obvious baseline is thus provided by the 1-vs-All technique which seeks to learn an an independent classifier per label. As expected, this technique is infeasible due to the prohibitive training and prediction costs given the large number of labels

hacer indice

estimacion de paginas por subseccion
