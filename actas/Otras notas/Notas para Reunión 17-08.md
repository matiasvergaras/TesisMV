# Notas Reunión 17-08

Resultados en tablesGenerator



**Optimizar CNN con hamming loss** no es factible: el gradiente de HL es 0 en casi todas partes. Usamos Binary Cross Entropy Loss with Logits como funcion de perdida 'proxy'.

**BCEWithLogits** nos permite ademas darle más peso a los positivos a través de pos_weights

Funcion make_positive_weights: 

- Un error positivo pesa (casos_negativos + casos_positivos) / casos_positivos

- Por ejemplo: si hay 250 casos negativos y 50 casos positivos para la etiqueta 'triangle', el peso de un falso negativo será equivalente a (250+50)/50 = 6 falsos positivos 

- Con esta medida AlexNet aprende a decir casi siempre que sí -> es demasiado.

- Se incluye un factor de ajuste que divide al factor anterior (ajuste=2 parece ser un valor razonable).

**Usar HS como score objetivo para el early stopping no escala bien al test de prueba. Sí lo hace F1-score.**

AlexNet se comporta en general igual o peor que metodos tradicionales (rakelD, aleatorio)

Podría estar sacando más provecho de datos sintéticos.



usar un scheduler
