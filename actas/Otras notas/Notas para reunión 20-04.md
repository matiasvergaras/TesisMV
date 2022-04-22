**Notas para reunión 20/04 **

- Se mueve todo el código a Drive para poder conectar el almacenamiento de archivos grandes  con la ejecución en Colab.
- Se ordena el código en distintos notebooks con el nombre del proceso al cual responden. El notebook "Testing multilabel algorithms" se documenta en profundidad.
- Al llevar a cabo el punto anterior, se descubren problemas relacionados con la forma en que se implementaba data augmentation.
  - **En el script para generar features,** los datos se separaban en Train, Val y Test, pero solo se generaban features para Val y Test. 
  - Como resultado, teniamos labels aumentados, pero no features. 
  - El código mergeaba ambos datasets y pasaba desapercibido. 
  - Se estaba realizando **data augmentation antes del benchmarking,** que separaba los datos en train y test. 
  - Como consecuencia, estabamos generando **resultados falsos** (podían existir copias en train y test).
- Se repara todo lo anterior y se lleva a cabo data augmentation mediante cropping.
- Los resultados son pésimos. Con los 12 labels más frecuentes:
  - Micro precision 0.25
  - Micro recall 0.14
- Se propone utilizar técnicas más avanzadas de data augmentation: [Augmentor](https://augmentor.readthedocs.io/en/master/userguide/mainfeatures.html) 





tener una sola gran pipeline, en donde se haga lo siguiente:

- escoger datos entre las siguientes opciones
  - original 
  - augmented
- Si es augmented, elegir un subconjunto de
  - [reflexions, rotations, crop, opciones del augmentor, ruido gaussiano]
- Escoger modelo para las features (tendremos dos modelos ya  entrenados)
- si la combinacion modelo+data existe en listdir, usar esa
- si no, generarla
- entregar los datos listos (Xtrain, Xval, Xtest, Ytrain, Yval, Ytest)

A la salida de la pipeline, hacer los experimentos