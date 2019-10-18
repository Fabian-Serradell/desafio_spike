#%% [markdown]
## Desafío Spike
### Pregunta 1
#Primero importamos los datos desde 'caudal_extra.csv' usando pandas.
#Antes de importar se revisó visualmente los nombres de las columnas,
#ya que en el documento del desafío se indicó que una de las columnas contiene la estampa de tiempo de las mediciones.
#Con esto importamos, dejando la columna 'fecha' como el index del DataFrame, debido que queremos revisar los datos como series de tiempo.

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline

#%%
df = pd.read_csv('caudal_extra.csv', sep=',', 
                 parse_dates={'dt' : ['fecha']}, infer_datetime_format=True, 
                 low_memory=False, index_col='dt').sort_index()

#%% [markdown]
### Pregunta 2
#Revisamos las columnas y los tipos de datos que contiene.

#%%
df.head()

#%%
cantidades = ['{:>25}: {:10}'.format(column, df[column].unique().shape[0]) for column in df.columns]
print(*cantidades, sep='\n')

#%% [markdown]
# Existen un total de 133 estaciones, obtenidos de las columnas: codigo_estacion y nombre.
#
# Existe untotal de 133 sensores (virtuales en este caso por la agregación de datos),
# obtenido de las columnas gauge_id y gauge_name.
#
# Existen 29 cuencas diferentes, obtenido de la columna codigo_cuenca.
#
# Existen 78 sub cuencas, obtenido de la columna nombre_sub_cuenca.

#%% [markdown]
#De las columnas vistas vamos a eliminar:
# + institucion, fuente: sólo contiene la descripción DGA.
# + nombre de cuenca: tenemos el código de cada cuenca, por lo que está de más.
# + nombre sub cuenca: tenemos el código de cada estación, que corresponde con cada sub cuenca.
# + cantidad observaciones: por ahora sabemos que este dato está implícito en los códigos de estación y cuenca.
# + gauge id, gauge name: es el mismo código de cada estación. 
# 
#Se dejarán dejar por ahora las de altitud y localización por si se alcanza a hacer un análisis en cuanto
#a cuencas cercanas. Cambiaremos por conveniencia la columna de id (la podríamos usar en el futuro para buscar elementos).

#%%
df = df.drop(['institucion','fuente','nombre','nombre_sub_cuenca','cantidad_observaciones','gauge_id','gauge_name'],axis=1)
df = df.rename(columns = {'Unnamed: 0':'id'})

#%%
df.describe()

#%% [markdown]

#%%
hist_caudal = df['caudal'].plot.hist(title='Histograma Caudal', bins=100, alpha=0.5)
hist_caudal.set(xlabel='Caudal', ylabel='Frecuecia')
hist_caudal.text(10000, 800000, r'$\mu=95.5, b=252.6$')

#%%
hist_temp = df['temp_max_promedio'].plot.hist(title='Histograma Temperatura Máxima', bins=100, alpha=0.5)
hist_temp.set(xlabel='Temperatura', ylabel='Frecuecia')
hist_temp.text(-10, 30000, r'$\mu=19.2, b=7.3$')

#%%
hist_precip = df['precip_promedio'].plot.hist(title='Histograma  Precipitaciones', bins=100, alpha=0.5)
hist_precip.set(xlabel='Precipitación', ylabel='Frecuecia')
hist_precip.text(100, 800000, r'$\mu=1.9, b=7.4$')

#%% [markdown]
#La temperatura sigue una distribución normal, mientras que el caudal y la precipitación siguen un distribución de poisson con una media mu pequeña.
#Llama mucho la atención de la distribución de los datos de precipitación y caudal. Pienso que será dificil modelar el comportamiento para grandes cantidades de caudales o precipitaciones,
#debido a que en general los datos son muy cercanos a 0 y hay poca representatividad de valores grandes. 
#
#Revisamos la cantidad de datos que se importaron por defecto como NaN en cada columna.
#%%
df.isna().sum()

#%% [markdown]
#Se encuentra que sólo las columnas **precip_promedio** y **temp_max_promedio** tenían datos sin definir.
#
#Para el caso de precipitaciones promedio, 27767 datos están indefinidos, lo que representa un 1.97% del total de esas mediciones.
#Mientras tanto, para el caos de temperaturas máximas promedio, 151563 datos están indefinidos, lo que representa un 10.74% del total de esas mediciones.
#
#Para poder entender mejor si los datos indefinidos provienen de cuencas sin estaciones de monitoreo de temperatura o precipitación,
#se mostrarán la cantidad de estaciones (codigo_estacion) por cuenca y el porcentaje de mediciones indefinidas de temperatura por estación.

#%%
codigos_cuencas, codigos_estaciones= df['codigo_cuenca'].unique(), df['codigo_estacion'].unique()

#%%
for i, codigo in enumerate(codigos_cuencas):
    print('{:3}: {:2}'.format(codigo, df[df['codigo_cuenca'] == codigos_cuencas[i]]['codigo_estacion'].unique().shape[0]))

#%%
for i, codigo in enumerate(codigos_estaciones):
    cantidad_nan = df[df['codigo_estacion'] == codigos_estaciones[i]]['temp_max_promedio'].isna().sum()
    total = df[df['codigo_estacion'] == codigos_estaciones[i]]['temp_max_promedio'].count()
    print(codigo, (cantidad_nan/total)*100)

#%% [markdown]
#Se esperaría que las estaciones que no tengan medidores de temperatura tengan un 100% de datos NaN, lo que no se da en este caso.
#Otra posibilidad es que los datos NaN correspondan a los primeros de la serie de tiempo par cada
#estación, lo que mostraría que los datos NaN corresponden a fechas donde todavía no se implementaba un sensor en la estación.
#Si existen datos sin definir intermedios se tomará como una falla en la medición para esa estampa de tiempo.
#
#Por restricciones de tiempo no revisaremos cada caso, pero graficaremos en el siguiente punto algunas mediciones de
#precipitaciones, temperatura y caudal para ver si las posiciones NaN son intermedias o al principio de la serie de tiempo.

#%%
df[df['codigo_estacion'] == 5423003].plot(y='temp_max_promedio')

#%% [markdown]
#TODO: hacer gráfico donde se muestre cada cuenca y los valores de mediciones indeterminadas

#%% [markdown]
### Pregunta 3
#a. Escribir una función que tome como input una estación
#y haga plot de los datos para una columna.

#%%
def time_plot_una_estacion(dataframe, codigo_estacion, columna, fecha_min, fecha_max):
    df_aux = df[fecha_min:fecha_max].copy()
    return df_aux[df_aux['codigo_estacion'] == codigo_estacion].plot(y=columna)

#%%
time_plot_una_estacion(df ,4540001, 'temp_max_promedio', '1968-01','1968-12')

#%% [markdown]
#Por ejemplo en la figura anterior se ve que existen meses o días sin medida para la estación 
#4540001, lo que indicaría que existen fallas de medida para este caso en particular.


#%% [markdown]
#3.b. Función que haga plots de varias columnas, para poder visualizar
#caudal, precipitación y temperatura al mismo tiempo.
#
#TODO: mejorar dividir por el primer valor que no sea NaN o 0 en la serie.

#%%
def time_plot_estaciones_varias_columnas(dataframe, codigo_estacion, columnas, fecha_min, fecha_max):
        df_aux = df[fecha_min:fecha_max].copy()
        for column in columnas:
            df_aux[[column]] = df_aux[[column]]/df_aux[[column]].values[0]
        return df_aux[df_aux['codigo_estacion'] == codigo_estacion].plot(y=columnas)
#%%
columnas_plot = ['temp_max_promedio', 'caudal', 'precip_promedio']
time_plot_estaciones_varias_columnas(df, 4540001, columnas_plot, '1974-06','1974-09')

#%% [markdown]
### Pregunta 4
#Primero vamos a crear 4 variables que contengan los ragos de tiempo para las estaciones del año.
#
#Para el caso de considerar la distribución histórica, se considerará la de las estaciones hasta el momento de la medición,
#esto es, si para el 01 de marzo de 1970 se tiene cierta medición de caudal/temperatura/precipitación,
#entonces para elegir si está por sobre o debajo del 95 percentil se tomarán todas las medidas de
#verano anteriores a esa fecha.
#
#También se podría haber tomado el valor histórico de toda la serie de tiempo, pero captura la idea de
#valores "mayores a lo esperado" como lo hace lo descrito en el párrafo anterior (valores esperados son mirando al pasado).
#
#Para simplificar tomaremos de enero-marzo como verano, sin contar que la estación cambia cerca del 21 del mes.
#Lo mismo se hará para las demás estaciones del año.

#%%
verano = ((df.index.month >= 1) & (df.index.month <= 3))
otono = ((df.index.month >= 4) & (df.index.month <= 6))
invierno = ((df.index.month >= 7) & (df.index.month <= 9))
primavera = ((df.index.month >= 10) & (df.index.month <= 12))

#%%
estaciones_tiempo = (verano, otono, invierno, primavera)

#%% [markdown]
#Las 6 celdas siguientes sólo muestran porque un percentil agregado puede traer problemas,
#especialmente por su comportamiento transitorio en un principio, debido a los pocos datos tomados 
#desde un inicio, lo que se va acotando en la medida que los datos aumentan y estos son menos relevantes
#para la historria completa.
#
#Con esto se decide que la mejor forma por ahora es volver al inicio y tomar el percentil del total.
#
#En el futuro se podría probar con un percentil movil con el tamaño de ventana como parámetro y con corrección de bias inicial.
#%%
test = df[verano].copy()

#%% [markdown]
#Se tomaron sólo las mediciones de verano de la estación "4540001"

#%%
test_array = test[test['codigo_estacion'] == 4540001]['temp_max_promedio'].dropna().values

#%%
np.percentile(test_array, 95)

#%% [markdown]
# Se calcula el percentil mirando sólo los datos pasados en cada timestep.
#%%
temp_aux = []
percentile = []
for i, item in enumerate(test_array):
    temp_aux = np.append(temp_aux, item)
    percentile = np.append(percentile, np.percentile(temp_aux, 95))

#%%
temp_max = []
for i, item in enumerate(test_array):
    if item > percentile[i]:
        temp_max = np.append(temp_max, 1)
    else:   
        temp_max = np.append(temp_max, 0)

#%%
plt.plot(percentile)
plt.title('Percentil mirando al pasado')
plt.show()
plt.plot(temp_max)
plt.title('Eventos extremos')
plt.show()

#%% [markdown]
#Se ve como el percentil se mueve en torno a los 30.7 aprox rápidamente al pasar el tiempo.
#Sin embargo, en los datos iniciales se genera una transiente que distorsiona los datos extremos, como se puede ver
#en el gráfico de los eventos extremos en un inicio. Por esto se tomarán por ahora todos los datos de la serie de tiempo (separados por sus estaciones y temporadas anuales).

#%%
caudal_extremo = {}
# Pasamos por cada una de las estaciones del tiempo.
for estacion_tiempo in estaciones_tiempo:
    df_temporada = df[estacion_tiempo].copy()
    for codigo_estacion in codigos_estaciones:
        df_estacion = df_temporada[df_temporada['codigo_estacion'] == codigo_estacion].copy()
        # Para calcular el percentil correspondiente se eliminan los datos inválidos.
        caud_percentile = df_estacion['caudal'].dropna().quantile(0.95)
        for row in df_estacion.itertuples():
            caudal = row.caudal
            if caudal > caud_percentile:
                caudal_extremo[row.id] = 1
            # Comparar con un NaN
            elif np.isnan(caudal):
                caudal_extremo[row.id] = np.nan
            else:  
                caudal_extremo[row.id] = 0

#%% [markdown]
#Se hace lo mismo para temperatura y precipitaciones. 
#TODO: En una futura iteración se puede definir una función para limpiar el código y hacerlo escalable con otras funciones.

#%%
temp_extremo = {}
# Pasamos por cada una de las estaciones del tiempo.
for estacion_tiempo in estaciones_tiempo:
    df_temporada = df[estacion_tiempo].copy()
    for codigo_estacion in codigos_estaciones:
        df_estacion = df_temporada[df_temporada['codigo_estacion'] == codigo_estacion].copy()
        # Para calcular el percentil correspondiente se eliminan los datos inválidos.
        temp_percentile = df_estacion['temp_max_promedio'].dropna().quantile(0.95)
        for row in df_estacion.itertuples():
            temp = row.temp_max_promedio
            if temp > temp_percentile:
                temp_extremo[row.id] = 1
            # Comparar con un NaN
            elif np.isnan(temp):
                temp_extremo[row.id] = np.nan
            else:  
                temp_extremo[row.id] = 0

#%%
precip_extremo = {}
# Pasamos por cada una de las estaciones del tiempo.
for estacion_tiempo in estaciones_tiempo:
    df_temporada = df[estacion_tiempo].copy()
    for codigo_estacion in codigos_estaciones:
        df_estacion = df_temporada[df_temporada['codigo_estacion'] == codigo_estacion].copy()
        # Para calcular el percentil correspondiente se eliminan los datos inválidos.
        prec_percentile = df_estacion['precip_promedio'].dropna().quantile(0.95)
        for row in df_estacion.itertuples():
            precip = row.precip_promedio
            if precip > prec_percentile:
                precip_extremo[row.id] = 1
            # Comparar con un NaN
            elif np.isnan(precip):
                precip_extremo[row.id] = np.nan
            else:  
                precip_extremo[row.id] = 0

#%% [markdown]
#Aprovechando el id único de los datos, se mapean las variables al DataFrame original.
#%%
df['caudal_extremo'] = df['id'].map(caudal_extremo)
df['temp_extremo'] = df['id'].map(temp_extremo)
df['precip_extremo'] = df['id'].map(precip_extremo)

#%% [markdown]
#Para poder revisar las distintas variables  se puede hacer un gráfico de los eventos extremos
#acumulados para cada variable. Esto está lejos de ser óptimo visualmente, pero permite ver rápidamente
#la relación de los eventos para cada una de las variables.

#%%
caudal_plot = df['caudal_extremo'].groupby('dt').agg(['cumsum']).plot()
temp_plot = df['temp_extremo'].groupby('dt').agg(['cumsum']).plot()
precip_plot = df['precip_extremo'].groupby('dt').agg(['cumsum']).plot()

caudal_plot.set(xlabel='fecha', ylabel='caudal extremo')
caudal_plot.legend(['cantidad de eventos diarios'])
temp_plot.set(xlabel='fecha', ylabel='temperatura extrema')
temp_plot.legend(['cantidad de eventos diarios'])
precip_plot.set(xlabel='fecha', ylabel='precipitación extrema')
precip_plot.legend(['cantidad de eventos diarios'])

caudal_plot.plot()
temp_plot.plot()
precip_plot.plot()
#%% [markdown]
#Es interesante ver los datos agregados por fechas, 
#aunque se debe tomar en cuenta que las estaciones de monitoreo no tienen datos distribuidos
#en los mismo rangos de tiempo muchas veces, por lo que agregarlos de esta forma puede ser engañoso.
#
#Por eso se procede a responder la pregunta 5 al ver el comportamiento por estaciones.

#%% [markdown]
### Pregunta 5
#Vamos a hacer gráficos en el tiempo de eventos extremos, pero diferenciando por cuenca.

#%%
for cuenca in codigos_cuencas:
    cuenca = df['caudal_extremo'][df['codigo_cuenca'] == cuenca].agg(['cumsum']).plot(title=cuenca)

#%% [markdown]
#Efectivamente el comportamiento de algunas cuencas es diferentes para el caudal extremo.
#Algunas tienen un comportamiento casi lineal, mientras que otras tienen un valle durante la decada de los 90s
#para luego volver a aumentar. En otros casos, como en la cuenca 94 y 103, se han desacelerado los eventos
#de caudal extremo en la decada del 2010. 
# 
#Dado esto, podríamos agregar las cuencas que tengan comportamientos similares
#para el modelo del punto 7.

#%% [markdown]
#Para analizar las diferentes cuencas haremos lo siguiente:
#
# 1. Tomar subcuencas y analizar si los eventos extremos son similares entre ellas.
# 2. Si las subcuencas son similares, entonces tomar una estación por cuenca para el análisis.
# 3. Si las subcuencas no son similares, entonces nuestro análisis se hará por estación.
# 
#Lo anterior debido a que, dependiendo de la cuenca, se pueden tener varias estaciones,
#lo que no hará comparable los resultados entre cuencas.

#%% [markdown]
### Pregunta 6
#Hagan un plot del porcentaje de eventos extremos a través del tiempo (caudal_extremo,
#temp_extremo, precip_extremo). Se han vuelto más o menos comunes?
#
#Primero se hará un resample de todos los datos extremos por trimestre, sin contar los indeterminados.

#%%
caudal_ano = df['caudal_extremo'].resample('Q').count()
temp_ano = df['temp_extremo'].resample('Q').count()
precip_ano = df['precip_extremo'].resample('Q').count()

#%% [markdown]
#Se calculan todos los eventos extremos, haciendo un resample por trimestre.
#%%
caudal_extremo_ano = df['caudal_extremo'].resample('Q').sum()
temp_extremo_ano = df['temp_extremo'].resample('Q').sum()
precip_extremo_ano = df['precip_extremo'].resample('Q').sum()

#%% [markdown]
#Con lo anterior se puede sacar el porcentaje de eventos extremos por trimestre,
#para ver si estos han subido a bajado en frecuencia.

#%%
caudal_porcent = (caudal_extremo_ano/caudal_ano)*100
temp_porcent = (temp_extremo_ano/temp_ano)*100
precip_porcent = (precip_extremo_ano/precip_ano)*100

#%%
caudal_porcent.plot()
#%% [markdown]
#Los eventos extremos de caudal han disminuido durante la última decada.
#%%
temp_porcent.plot()

#%% [markdown]
#Los eventos extremos de temperatura han aumentado durante la última decada.
#%%
precip_porcent.plot()
#%% [markdown]
#Los eventos extremos de precipitaciones habría que analizarlos con mayor detalle.

#%% [markdown]
### Pregunta 7 y 8
#### Modelos
#Para este caso en particular eliminaremos los datos NaN. Esto tendrá el problema de romper el step de la serie de tiempo.
#Para otra iteración ver como imputar los datos. También se podría usar un valor que sabemos que caudal, precipitación o temperatura máxima diaria no alcanza
#para ver si el modelo aprende que cuando hay NaN, entonces el resultado
#también debiese ser NaN. Si una de las variables antes mencionadas tienen un NaN en la fila, entonces la fila completa lo tendrá.
#
#Por ahora sólo se usaran los datos que se han obtenido hasta ahora, pero por ejemplo se podrían
#crear 3 variables más, correspondientes a las derivadas de la temperatura máxima promedio, precipitación promedio y caudal.

#%%
dataset = df[['codigo_estacion','codigo_cuenca','caudal','precip_promedio','temp_max_promedio','caudal_extremo','temp_extremo','precip_extremo',]].dropna().copy()
#dataset = df[['caudal','precip_promedio','temp_max_promedio']].dropna().copy()

#for column in dataset.columns:
#    dataset[column] = dataset[column].fillna(dataset[column][:1900000].mean())

#%% [markdown]
#Para hacer el modelo comenzaremos por modelar sólo el caudal extremo para una estación.

#%%
dataset_prueba = dataset[dataset['codigo_estacion'] == 8140002][['caudal','precip_promedio','temp_max_promedio','caudal_extremo','temp_extremo','precip_extremo']].sort_index().values



#%% [markdown]
'''
### Decision Trees
#%%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset_prueba.drop('caudal_extremo',axis=1).values, dataset_prueba['caudal_extremo'].values, test_size=0.25, shuffle=False)
print(X_train.shape)
print(y_train.shape)

#%%
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

tree_param_grid = {
    'criterion': ['mse','mae'],
    'splitter': ['best','random'],
    'max_depth': [2,4,8,16]
}

scores = ['r2']


for score in scores:
    clf = GridSearchCV(DecisionTreeRegressor(), tree_param_grid, cv=5,
                       scoring=score)
    clf.fit(X_train, y_train)

    print("Mejor paŕametro encontrado para set de desarrollo:")
    print()
    print(clf.best_params_)
    print()
    print("Puntuación para el set de desarrollo:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    y_true, y_pred = y_test, clf.predict(X_test)
    print(r2_score(y_true, y_pred))
    print()

#%%
from sklearn.metrics import mean_squared_error

tree = DecisionTreeRegressor(criterion='mse',splitter='best',max_depth=8)
tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)
mean_squared_error(y_test,y_pred)

#%%
plt.plot(y_pred)
y_pred = [1 if x > 0.5 else 0 for x in y_pred] 
#%%
print(np.sum(abs(y_pred - y_test))/len(y_test))
'''
#%% [markdown]
#### Redes neuronales recurrentes
#Se usarán en una primera instancia una red neuronal recurrente, que tomará un batch de datos,
#mirando 32 días hacia atrás y prediciendo con eso el caudal extremo para el día siguiente.
#El paso será de un día y cada batch será de 64 muestras.

#%%
#mean = dataset_prueba[:3500].mean(axis=0)
#dataset_prueba -= mean
#std = dataset_prueba[:3500].std(axis=0)
#dataset_prueba /= std

#%%
#print(mean)
#print(std)
#print(dataset_prueba.shape)

#%% [markdown]
#Se usa un generador para entregar datos de entrenamiento y validación a la red. 
#La salida del modelo es la cuarta posición del arreglo.
#%%
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        
        X = np.zeros((len(rows), lookback // step, data.shape[-1]))
        y = np.zeros((len(rows),))
    
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            X[j] = data[indices]
            y[j] = data[rows[j] + delay][3]
        
        yield X, y

#%%
# lookback es la cantidad de días que estoy mirando hacia atrás.
# Se usará 32 que es aprox 1 meses de datos. 
lookback = 32
# paso para la muestra, por ejemplo en este caso en que cada muestra representa
# un día, un step de 7 tomaría una muestra por semana.
step = 1
# Delay representa la cantidad de pasos mirando hacia el futuro. En este caso
# 1 sería para poder calcular el caudal del día siguiente, teniendo las muestras de los 
# días anteriores.
delay = 1
batch_size = 64

train_gen = generator(dataset_prueba,
                    lookback=lookback,
                    delay=delay,
                    min_index=0,
                    max_index=3499,
                    shuffle=False,
                    step=step,
                    batch_size=batch_size)
val_gen = generator(dataset_prueba,
                    lookback=lookback,
                    delay=delay,
                    min_index=3500,
                    max_index=3999,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(dataset_prueba,
                    lookback=lookback,
                    delay=delay,
                    min_index=4000,
                    max_index=None,
                    step=step,
                    batch_size=batch_size)
val_steps = (3999 - 3500 - lookback)
test_steps = (len(dataset_prueba) - 4000 - lookback)

#%%
X_test = []
y_test = []
for i in range(test_steps):
    X_test, y_test = next(test_gen)

#%%
X_test.shape

#%%[markdown]
#Se usará una red con una capa GRU de 32 unidades escondidas y luego se condensará un una sóla en la siguiente capa, con 
#una función sigmoide de activación. Se usará la función de pérdida crossentropy, y se optimizará el modelo usando el algoritmo Adam por su eficiencia.
#En un principio intentaremos entrenar la red sin estándarizar los datos, lo que puede provocar problemas de llegar al óptimo.

#%%
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam

model = Sequential()
model.add(layers.GRU(32, activation='relu',
                    input_shape=(None, dataset_prueba.shape[-1])))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_gen,
                            steps_per_epoch=50,
                            epochs=20,
                            validation_data=val_gen,
                            validation_steps=val_steps)

#%%
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Pérdida de entrenamiento')
plt.plot(epochs, val_loss, 'b', label='Pérdida de validación')
plt.title('Pérdida de entrenamiento y validación')
plt.legend()
plt.show()

#%%
y_pred = model.predict(X_test, batch_size=64)
print(y_pred)

#%%
print(y_test)

#%% [markdown]
#El modelo no predice ningún evento de caudal extremo. Para este caso la métrica de accuracy no es buena
#porque puedo tener un vector completo con 0s y aún asi tener un buen porcentaje. En este caso se debe usar
#alguna métrica de clasificación binaria para dar cuenta de los casos negativos.
#
#Para seguir trabajando con redes neuronales sería bueno agregar datos para poder alimentar mejor la red, queda pendiente.