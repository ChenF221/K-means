import pandas as pd
import matplotlib.pyplot as plot
import sklearn.cluster as skl

#Adquiere los datos desde un archivo csv usando biblioteca PANDAS
dataframe = pd.read_csv('p9_team_data.csv', encoding='ISO-8859-1')
x = dataframe['off']
y = dataframe['def']


#Tomamos únicamente las características x,y para el set de entrenamiento
X = pd.DataFrame()
X['x'] = x
X['y'] = y

#Solicita al usuario el número de Clusters - Parámetro K
k = int(input("Indique el número de clusters k: "))

#Inicialización para algoritmo de clustering con el parámetro K indicado por el usuario
kmeansModel = skl.KMeans(n_clusters=k)

#Carga los datos
kmeansModel.fit(X)

#Obtiene los centroides
centroides = kmeansModel.cluster_centers_


#Obtiene una lista con las etiquetas de los datos
etiquetas = kmeansModel.predict(X)


#Agrega al frame de datos una columna para las etiquetas de clasificación
dataframe['clase_K-means'] = etiquetas

#Tabla de colores
colores =['maroon','teal','aqua','tan','indigo'] 

colores_datos=[]
colores_centroides=[]

for row in etiquetas:
    colores_datos.append(colores[row])

for i in range(len(centroides)):
    colores_centroides.append(colores[i])

#Grafica de dispersión
ax = plot.axes()
ax.scatter(x, y, c=colores_datos, marker='o', s=40)
ax.scatter(centroides[:,0], centroides[:,1], c=colores_centroides, marker='+', s=200)
plot.xlabel('Goal scored per game')
plot.ylabel('Goal lost per game')
plot.title('K-Means Clustering Goal scored vs Goal losted')
plot.grid(color='gray', linestyle='--', linewidth = 0.5)
plot.show()


dataframe.to_csv('p9_team_data_agrup.csv', encoding='ISO-8859-1')