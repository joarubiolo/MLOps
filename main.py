from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from rake_nltk import Rake
import numpy as np

# Creacion de una aplicacion FastApi

app = FastAPI()

#df = pd.read_csv(r'C:\Users\rubio\Documents\SoyHenry\Proyecto_individual_2\df.csv')

df_endpoint1 = pd.read_parquet(r'C:\Users\rubio\Documents\SoyHenry\Proyecto_individual_2\end1.parquet')
df_endpoint2 = pd.read_parquet(r'C:\Users\rubio\Documents\SoyHenry\Proyecto_individual_2\end2.parquet')
df_endpoint3 = pd.read_parquet(r'C:\Users\rubio\Documents\SoyHenry\Proyecto_individual_2\end3.parquet')
df_endpoint4 = pd.read_parquet(r'C:\Users\rubio\Documents\SoyHenry\Proyecto_individual_2\end4.parquet')
df_endpoint5 = pd.read_parquet(r'C:\Users\rubio\Documents\SoyHenry\Proyecto_individual_2\end5.parquet')
df_endpoint6 = pd.read_parquet(r'C:\Users\rubio\Documents\SoyHenry\Proyecto_individual_2\end6.parquet')
df_reco = pd.read_parquet(r'C:\Users\rubio\Documents\SoyHenry\Proyecto_individual_2\reco.parquet')
#df_endpoint1 = df[['title', 'release_month']]
#df_endpoint2 = df[['title', 'release_day']]
#df_endpoint3 = df[['title','popularity','release_year']]
#df_endpoint4 = df[['title','release_year','vote_average','vote_count']]
#df_endpoint5 = df[['title', 'actors', 'return']]
#df_endpoint6 = df[['title', 'return','director','release_date','budget','revenue']]


@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes:str):
    filmaciones = df_endpoint1['title'][df_endpoint1['release_month'] == mes]
    cantidad = filmaciones.count()

    return f'{cantidad} películas fueron estrenadas en el mes de {mes}'


@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia:str):
    filmaciones = df_endpoint2['title'][df_endpoint2['release_day'] == dia]
    cantidad = filmaciones.count()
    return f'{cantidad} películas fueron estrenadas en el dia {dia}'


@app.get("/score_titulo/{titulo}")
def score_titulo(titulo:str):
    film = df_endpoint3[df_endpoint3['title'] == titulo]
    title = film['title'].values[0]
    year = film['release_year'].values[0]
    score = film['popularity'].values[0]
    return f'La película {title} fue estrenada en el año {int(year)} con un score/popularidad de {score}'


@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo:str):
    film = df_endpoint4[df_endpoint4['title'] == titulo]
    valoraciones = film['vote_count'].values[0]
    promedio = film['vote_average'].values[0]
    año = film['release_year'].values[0]
    titulo = film['title'].values[0]
    if valoraciones < 2000:
        return 'no cumple con la cantidad de valoraciones minimas'
    else:
        return f'La película {titulo} fue estrenada en el año {int(año)}. La misma cuenta con un total de {valoraciones} valoraciones, con un promedio de {promedio}'


@app.get("/get_actor/{nombre}")
def get_actor(nombre:str):
    peliculas = df_endpoint5[df_endpoint5['actors'].apply(lambda actors: nombre in actors)]
    pelis = peliculas['title'].tolist()
    retorno = peliculas['return'].tolist()
    cantidad = len(pelis)
    ganancia = sum(retorno)
    promedio = ganancia/cantidad

    return f'El actor {nombre} ha participado en {cantidad} filmaciones, el mismo ha conseguido un retorno de {round(ganancia,3)} millones con un promedio de {round(promedio,3)} millones por filmación'


@app.get("/get_director/{nombre}")
def get_director(nombre:str):
    peliculas_director = df_endpoint6[df_endpoint6['director'] == nombre]
    
    if peliculas_director.empty:
        return f"No se encontraron películas para el director: {nombre}"
    
    retorno_total = peliculas_director['return'].sum()
    
    # Crear el DataFrame con la información requerida
    informacion_peliculas = peliculas_director[['title', 'release_date', 'return', 'budget', 'revenue']]
    
    return {
        'nombre_director': nombre,
        'retorno_total': retorno_total,
        'informacion_peliculas': informacion_peliculas
    }


df_reco = df[['title','overview','generos']]
df_reco['overview'] = df_reco['overview'].fillna('')


@app.get("/recomendacion/{titulo}")
def recomendacion( titulo ):
    # Inicializar el extractor RAKE
    rake = Rake()

    # Función para extraer palabras clave
    def extract_keywords(text):
        if isinstance(text,str):
            rake.extract_keywords_from_text(text)
            return ' '.join(rake.get_ranked_phrases())
        else:
            return []

    # Aplicar la extracción de palabras clave a la columna 'overview'
    df_reco['keywords'] = df_reco['overview'].apply(extract_keywords)

    # Crear una nueva columna combinando 'title', 'overview' y 'keywords'
    df_reco['combined_text'] = df_reco['title'] + ' ' + df_reco['keywords'] #'keywords'

    # Aplicar TfidfVectorizer sobre el texto combinado, limitando el número de características
    vect = TfidfVectorizer(stop_words='english', max_features=5000)  # Ajustar 'max_features' según tu memoria
    vect_matrix = vect.fit_transform(df_reco['combined_text'])

    # Usar NearestNeighbors para encontrar las películas más similares
    nn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
    nn_model.fit(vect_matrix)

    # Función de recomendación
    def recomendacion(titulo):
        # Verificar si el título está en el índice
        if titulo not in df_reco['title'].values:
            return f"No se encontró la película: {titulo}"

        # Obtener el índice de la película
        idx = df_reco[df_reco['title'] == titulo].index[0]

        # Encontrar los índices de las películas más cercanas
        distances, indices = nn_model.kneighbors(vect_matrix[idx], n_neighbors=6)

        # Obtener los títulos de las películas más similares
        similar_titles = df_reco['title'].iloc[indices[0][1:]]  # Excluir la misma película
        
        return similar_titles.tolist()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)