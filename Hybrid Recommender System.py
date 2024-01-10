# Hybrid Recommender System

# ID'si verilen kullaniclar icin item-based ve user-based recommender yontemlerini kullanarak tahmin yapalim.
# user-based ve item-based'den 5'er oneri alarak toplamda 10 oneriyi iki modelden yapalim.

## User-Based Recommendation
# Verinin Hazirlanmasi
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)

movie = pd.read_csv('datasets/movie.csv')
rating = pd.read_csv('datasets/rating.csv')

movie.head()
movie.shape
rating.head()
rating.shape

rating['userId'].nunique()

df = movie.merge(rating, how='left', on='movieId')
df.head()
df.shape

# Her bir film icin toplam kac oy kullanilmis hesapliyoruz.
# Oy sayisi 1000'den az ise veri setinden cikaracagiz.
comment_counts = pd.DataFrame(df['title'].value_counts())
comment_counts.head()
rare_movies = comment_counts[comment_counts['title'] <= 1000].index
common_movies = df[~df['title'].isin(rare_movies)]
common_movies.shape

# index'de userID'lerin, kolonlarda film isimlerinin ve deger olarak ratinglerin bulundugu bir dataframe icin pivot table olusturuyoruz.
user_movie_df = common_movies.pivot_table(index=['userId'], columns=['title'], values='rating')
user_movie_df.head()

# Yukarida yaptigimiz tum islemleri fonksiyonlastiralim
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie.csv')
    rating = pd.read_csv('datasets/rating.csv')
    df = movie.merge(rating, how='left', on='movieId')
    comment_counts = pd.DataFrame(df['title'].value_counts())
    rare_movies = comment_counts[comment_counts['title'] <= 1000].index
    common_movies = df[~df['title'].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=['userId'], columns=['title'], values='rating')
    return user_movie_df


user_movie_df = create_user_movie_df()

# Oneri yapilacak kullanicilarin izledigi filmlerin belirlenmesi
random_user = 108170
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()
random_user_df.shape

# Secilen kullanicinin oy kullandigi filmleri movies_watched adli bir listeye atalim
movies_watched = random_user_df.columns[random_user_df.notna().any()].to_list()

# Ayni filmleri izleyen diger kullanicilarin verisine ve ID'sine eriselim
# Secilen kullanicinin izledigi filmlere ait sutunlari user_movie_df'ten secelim
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()

# Herbir kullanicinin secili user'in izledigi filmlerin kacini izledigini bulalim
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

# Secilen kullanicilarin oy verdigi filmlerin %60 ve ustunu izleyenleri benzer kullanicilar olarak kabul ediyoruz
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count['movie_count'] > perc]['userId']
len(users_same_movies)

# Oneri yapilacak kullanici ile en benzer kullanicilarin belirlenmesi
final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.head()

# Kullanicilarin birbirleri ile olan korelasyonuna bakalim
corr_df = final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

corr_df[corr_df["user_id_1"] == random_user]

# Secili kullanimiz ile yuksek korelasyona sahip kullanicilari bulalim (0.65'in uzerinde)
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.head()

# top_users dataframe’ini rating veri seti ile merge edelim
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings["userId"].unique()
top_users_ratings.head()

# Weighted Average Recommendation Score'u hesaplayalim
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

# Film id'si ile her bir filme ait tum kullanicilarin weighted rating'lerinin ortalama degerini iceren yeni bir df olusturuyoruz
recommendation_df = top_users_ratings.groupby('movieId').agg({'weighted_rating':'mean'})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()

# weighted_rating'i 3.5'ten buyuk olan ilk 10 film
movies_to_be_recommend = recommendation_df[recommendation_df['weighted_rating'] > 3.5].sort_values('weighted_rating', ascending=False)
movies_to_be_recommend.head()

movies_to_be_recommend.merge(movie[['movieId', 'title']])['title'].head()

## Item-Based Recommendation
# Verinin Hazirlanmasi
user = 108170
movie = pd.read_csv('datasets/movie.csv')
rating = pd.read_csv('datasets/rating.csv')

# Oneri yapilacak kullanicinin 5 puan verdigi filmlerden puani en guncel olan filmin id'sini alalim
movie_id = rating[(rating['userId'] == user) & (rating['rating'] == 5.0)].sort_values(by='timestamp', ascending=False)['movieId'][0:1].values[0]
movie[movie["movieId"] == movie_id]["title"].values[0]

# User based recommendation bolumunde olusturdugumuz user_movie_df'ni secilen film id'sine gore filtreleyelim
movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]
movie_df

# Filtrelenen df'i kullanarak secili filmle diger filmlerin korelasyonuna bakalim
user_movie_df.corrwith(movie_df).sort_values(ascending=False).head()

# Son iki adimi fonksiyonlastiralim
def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)

# Secili filmin kendisi disinda ilk 5 filmi oneri olarak verelim
movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)
movies_from_item_based[1:6] # 0:6'da filmin kendisi var

# 'My Science Project (1985)',
# 'Mediterraneo (1991)',
# 'Old Man and the Sea,
# The (1958)',
# 'National Lampoon's Senior Trip (1995)',
# 'Clockwatchers (1997)']
