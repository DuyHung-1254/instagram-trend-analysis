import re
import os
import pandas as pd
from nltk.corpus import words
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from langdetect import detect
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import nltk 
import enchant
import splitter
from nltk.corpus import words
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
nltk.download('stopwords')
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity



def delete_column(df):
    
    # del df['Unnamed: 0']
    del df['index']
    del df['type']
    del df['id']
    del df['dimensionsWidth']
    del df['dimensionsHeight']
    del df['displayUrl']
    del df['childPosts']
    del df['images']
    del df['ownerId']
    del df['isSponsored']
    del df['taggedUsers']
    del df['alt']
    del df['videoUrl']
    del df['videoViewCount']
    del df['videoPlayCount']
    del df['productType']
    del df['videoDuration']
    del df['paidPartnership']
    del df['sponsors']
    
    # del df['data']
    # del df['extensions']
    # del df['status']
    
    del df['mentions']
    del df['commentsCount']
    del df['firstComment']
    del df['latestComments']
    del df['shortCode']
    del df['likesCount']
    del df['ownerFullName']
    del df['ownerUsername']
    del df['locationName']
    del df['locationId']


    return df




def get_time(df):
    
    # 2023-05-25 06:21:32+00:00
    time = df['timestamp'].str.split('-')
    df['year'] = time.str[0]
    df['month'] = time.str[1]
    df['day'] =  time.str[2].str.split(' ').str[0]
    return df

def delete_stop_word(text):
    stop_words = set(stopwords.words('english'))
    
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    cleaned_text = ' '.join(filtered_sentence)
    
    return cleaned_text.lower()
def split_hashtags(hashtag_string):
    data = hashtag_string.split(',')
    global index_counter
    result = []
    for text in data:
        # print(text)
        x = splitter.split(text)
        result.extend(x)
        
    print("  bộ mới ",index_counter)
    index_counter+=1
    
    return result 




    
    

def replace_covid_with_coronavirus(df,column_name):
 
    df[column_name] = df[column_name].str.replace("covid", 'coronavirus', case=False)
    # df[column_name] = df[column_name].str.replace("corona",'coronavirus',case = False)
    
    return df

def replace_corona(column_name):
    return re.sub(r'\bcorona\b', 'coronavirus', column_name, flags=re.IGNORECASE)

# Áp dụng hàm replace_corona cho cột trong DataFrame

def list_to_string(my_list):
    result_str = ', '.join(my_list)  
    return result_str
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

english_words = set(words.words())
def is_english_word(word):
    return word.lower() in english_words
def remove_non_english_words(text):
    
    words_in_text = nltk.word_tokenize(text)
    english_words_only = [word for word in words_in_text if is_english_word(word)]
    cleaned_text = ' '.join(english_words_only)
    return cleaned_text.lower()
def remove_hashtags(str_input):
    words = str_input.split()
    filtered_words = [word for word in words if '#' not in word]
    new_str = ' '.join(filtered_words)
    return new_str
def delete_one_word(text):
    split = text.split(" ")
    cleaned_text = [word for word in split if len(word) > 1 ]
    cleaned_text = " ".join(cleaned_text)
    return cleaned_text.lower()



def remove_rows_with_specific_words(df,column_name, specific_words):
    mask = df[column_name].apply(lambda x: not any(word in x for word in specific_words))
    df = df[mask]
    return df


specific_words = ['si', 'de', 'en', 'legal', 'toi', 'maestri', 'di', 'na', 'nu', 'ama', 'wa', 'kan', 'ta', 'lemon', 'ni', 'tu', 'pon', 'nak', 'da', 'flora', 'planeta', 'pandemia', 'das', 'nadir', 'ne', 'bu', 'em', 'se', 'ha', 'mi', 'ser', 'familia', 'sa', 'kang', 'mo', 'la', 'al', 'mil']



def remove_short_word(df,column_name):
    mask = df[column_name].apply(lambda x: len(x.split()) > 5 )  # Kiểm tra độ dài của từng hashtag
    df = df[mask]  # Loại bỏ các dòng theo điều kiện
    return df




def precrssing_hashtags(df):
    df = df.dropna(subset=['hashtags'])
    df = df.reset_index()
    df['hashtags'] = df['hashtags'].apply(lambda x : x.replace('[','').replace(']','').replace("'","").replace(' ','')   )
    # df['hashtags'] = df['hashtags'].apply(lambda x : x.replace(',',' '))
    # df['hashtags'] = df['hashtags'].apply(remove_punctuation) # covid-19 -> covid19
    df['hashtags'] = df['hashtags'].str.replace("covid-19", 'covid', case=False) # covid thì sẽ k bị xóa
    df['hashtags'] = df['hashtags'].str.replace("covid19", 'covid', case=False) # covid thì sẽ k bị xóa
    df['hashtags'] = df['hashtags'].str.replace("covid_19", 'covid', case=False) 
    df['hashtags'] = df['hashtags'].str.replace("covid", 'corona', case=False)
    # df = df.drop(250) # của bộ cũ 
    f
    
    df = df.drop(16389) # bộ mới
    df = df.drop(20975)
    
    
    df = df[df['hashtags'].str.strip() != ''] 
    df = df.reset_index()
    index_counter = 1
    df['hashtags'] = df['hashtags'].apply(split_hashtags) # covid sẽ bị tách
    df['hashtags'] = df['hashtags'].apply(list_to_string) 
    df['hashtags']  = df['hashtags'].apply(delete_one_word)
    df['hashtags']  = df['hashtags'].apply(delete_stop_word)
    df['hashtags'] = df['hashtags'].apply(remove_non_english_words) 
    df['hashtags'] = df['hashtags'].str.replace("corona", 'coronavirus', case=False)
    df = df[df['hashtags'].str.strip() != ''] 
    df = remove_short_word(df,'hashtags')
    
    return df

def precrssing_caption(df):
    df = df.dropna(subset=['caption'])
    
    df['caption'] = df['caption'].apply(remove_hashtags) 
    
    df['caption'] = df['caption'].apply(remove_punctuation) # covid-19 -> covid19
    df['caption']  = df['caption'].apply(delete_one_word)
    df['caption']  = df['caption'].apply(delete_stop_word)
    
    df['caption'] = df['caption'].str.replace("covid19", 'covid', case=False) # covid thì sẽ k bị xóa
    df['caption'] = df['caption'].str.replace("coronavirus", 'covid', case=False) # covi
    df['caption'] = df['caption'].apply(remove_non_english_words)  # nó sẽ xóa coronavirus
    df = remove_short_word(df,'caption')
    df = remove_short_word(df,'hashtags')
    
    
    
    df = df[df['caption'].str.strip() != ''] 
    df['caption'] = df['caption'].apply(replace_corona) # đổi coroa thành coronavirus( corona là bia )
    # df = replace_covid_with_coronavirus(df,'caption') # covid to coronavirus
    df['caption'] = df['caption'].str.replace("covid", 'coronavirus', case=False)
    
    
    
    
    
    
    # result = df[df['hashtags'].str.contains('covid', case=False)]
    result1 = df[df['hashtags'].str.contains('coronavirus', case=False)]
    result2 = df[df['hashtags'].str.contains('flllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll', case=False)]
    return df
    

def main():
    df_hai = pd.read_csv("D:/document/get_data_twitter/config_data/data/clean_data/clean_data/data_4nam_Hai.csv")
    # df = pd.read_csv("D:/document/scrawl_instagram/data/have_not_processing/csv/data_have_not_processing.csv")
    # df = df[0:100]
    
    
    df = pd.read_csv("D:/document/scrawl_instagram/data/have_not_processing/csv/have_not_delete_hand_new_dataset_deleting_3.csv")
    df = get_time(df)
    df = delete_column(df)
    df = df[20970:]
    df1 = df[:20]
    df = precrssing_caption(df)
    df = precrssing_hashtags(df)


    df123 = df[950:]
    return df


    df = df.drop_duplicates(subset = ['caption'])


df = main()

# df.to_csv('D:/document/scrawl_instagram/data/processed/data.csv', index=False)  



df_random_2020= df[df['year'] == 2020]
df_rd_2020 = df_rd_2020.sample(n = 320, random_state = 42)

df_random_2021 = df[df['year'] == 2021]
result_df_random_2021 = df_random_2021[df_random_2021['caption'].str.contains('coronavirus', case=False)]



df_rd_2021 = df_random_2021.sample(n= 320, random_state = 42)


df_random_2022= df[df['year'] == 2022]
df_rd_2022 = df_random_2022.sample(n= 320, random_state = 42)


df_random_2023 = df[df['year'] == 2023]
df_rd_2023 = df_random_2023.sample(n= 320, random_state = 42)




df_random_450 = pd.concat([df_rd_2020,df_rd_2021,df_rd_2022,df_rd_2023])


Drakes “Dark Lane Demo Tapes” Came Out 3 Years Ago

What’s your fav track⁉️

@champagnepapi

#drake #champagnepapi #lilwayne #darklanedemotapes #21savage #herloss #playboicarti #amg #eminem #covıd #quarantine #2020 #hiphop #certifiedloverboy #clb #toronto #drakelyrics #drakequotes #youngmoney #ovosound #ovo #octobersveryown #drakerelated #honestlynevermind #jacobandco #streetwear #hiphopculture #hiphopfashion #toosieslide #weoutside

