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
import nltk 
import pandas as pd
import nltk
from nltk.corpus import words

nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
nltk.download('stopwords')
print(stopwords.words('english'))

from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity


word_vectors = KeyedVectors.load_word2vec_format('D:/document/scrawl_instagram/data/GoogleNews-vectors-negative300.bin.gz', binary=True)

# virus
# corronavirus
# isolation 
#  social distance
#  face mask

grad graduation canon happy graduate queen
+-+-+
3
covid19_vector = word_vectors['coronavirus']

# # Tìm các từ liên quan bằng cách tính độ tương đồng cosine
similar_words = word_vectors.most_similar(positive=['coronavirus'], topn=10)

# # In ra danh sách các từ liên quan
# for word, score in similar_words:
#     print(f"{word}: {score}")


df.columns
text1 = word_vectors['coronavirus']
text2= word_vectors['pandemic']
similarity = cosine_similarity([text1],[text2])
similarity


print( (0.2112557 + 0.07796284 + 1.0000001 + 0.01261418 + 0.5741951 + 1.0000001) /6)
dosis booster coronavirus dari virus coronavirus


( 1.0 + 0.055 + 0.058 + 0.103 + 1.0 ) / 5  
print(0.5741951 * 2  + 1)


print((4/20) * 100 )
`print(1.5)
print(2.14/ 3)
df_hai = pd.read_csv("D:/document/get_data_twitter/config_data/data/clean_data/clean_data/data_4nam_Hai.csv")

def delete_column():
    df_hai.columns
    del df_hai['Unnamed: 0']
    del df_hai['index']
    del df_hai['type']
    del df_hai['id']
    del df_hai['dimensionsWidth']
    del df_hai['dimensionsHeight']
    del df_hai['displayUrl']
    del df_hai['childPosts']
    del df_hai['images']
    # del df_hai['ownerFullName']
    # del df_hai['ownerUsername']
    del df_hai['ownerId']
    del df_hai['isSponsored']
    del df_hai['taggedUsers']
    del df_hai['alt']
    del df_hai['videoUrl']
    del df_hai['videoViewCount']
    del df_hai['videoPlayCount']
    del df_hai['productType']
    del df_hai['videoDuration']
    del df_hai['paidPartnership']
    del df_hai['sponsors']
    
    del df_hai['data']
    del df_hai['extensions']
    del df_hai['status']

    return df_hai

df_hai = delete_column()
df = df_hai

df = df.dropna(subset=['caption'])

df['hashtags'] = df['hashtags'].apply(lambda x : x.replace('[','').replace(']','').replace("'","").replace(' ','')   )




df = df[0:500]


def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

# Áp dụng hàm remove_punctuation cho cột "caption"
df['caption'] = df['caption'].apply(remove_punctuation)



def get_time(df):
    
    # 2023-05-25 06:21:32+00:00
    time = df['timestamp'].str.split('-')
    df['year'] = time.str[0]
    df['month'] = time.str[1]
    df['day'] =  time.str[2].str.split(' ').str[0]
    return df

df = get_time(df)
    



english_words = set(words.words())
# Hàm kiểm tra xem một từ có phải là tiếng Anh không
def is_english_word(word):
    return word.lower() in english_words

# Hàm loại bỏ các từ không phải tiếng Anh khỏi một chuỗi
def remove_non_english_words(text):
    
    words_in_text = nltk.word_tokenize(text)
    english_words_only = [word for word in words_in_text if is_english_word(word)]
    cleaned_text = ' '.join(english_words_only)
    # print(cleaned_text)
    return cleaned_text.lower()

df['caption'] = df['caption'].apply(remove_non_english_words)


df['hashtags'] = df['hashtags'].apply(remove_non_english_words)







the world coming to a halt as a result of a pandemic leading to unparalleled in everyday life and the economy video
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


Discipline 💙💚💚❤️♥️
Follow - @aspirants_house 

#goodmorning #good #morning #love #discipline #persistence #bestfriends #beautiful #behavior #beaware #covid_19 #congratulations #covıd #corona #vaccinationdone✔️ #boosterdose #vaccinazioni💉 #god #upsc #up #ips #ias #mppsc #uppsc #bppsc #upscmotivation


def delete_stop_word(text):
    
    
    
    
    stop_words = set(stopwords.words('english'))
    
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    cleaned_text = ' '.join(filtered_sentence)
    
    return cleaned_text.lower()
    


df['caption']  = df['caption'].apply(delete_stop_word)
df['hashtags']  = df['hashtags'].apply(delete_stop_word)


def delete_one_word(text):
    # text = "r e l w e n g neither elaine traditional wedding wedding exactly like would party traditional wedding schedule day event fun casual memorable really us wedding party turned awesome vouch click real link see beautiful wedding photographer ceremony reception location celebrant entertainment acoustic guitar player brother elaine ukulele player hire flora flower elaine hair close friend bride emma cake elaine mother stationery designed elaine wedding dress linen dress groom engagement ring wedding bespoke elaine nelson made minimal"
    split = text.split(" ")
    cleaned_text = [word for word in split if len(word) > 1 ]
    cleaned_text = " ".join(cleaned_text)
    # print(cleaned_text)
    return cleaned_text.lower()


df['caption']  = df['caption'].apply(delete_one_word)
df['hashtags']  = df['hashtags'].apply(delete_one_word)

df = df[df['caption'].str.strip() != '']
df = df[df['hashtags'].str.strip() != '']



df['caption'] = df['caption'].str.replace('covid', 'coronavirus',case= False)
df['hashtags'] = df['hashtags'].str.replace('covid', 'coronavirus',case= False)
df = df.reset_index()





def calculator_max_similarity_caption_with_popular_word(popular_word):
    # Tạo danh sách để lưu thông tin về độ tương đồng trung bình
    # max_similarity_values = []
    max_similarity_caption_coronavirus = []
    max_similarity_caption_coronavirus_values = []
    
    # Lặp qua mỗi dòng trong DataFrame
    for index, row in df.iterrows():
        print(index)
        caption = row['caption']
        popular_word_vector = word_vectors[popular_word]
    
        # Tách các từ trong caption
        caption_words = caption.split()
        
        
        max_similarity = -1.0  # Để lưu độ tương đồng lớn nhất tìm thấy
        most_similar_caption = None  # Để lưu hashtag có độ tương đồng lớn nhất
        
        # Lặp qua từng từ trong caption và tính toán độ tương đồng với "coronavirus"
        for word in caption_words:
            if word in word_vectors:
                word_vector = word_vectors[word]
                similarity = cosine_similarity([word_vector], [popular_word_vector])

    
                # Nếu độ tương đồng lớn hơn độ tương đồng lớn nhất đã tìm thấy
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_caption = word
        
        # Lưu hashtag có độ tương đồng lớn nhất và giá trị tương đồng
        max_similarity_caption_coronavirus.append(most_similar_caption)
        max_similarity_caption_coronavirus_values.append(max_similarity)
    
    df['max_similarity_caption_coronavirus'] = max_similarity_caption_coronavirus
    df[f'calculator_max_similarity_caption_with_popular_word_{popular_word}'] = max_similarity_caption_coronavirus_values
    
    
    return df 

df = calculator_max_similarity_caption_with_popular_word('coronavirus')
df = calculator_max_similarity_caption_with_popular_word('virus')
df = calculator_max_similarity_caption_with_popular_word('isolation')
df = calculator_max_similarity_caption_with_popular_word('pandemic')
df = calculator_max_similarity_caption_with_popular_word('vaccine')
df = calculator_max_similarity_caption_with_popular_word('lockdown')


df['calculator_max_similarity_caption_with_popular_word_coronavirus'] = df['calculator_max_similarity_caption_with_popular_word_coronavirus'].apply(lambda x: x[0][0])

def plot_histograms_by_year_max_similarity(column_name):


    df_temp = df
    years = ["2020", "2021", "2022", "2023"]
    data_by_year = {}

    for year in years:
        data_by_year[year] = df_temp[df_temp['year'] == year][column_name]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Biểu đồ histogram cho các năm')

    colors = ['blue', 'green', 'red', 'purple']
    for i, year in enumerate(years):
        row = i // 2
        col = i % 2
        axes[row, col].hist(data_by_year[year], bins=10, color=colors[i], alpha=0.7)
        axes[row, col].set_xlabel('Giá trị tương đồng')
        axes[row, col].set_ylabel('Tần số')
        axes[row, col].set_title(f'Năm {year}')

    plt.tight_layout()
    plt.show()

plot_histograms_by_year_max_similarity('calculator_max_similarity_caption_with_popular_word_coronavirus')  # Thay đổi tên tệp CSV tương ứng



def process_and_analyze_data_ratio(column_name,threshold):
    df_temp = df
    years = ["2020", "2021", "2022", "2023"]
    for year in years:
        count_similarity_year = df_temp[df_temp['year'] == year]['year'].count()
        count_similarity_year_over_threshold = df_temp[(df_temp['year'] == year) & (df_temp[column_name] >= threshold)][column_name].count()
        ratio = (count_similarity_year_over_threshold/count_similarity_year  ) 
        print(f"Year {year}:")
        print(f"Count The Number Of Similarity The Year: {count_similarity_year}")
        print(f"ount The Number Of Similarity The Year Over Threshold {count_similarity_year_over_threshold}")
        print(f"Ratio: {ratio}")
        print()

process_and_analyze_data_ratio('calculator_max_similarity_caption_with_popular_word_coronavirus',0.7)  


df.columns

import pandas as pd
import matplotlib.pyplot as plt

def pie(column_name, threshold):
    df_temp = df
    years = ["2020", "2021", "2022", "2023"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Biểu đồ tròn thể hiện phần trăm giá trị tương đồng với threshold >= {threshold}")

    for i, year in enumerate(years):
        count_similarity_year = df_temp[df_temp['year'] == year]['year'].count()
        count_similarity_year_over_threshold = df_temp[(df_temp['year'] == year) & (df_temp[column_name] >= threshold)][column_name].count()
        ratio = count_similarity_year_over_threshold / count_similarity_year if count_similarity_year > 0 else None
        data = {
            'Category': [f'threshold >= {threshold}', 'Các Giá Trị Khác'],
            'Count': [count_similarity_year_over_threshold, count_similarity_year - count_similarity_year_over_threshold]
        }
        pie_df = pd.DataFrame(data)
        ax = axs[i // 2, i % 2]
        ax.pie(pie_df['Count'], labels=pie_df['Category'], autopct='%1.1f%%', startangle=90)
        ax.set_title(f"Year {year}")

    plt.show()
    
pie('calculator_max_similarity_caption_with_popular_word_coronavirus', 0.7)












def calculator_average_similarity_caption_with_popular_word(popular_word):
    # Tạo danh sách để lưu thông tin về độ tương đồng trung bình
    average_similarity_values = []
    
    # Lặp qua mỗi dòng trong DataFrame
    for index, row in df.iterrows():
        print(index)
        caption = row['caption']
        popular_word_vector = word_vectors[popular_word]
    
        # Tách các từ trong caption
        caption_words = caption.split()
        
        total_similarity = -1
        count = 0
        
        # Lặp qua từng từ trong caption và tính toán độ tương đồng với "coronavirus"
        for word in caption_words:
            if word in word_vectors:
                word_vector = word_vectors[word]
                similarity = cosine_similarity([word_vector], [popular_word_vector])
                total_similarity += similarity
                count += 1
        
        # Tính độ tương đồng trung bình (nếu có ít nhất một từ)
        if count > 0:
            average_similarity = total_similarity / count
        else:
            average_similarity = 0.0
        
        average_similarity_values.append(average_similarity)

    # Thêm cột mới vào DataFrame chứa thông tin về độ tương đồng trung bình
    df[f'average_similarity_caption_{popular_word}'] = average_similarity_values
    
    return df


df = calculator_average_similarity_caption_with_popular_word('coronavirus')
df = calculator_average_similarity_caption_with_popular_word('virus')
df = calculator_average_similarity_caption_with_popular_word('isolation')
df = calculator_average_similarity_caption_with_popular_word('pandemic')
df = calculator_average_similarity_caption_with_popular_word('vaccine')
df = calculator_average_similarity_caption_with_popular_word('lockdown')


def calculator_average_similarity_caption_with_popular_word_threshold(popular_word, threshold=0.3):
    # Tạo danh sách để lưu thông tin về độ tương đồng trung bình
    average_similarity_values = []
    
    # Lặp qua mỗi dòng trong DataFrame
    for index, row in df.iterrows():
        print(index)
        caption = row['caption']
        popular_word_vector = word_vectors[popular_word]
    
        # Tách các từ trong caption
        caption_words = caption.split()
        
        total_similarity = -1
        count = 0
        
        # Lặp qua từng từ trong caption và tính toán độ tương đồng với "coronavirus"
        for word in caption_words:
            if word in word_vectors:
                word_vector = word_vectors[word]
                similarity = cosine_similarity([word_vector], [popular_word_vector])
                if similarity >= threshold:
                    total_similarity += similarity
                    count += 1
        
        # Tính độ tương đồng trung bình (nếu có ít nhất một từ)
        if count > 0:
            average_similarity = total_similarity / count
        else:
            average_similarity = 0.0
        
        average_similarity_values.append(average_similarity)

    # Thêm cột mới vào DataFrame chứa thông tin về độ tương đồng trung bình
    df[f'average_similarity_caption_{popular_word}_threshold_{threshold}'] = average_similarity_values
    
    return df

df = calculator_average_similarity_caption_with_popular_word_threshold('coronavirus', threshold=0.3)
df = calculator_average_similarity_caption_with_popular_word_threshold('virus', threshold=0.3)
df = calculator_average_similarity_caption_with_popular_word_threshold('isolation', threshold=0.3)
df = calculator_average_similarity_caption_with_popular_word_threshold('pandemic', threshold=0.3)
df = calculator_average_similarity_caption_with_popular_word_threshold('vaccine', threshold=0.3)
df = calculator_average_similarity_caption_with_popular_word_threshold('lockdown', threshold=0.3)

df = calculator_average_similarity_caption_with_popular_word_threshold('coronavirus', threshold=0.5)
df = calculator_average_similarity_caption_with_popular_word_threshold('virus', threshold=0.5)
df = calculator_average_similarity_caption_with_popular_word_threshold('isolation', threshold=0.5)
df = calculator_average_similarity_caption_with_popular_word_threshold('pandemic', threshold=0.5)
df = calculator_average_similarity_caption_with_popular_word_threshold('vaccine', threshold=0.5)
df = calculator_average_similarity_caption_with_popular_word_threshold('lockdown', threshold=0.5)


df = calculator_average_similarity_caption_with_popular_word_threshold('coronavirus', threshold=0.7)
df = calculator_average_similarity_caption_with_popular_word_threshold('virus', threshold=0.7)
df = calculator_average_similarity_caption_with_popular_word_threshold('isolation', threshold=0.7)
df = calculator_average_similarity_caption_with_popular_word_threshold('pandemic', threshold=0.7)
df = calculator_average_similarity_caption_with_popular_word_threshold('vaccine', threshold=0.7)
df = calculator_average_similarity_caption_with_popular_word_threshold('lockdown', threshold=0.7)


def calculator_ratio_similarity_caption_with_popular_word(popular_word,threshold):
    # Tạo danh sách để lưu thông tin về độ tương đồng trung bình
    average_similarity_values = []
    count_over_threshold = []
    len_caption = []
    # Lặp qua mỗi dòng trong DataFrame
    for index, row in df.iterrows():
        print(index)
        caption = row['caption']
        popular_word_vector = word_vectors[popular_word]
    

        caption_words = caption.split()
        
        
        

        
        total_similarity = 0.0
        count = 0
        count_threshold= 0 
        # Lặp qua từng từ trong caption và tính toán độ tương đồng với "coronavirus"
        for word in caption_words:
            if word in word_vectors:
                word_vector = word_vectors[word]
                similarity = cosine_similarity([word_vector], [popular_word_vector])
                if similarity > threshold:  # Kiểm tra độ tương đồng > 0.3
                    total_similarity += 1  # Tăng biến đếm lên 1
                    count_threshold +=1
                count += 1
        
        # Tính độ tương đồng trung bình (nếu có ít nhất một từ)
        if count > 0:
            average_similarity = total_similarity / count
        else:
            average_similarity = 0.0
        
        average_similarity_values.append(average_similarity)
        count_over_threshold.append(count_threshold)
        len_caption.append(count)
    # Thêm cột mới vào DataFrame chứa thông tin về độ tương đồng trung bình
    df[f'ratio_similarity_caption_with_popular_word_{popular_word}_{threshold}'] = average_similarity_values
    df[f'ratio_similarity_caption_with_popular_word_{popular_word}_{threshold}_count'] = count_over_threshold
    df[f'ratio_similarity_caption_with_popular_word_{popular_word}_{threshold}_len_caption'] = len_caption
    return df



df = calculator_ratio_similarity_caption_with_popular_word('coronavirus',0.3)
df = calculator_ratio_similarity_caption_with_popular_word('virus',0.3)
df = calculator_ratio_similarity_caption_with_popular_word('isolation',0.3)
df = calculator_ratio_similarity_caption_with_popular_word('pandemic',0.3)
df = calculator_ratio_similarity_caption_with_popular_word('vaccine',0.3)
df = calculator_ratio_similarity_caption_with_popular_word('lockdown',0.3)


df = calculator_ratio_similarity_caption_with_popular_word('coronavirus',0.5)
df = calculator_ratio_similarity_caption_with_popular_word('virus',0.5)
df = calculator_ratio_similarity_caption_with_popular_word('isolation',0.5)
df = calculator_ratio_similarity_caption_with_popular_word('pandemic',0.5)
df = calculator_ratio_similarity_caption_with_popular_word('vaccine',0.5)
df = calculator_ratio_similarity_caption_with_popular_word('lockdown',0.5)


df.columns
df = calculator_ratio_similarity_caption_with_popular_word('coronavirus',0.7)


import matplotlib.pyplot as plt
def plot_histograms_by_year_max_similarity(column_name):


    df_temp = df
    years = ["2020", "2021", "2022", "2023"]
    data_by_year = {}

    for year in years:
        data_by_year[year] = df_temp[df_temp['year'] == year][column_name]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Biểu đồ histogram cho các năm')

    colors = ['blue', 'green', 'red', 'purple']
    for i, year in enumerate(years):
        row = i // 2
        col = i % 2
        axes[row, col].hist(data_by_year[year], bins=10, color=colors[i], alpha=0.7)
        axes[row, col].set_xlabel('Giá trị tương đồng')
        axes[row, col].set_ylabel('Tần số')
        axes[row, col].set_title(f'Năm {year}')

    plt.tight_layout()
    plt.show()

plot_histograms_by_year_max_similarity('ratio_similarity_caption_with_popular_word_coronavirus_0.3') # Thay đổi tên tệp CSV tương ứng

df.columns












def calculator_average_similarity_hashtags_with_popular_word_threshold(popular_word, threshold=0.3):
    # Tạo danh sách để lưu thông tin về độ tương đồng trung bình
    average_similarity_values = []
    count_over_threshold = []
    # Lặp qua mỗi dòng trong DataFrame
    for index, row in df.iterrows():
        print(index)
        hashtags = row['hashtags']
        popular_word_vector = word_vectors[popular_word]
        # Tách các từ trong caption
        hashtag_words = hashtags.split()

        total_similarity = 0.0
        count = 0
        count_threshold = 0
        # Lặp qua từng từ trong caption và tính toán độ tương đồng với "coronavirus"
        for word in hashtag_words:
            if word in word_vectors:
                word_vector = word_vectors[word]
                similarity = cosine_similarity([word_vector], [popular_word_vector])
                if similarity >= threshold:
                    total_similarity += similarity
                    count_threshold+=1
                count += 1
        
        # Tính độ tương đồng trung bình (nếu có ít nhất một từ)
        if count > 0:
            average_similarity = total_similarity / count
        else:
            average_similarity = 0.0
        
        average_similarity_values.append(average_similarity)
        count_over_threshold.append(count_threshold)

    # Thêm cột mới vào DataFrame chứa thông tin về độ tương đồng trung bình
    df[f'average_similarity_hashtags_{popular_word}_threshold_{threshold}'] = average_similarity_values
    df[f'average_similarity_hashtags_{popular_word}_threshold_{threshold}_count'] = count_over_threshold
    
    return df

df = calculator_average_similarity_hashtags_with_popular_word_threshold('coronavirus', threshold=0.3)




def calculator_max_similarity_caption_with_hashtag():
    

    # Tạo các danh sách để lưu thông tin về hashtag có similarity lớn nhất
    max_similarity_caption = []
    max_similarity_hashtag = []
    max_similarity_caption_hashtag_values = []
    
    
    # Lặp qua mỗi dòng trong DataFrame
    for index, row in df.iterrows():
       
        caption = row['caption']
        hashtag = row['hashtags']
        # coronavirus = word_vectors["coronavirus"]    
        # Tách các hashtag ra thành danh sách
        caption_list = caption.split()
        hashtag_list = caption.split()
        
        max_similarity = -1.0  # Để lưu độ tương đồng lớn nhất tìm thấy
        most_similar_hashtag = None  # Để lưu hashtag có độ tương đồng lớn nhất
        most_similar_caption = None  # Để lưu hashtag có độ tương đồng lớn nhất
        # Lặp qua từng hashtag và tính toán độ tương đồng
        for caption in caption_list:
            print(index)
            for hashtag in hashtag_list:
                
                if(caption in word_vectors and hashtag in word_vectors ):
                    text_caption = word_vectors[caption]
                    text_hashtag = word_vectors[hashtag]
                    similarity = cosine_similarity([text_caption],[text_hashtag])
                    
                    # Nếu độ tương đồng lớn hơn độ tương đồng lớn nhất đã tìm thấy
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_hashtag = hashtag
                        most_similar_caption = caption
    
        
        max_similarity_hashtag.append(most_similar_hashtag)
        max_similarity_caption.append(most_similar_caption)
        max_similarity_caption_hashtag_values.append(max_similarity)

# Thêm cột mới vào DataFrame chứa thông tin về hashtag có độ tương đồng lớn nhất và giá trị tương đồng
    df['max_similarity_hashtag'] = max_similarity_hashtag
    df['max_similarity_caption'] = max_similarity_caption
    df['max_similarity_caption_hashtag_values'] = max_similarity_caption_hashtag_values
    
    
    return df 


df = calculator_max_similarity_caption_with_hashtag()



# df = df[df['average_similarity_caption_coronavirus'] !=0]

def process_and_analyze_data(column_name,threshold):
    df_temp  = df
    # Tạo DataFrame mới
    new_df_temp = {'year': df_temp['year'], column_name: [val[0][0] for val in df_temp[column_name]]}
    df_temp = pd.DataFrame(new_df_temp)



    years = ["2020", "2021", "2022", "2023"]

    for year in years:
        count_similarity_year = df_temp[df_temp['year'] == year]['year'].count()
        count_similarity_year_over_threshold = df_temp[(df_temp['year'] == year) & (df_temp[column_name] >= threshold)][column_name].count()

        if count_similarity_year_over_threshold == 0:
            ratio = None
        else:
            ratio = (count_similarity_year_over_threshold/count_similarity_year  ) 

        print(f"Year {year}:")
        print(f"Count The Number Of Similarity The Year: {count_similarity_year}")
        print(f"ount The Number Of Similarity The Year Over Threshold {count_similarity_year_over_threshold}")
        print(f"Ratio: {ratio}")
        print()

# Gọi hàm với tệp dữ liệu của bạn
process_and_analyze_data('average_similarity_caption_coronavirus',0.3)  


df.columns

import matplotlib.pyplot as plt
import pandas as pd

def plot_histograms_by_year(column_name):

    df_temp = df[df[column_name] != -1]
    # df_temp = df
    # Tạo DataFrame mới
    new_df_temp = {'year': df_temp['year'], column_name: [val[0][0] for val in df_temp[column_name]]}
    df_temp = pd.DataFrame(new_df_temp)

    # Lọc dữ liệu cho từng năm
    years = ["2020", "2021", "2022", "2023"]
    data_by_year = {}

    for year in years:
        data_by_year[year] = df_temp[df_temp['year'] == year][column_name]

    # Khởi tạo subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Biểu đồ histogram cho các năm')

    # Vẽ biểu đồ histogram cho từng năm
    colors = ['blue', 'green', 'red', 'purple']

    for i, year in enumerate(years):
        row = i // 2
        col = i % 2

        axes[row, col].hist(data_by_year[year], bins=10, color=colors[i], alpha=0.7)
        axes[row, col].set_xlabel('Giá trị tương đồng')
        axes[row, col].set_ylabel('Tần số')
        axes[row, col].set_title(f'Năm {year}')

    # Đảm bảo không còn subplot trống
    plt.tight_layout()
    plt.show()

# Gọi hàm với tên tệp dữ liệu của bạn
plot_histograms_by_year('average_similarity_caption_coronavirus')  # Thay đổi tên tệp CSV tương ứng




# virus
# corronavirus
# isolation 
#  social distance
#  face mask

# comment : -> texxt -> max, tb 
# phan tram 

# dem ty le phan tram so tu` co do tuong quan > 0.3 trong tung caption


# Lưu DataFrame thành tệp CSV
df.to_csv('D:/document/get_data_twitter/data/data_official/data_official_mai_lam_tiep.csv', index=False)  # Thay 'ten_file.csv' bằng tên tệp bạn muốn lưu



## covid /covid 19-> coronavirus 
## text in text, hashtag with .... 6 tuwf
## tach' hashtag 
# plot từng từ / tổng
# threshold : 0.3 , 0.5, 0.7
# biểu đò phần trăm 

df.columns

import re
import re

def split_words(input_string):
    # Tách các từ bắt đầu bằng chữ cái viết thường
    words = re.findall(r'\b[a-z][a-z ]*\b', input_string)
    print(words)
    # return ' '.join(words)
split_words('blackandwhitephotography')




input_strings = ["classof2021", "blackandwhitephotography"]

for input_string in input_strings:
    result = split_words(input_string)
    print(result)








input_string = "classof2021"

# Tìm vị trí của ký tự đầu tiên không phải là chữ cái
split_index = next((i for i, c in enumerate(input_string) if not c.isalpha()), len(input_string))

# Tách chuỗi thành hai phần
word1 = input_string[:split_index]
word2 = input_string[split_index:]

print(word1)  # Output: "class"
print(word2)  # Output: "of2021"




english_words = set(words.words())
# Hàm kiểm tra xem một từ có phải là tiếng Anh không
def is_english_word(word):
    return word.lower() in english_words

# Hàm loại bỏ các từ không phải tiếng Anh khỏi một chuỗi
def remove_non_english_words(text):
    
    words_in_text = nltk.word_tokenize('text hello abc')
    print(words_in_text)
    
    
    
    english_words_only = [word for word in words_in_text if is_english_word(word)]
    english_words_only
    
    
    cleaned_text = ' '.join(english_words_only)
    # print(cleaned_text)
    return cleaned_text.lower()

df['caption'] = df['caption'].apply(remove_non_english_words)




input_string = "whiteblackandwhite"
result1 = ' '.join(input_string)
print(result1)


for word in result1:
    str_eng += word
    str_eng = str_eng.replace(' ','')
    print(str_eng)
    # print(word)
    if(len(str_eng)>2):
        if is_english_word(str_eng):
            list_eng.append(str_eng)
            print(str_eng)
            
    
    
    
input_string = "whiteblackandhello"


def split_camel_case_to_words(text):
    
    # text = "whiteblackandhello"
    list_eng = []
    total_str_eng = ''
    str_eng = ""
    i = 0
    index = 0 
    
    while i < len(text)  :
        # print(i)
        str_eng += text[i]
        if len(str_eng) > 2:
            if is_english_word(str_eng):
                list_eng.append(str_eng)
                total_str_eng += str_eng + " "
                print(total_str_eng)
                index = i + 1
                
        i+=1
        
        if( i == (len(text))):
           i = index 
           # print(index)
           # print(input_string[index])
           str_eng = ""
    # print(total_str_eng)
           
    return total_str_eng
    
df['hashtag_test'] = df['hashtags'].apply(split_camel_case_to_words)


(df.loc[1,'hashtags'])

df.columns


    text = "whiteblackandhello,good"
    text_split = text.split(',')
    for word in text_split:
        print(word)

        list_eng = []
        total_str_eng = ''
        str_eng = ""
        i = 0
        index = 0 
        
        while i < len(word)  :
            # print(i)
            str_eng += word[i]
            if len(str_eng) > 2:
                if is_english_word(str_eng):
                    list_eng.append(str_eng)
                    total_str_eng += str_eng + " "
                    index = i + 1
                    
            i+=1
            
            if( i == (len(word))):
               i = index 
               str_eng = ""
        print(total_str_eng)
           
    # return total_str_eng




def extract_english_words(origin_text):
    

    # origin_text = "whiteblackandhello,good"
    split_text = origin_text.split(',')
    output = ''
    list_eng = []
    for text in split_text:
        print(text)

        total_str_eng = ''
        str_eng = ""
        i = 0
        index = 0 
        while i < len(text)  :
            str_eng += text[i]
            if len(str_eng) > 2:
                if is_english_word(str_eng):
                    list_eng.append(str_eng)
                    index = i + 1
            i+=1
            if( i == (len(text))):
               i = index 
               str_eng = ""
    return list_eng
    
output = extract_english_words('graduation,blackandwhitephotography,canon,classof2021')
output 

df['hashtag_test'] = df['hashtags'].apply(extract_english_words)
    

graduation,classof2021,blackandwhitephotography,canon,canonaustralia,canoneosr,covıd,happy,graduate,gradstudent,queen,teacherlife,teachersofinstagram
    

text = "blackandwhitephotography"
list_eng = []
total_str_eng = ''
str_eng = ""
i = 0
index = 0 

while i < len(text)  :
    # print(i)
    str_eng += text[i]
    print(i)
    print(text[i])
    print(str_eng)
    
    if len(str_eng) > 2:
        if is_english_word(str_eng):
            list_eng.append(str_eng)
            total_str_eng += str_eng + " "
            print(str_eng)
            index = i + 1  # 5
        # else:
        #     break
            
    i+=1
    
    if( i == (len(text))):
       i = index 
       str_eng = ""
       list_eng.append(str_eng)
print(list_eng)


# pip install pyenchant
import enchant
import splitter

splitter.split('💉')
20220724,hongkong,covidtravel,香港,homekong,香港鐵路博物館,hongkongrailwaymuseum,🚂,🇭🇰,hkgirl,大埔,taipo,maskup,railway,唔適合呀嘛真係無計
20230510,🌞,25歳,夏日,💉,covidvacccine,新型コロナ,ワクチン接種,ファイザー,6回目,有明スポーツセンター,水再生センター,エントランス,錦鯉

import pandas as pd
import splitter

# Tạo một DataFrame mẫu
data = {'hashtag': ['coronavirus,pandemic,health', 'data,analysishello', 'python,programming,blackandwhitephotography']}
df = pd.DataFrame(data)



df = df[0:200]

def split_hashtags(hashtag_string):
    data = hashtag_string.split(',')
    result = []

    for text in data:
        x = splitter.split(text)
        result.extend(x)
    
    return result


df['words'] = df['hashtags'].apply(split_hashtags)

loyola,loyolite,loyolites😎,vitchennai,madras,coronamemes,srmuniversity,itz,modelunitednations,chennailife,chennaidays,ssncollege,chennailove,covidmemes,loyolites,lastbenchers,chennaiphotographer,kgf,unitednations,loyolaalumni,alumni,trendies,ethiraj,chennaidiaries,chennaievents,modelunitednation,chennaimuncircuit,chennainews,chennaitimes,stellamariscollege

x = ['x']
y = ['u']
print(x+y)


graduation,classof2021,blackandwhitephotography,canon,canonaustralia,canoneosr,covıd,happy,graduate,gradstudent,queen,teacherlife,teachersofinstagram
['graduation', 'class', 'of', '2021', 'black', 'and', 'white', 'photography', 'canon', 'canon', 'austral', 'Ia', 'canon', 'Eos', 'r', 'happy', 'graduate', 'grads', 'Tu', 'dent', 'queen', 'teacher', 'life', 'teachers', 'of', 'inst', 'Agra', 'm']