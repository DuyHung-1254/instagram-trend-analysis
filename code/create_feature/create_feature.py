import re
import os
import numpy as np
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
import matplotlib.pyplot as plt
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
nltk.download('stopwords')

from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

word_vectors = KeyedVectors.load_word2vec_format('D:/document/scrawl_instagram/data/GoogleNews-vectors-negative300.bin.gz', binary=True)



    

def calculator_max_similarity_caption_with_popular_word(popular_word):
    max_similarity_caption_coronavirus = []
    max_similarity_caption_coronavirus_values = []
    
    for index, row in df.iterrows():
        print(index)
        caption = row['caption']
        popular_word_vector = word_vectors[popular_word]
    
        caption_words = caption.split()
        
        
        max_similarity = -1.0  # Để lưu độ tương đồng lớn nhất tìm thấy
        most_similar_caption = None  # Để lưu hashtag có độ tương đồng lớn nhất
        
        for word in caption_words:
            if word in word_vectors:
                word_vector = word_vectors[word]
                similarity = cosine_similarity([word_vector], [popular_word_vector])

                if similarity > max_similarity:
                    max_similarity = float(similarity[0][0])
                    most_similar_caption = word
        
        # Lưu hashtag có độ tương đồng lớn nhất và giá trị tương đồng
        max_similarity_caption_coronavirus.append(most_similar_caption)
        max_similarity_caption_coronavirus_values.append(max_similarity)
    
    df[f'max_similarity_caption_{popular_word}'] = max_similarity_caption_coronavirus
    df[f'calculator_max_similarity_caption_with_{popular_word}'] = max_similarity_caption_coronavirus_values
    
    
    return df 



def calculator_max_similarity_hashtags_with_popular_word(popular_word):
    name_of_max_similarity_hashtags = []
    max_similarity_hashtags_popular_word_values = []
    
    for index, row in df.iterrows():
        print(index)
        hashtags = row['hashtags']
        popular_word_vector = word_vectors[popular_word]
        hashtag_words = hashtags.split()
        
        
        max_similarity = -1.0  # Để lưu độ tương đồng lớn nhất tìm thấy
        most_similar_hashtag = None  # Để lưu hashtag có độ tương đồng lớn nhất
        
        for word in hashtag_words:
            if word in word_vectors:
                word_vector = word_vectors[word]
                similarity = cosine_similarity([word_vector], [popular_word_vector])

                if similarity > max_similarity:
                    max_similarity = float(similarity[0][0])
                    most_similar_hashtag = word
        
        # Lưu hashtag có độ tương đồng lớn nhất và giá trị tương đồng
        name_of_max_similarity_hashtags.append(most_similar_hashtag)
        max_similarity_hashtags_popular_word_values.append(max_similarity)
    
    df[f'max_similarity_hashtags_{popular_word}'] = name_of_max_similarity_hashtags
    df[f'calculator_max_similarity_hashtags_with_{popular_word}'] = max_similarity_hashtags_popular_word_values
    
    
    return df 


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
        
        total_similarity = 0
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
            average_similarity = float(average_similarity[0][0])
        else:
            average_similarity = 0.0
        
        average_similarity_values.append(average_similarity)

    # Thêm cột mới vào DataFrame chứa thông tin về độ tương đồng trung bình
    df[f'average_similarity_caption_{popular_word}'] = average_similarity_values
    
    return df


def calculator_average_similarity_with_hashtags_popular_word(popular_word):
    average_similarity_values = []
    
    for index, row in df.iterrows():
        print(index)
        hashtags = row['hashtags']
        popular_word_vector = word_vectors[popular_word]
        hashtag_words = hashtags.split()
        
        total_similarity = 0
        count = 0
        for word in hashtag_words:
            if word in word_vectors:
                word_vector = word_vectors[word]
                similarity = cosine_similarity([word_vector], [popular_word_vector])
                total_similarity += similarity
                count += 1
        
        if count > 0:
            average_similarity = total_similarity / count
            average_similarity = float(average_similarity[0][0])
        else:
            average_similarity = 0.0
        
        average_similarity_values.append(average_similarity)

    df[f'average_similarity_hashtags_{popular_word}'] = average_similarity_values
    
    return df


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
            # average_similarity = float(average_similarity[0][0])
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
df = calculator_ratio_similarity_caption_with_popular_word('pandemic',0.3)

df.columns


df = calculator_ratio_similarity_caption_with_popular_word('isolation',0.3)
df = calculator_ratio_similarity_caption_with_popular_word('vaccine',0.3)
df = calculator_ratio_similarity_caption_with_popular_word('lockdown',0.3)



def calculator_ratio_similarity_hashtags_with_popular_word(popular_word,threshold):
    # Tạo danh sách để lưu thông tin về độ tương đồng trung bình
    average_similarity_values = []
    count_over_threshold = []
    len_hashtags = []
    # Lặp qua mỗi dòng trong DataFrame
    for index, row in df.iterrows():
        print(index)
        hashtags = row['hashtags']
        popular_word_vector = word_vectors[popular_word]
        hashtag_words = hashtags.split()
        
        total_similarity = 0.0
        count = 0
        count_threshold= 0 
        # Lặp qua từng từ trong caption và tính toán độ tương đồng với "coronavirus"
        for word in hashtag_words:
            if word in word_vectors:
                word_vector = word_vectors[word]
                similarity = cosine_similarity([word_vector], [popular_word_vector])
                if similarity > threshold:  # Kiểm tra độ tương đồng > 0.3
                    total_similarity += 1  # Tăng biến đếm lên 1
                    count_threshold +=1
                count += 1
        if count > 0:
            average_similarity = (total_similarity / count ) * 100
            # average_similarity = float(average_similarity[0][0])
        else:
            average_similarity = 0.0
        
        average_similarity_values.append(average_similarity)
        count_over_threshold.append(count_threshold)
        len_hashtags.append(count)
    # Thêm cột mới vào DataFrame chứa thông tin về độ tương đồng trung bình
    df[f'ratio_similarity_hashtags_with_popular_word_{popular_word}_{threshold}'] = average_similarity_values
    df[f'ratio_similarity_hashtags_with_popular_word_{popular_word}_{threshold}_count'] = count_over_threshold
    df[f'ratio_similarity_hashtags_with_popular_word_{popular_word}_{threshold}_len_hashtags'] = len_hashtags
    return df
df = calculator_ratio_similarity_hashtags_with_popular_word('coronavirus',0.3)
df = calculator_ratio_similarity_hashtags_with_popular_word('virus',0.3)
df = calculator_ratio_similarity_hashtags_with_popular_word('pandemic',0.3)



df.columns
def plot_histograms_by_year_max_similarity(column_name):

    df_temp = df
    # df_temp = df
    years = [2020, 2021, 2022, 2023]
    data_by_year = {}

    for year in years:
        data_by_year[year] = df_temp[df_temp['year'] == year][column_name]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Biểu đồ Histogram thể hiện các giá trị tương đồng trung bình giữa Caption và từ Pandemic trong 4 năm')
    # fig.suptitle('Biểu đồ Histogram thể hiện các giá trị tương đồng lớn nhất giữa Caption và từ Pandemic trong 4 năm')
    # fig.suptitle("Biểu đồ Histogram thể hiện các giá trị tương đồng trung bình lớn nhất giữa C và ba từ CoronaVirus, Virus, Pandemic")
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

plot_histograms_by_year_max_similarity('average_similarity_caption_pandemic')  # Thay đổi tên tệp CSV tương ứng





average_similarity_hashtags_pandemic
calculator_max_similarity_hashtags_with_pandemic

average_similarity_caption_coronavirus
calculator_max_similarity_caption_with_coronavirus


df.columns

4/ 320 



def pie(column_name, threshold):
    df_temp = df
    years = [2020, 2021, 2022, 2023]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # fig.suptitle(f"Biểu đồ tròn thể hiện tần suất xuất hiện của các tỷ lệ phần trăm có gía trị tương đồng trên 0.3 của Hashtags và từ  với threshold >= {threshold}")
    # fig.suptitle(f"Biểu đồ Pie thể hiện phần trăm các giá trị tương quan trung bình giữa Caption và từ Pandemic trong 4 năm")
    
    fig.suptitle(f"Biểu đồ Pie thể hiện phần trăm các giá trị tương quan của các từ trong Caption và từ Pandemic lớn hơn 0.2 chiếm trên 20% trên tổng số từ của Caption")
    for i, year in enumerate(years):
        
        
        
        count_similarity_year = df_temp[df_temp['year'] == year]['year'].count()
        count_similarity_year_over_threshold = df_temp[(df_temp['year'] == year) & (df_temp[column_name] >= threshold)][column_name].count()
        ratio = count_similarity_year_over_threshold / count_similarity_year
        data = {
            'Category': ['Trên 0.2' , 'Dưới 0.2'],
            'Count': [count_similarity_year_over_threshold, count_similarity_year - count_similarity_year_over_threshold]
        }
        pie_df = pd.DataFrame(data)
        ax = axs[i // 2, i % 2]
        ax.pie(pie_df['Count'], labels=pie_df['Category'], autopct='%1.1f%%', startangle=90)
        ax.set_title(f"Year {year}")

    plt.show()
    
pie('ratio_similarity_caption_with_popular_word_pandemic_0.2', 0.2)


df.columns


print((6 / 320 )* 100)




    df_temp = df
    years = [2020, 2021, 2022, 2023]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # fig.suptitle(f"Biểu đồ tròn thể hiện tần suất xuất hiện của các tỷ lệ phần trăm có gía trị tương đồng trên 0.3 của Hashtags và từ Coronavirus với threshold >= {threshold}")
    fig.suptitle(f"Biểu đồ Pie thể hiện phầm trăm của các tương đồng trung bình giữa Caption và từ Coronavirus với threshold >= ")
    for i, year in enumerate(years):
        
        
        
        count_similarity_year = df_temp[df_temp['year'] == 2020]['year'].count()
        count_similarity_year_over_threshold = df_temp[(df_temp['year'] == 2020) & (df_temp['ratio_similarity_caption_with_popular_word_coronavirus_0.2'] >= 0.2)]['ratio_similarity_caption_with_popular_word_coronavirus_0.2'].count()
        ratio = count_similarity_year_over_threshold / count_similarity_year
        data = {
            'Category': [f'threshold >= ', 'Các Giá Trị Khác'],
            'Count': [count_similarity_year_over_threshold, count_similarity_year - count_similarity_year_over_threshold]
        }
        pie_df = pd.DataFrame(data)
        ax = axs[i // 2, i % 2]
        ax.pie(pie_df['Count'], labels=pie_df['Category'], autopct='%1.1f%%', startangle=90)
        ax.set_title(f"Year {year}")

    plt.show()
    


9.1

(29/320)* 100

df.columns
print(4/320)

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

def pie(column_name, threshold):
    df_temp = df
    years = [2020, 2021, 2022, 2023]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Biểu đồ Pie thể hiện phần trăm tương đồng trung bình giữa Hashtags và từ Virus với threshold >= {threshold}")
    colors = ['white','black']
    
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

        # Thử một số điều chỉnh để làm cho phần nhỏ hơn trở nên rõ ràng hơn
        explode = (0.1, 0)  # Phần đầu tiên sẽ được "nổ" ra một chút
        ax.pie(pie_df['Count'], labels=pie_df['Category'], autopct='%1.1f%%', startangle=90, explode=explode, radius=1.2,colors = colors)
        ax.set_title(f"Year {year}")

    plt.show()

# Gọi hàm với threshold cụ thể
pie('average_similarity_caption_pandemic', 0.2)


df.columns


import matplotlib.pyplot as plt
import pandas as pd

# Giả sử df là DataFrame của bạn
# df = ...

def pie(column_name, threshold):
    df_temp = df
    years = [2020, 2021, 2022, 2023]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Biểu đồ Pie thể hiện phần trăm tương đồng trung bình giữa Hashtags và từ Virus với threshold >= {threshold}")
    colors = ['white', 'black']

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

        # Thử một số điều chỉnh để làm cho phần nhỏ hơn trở nên rõ ràng hơn
        explode = (0.1, 0)  # Phần đầu tiên sẽ được "nổ" ra một chút
        wedges, texts, autotexts = ax.pie(
            pie_df['Count'],
            labels=pie_df['Category'],
            autopct='%1.1f%%',
            startangle=90,
            explode=explode,
            radius=1.2,
            colors=colors,
            textprops={'color': 'white', 'fontsize': 12}  # Đặt màu và kích thước cho văn bản
        )

        # Đặt màu nền của phần trắng thành trắng
        for autotext in autotexts:
            autotext.set_backgroundcolor('white')

        ax.set_title(f"Year {year}")

    plt.show()

# Gọi hàm với threshold cụ thể
pie('average_similarity_caption_pandemic', 0.2)




result0 = df[df['caption'].str.contains('pandemic', case=False)]
result1 = df1[df1['caption'].str.contains('pandemic', case=False)]

df.columns




def main(df):
    # df2 = pd.read_csv("D:/document/scrawl_instagram/data/processed/data_clean.csv")
    # result_df2 = df2[df2['caption'].str.contains('coronavirus',case  = False)]
    
    
    # df1 = pd.read_csv("D:/document/scrawl_instagram/data/processed/scrawl_insta_2.csv")
    # df = df.dropna(subset = ['hashtags'])
    # df = df.dropna(subset = ['caption'])
    
    
    
    
    # df = delete_columns(df)
    # df = df.drop_duplicates(subset=['url'], keep='first')
    # df = df.drop_duplicates(subset=['caption'], keep='first')
    # mask = df['caption'].apply(lambda x: len(x.split()) < 5)  # Kiểm tra độ dài của từng hashtag
    # df = df[~mask]  # Loại bỏ các dòng theo điều kiện
    
    
    # df = pd.read_csv("D:/document/scrawl_instagram/data/have_not_processing/csv/have_not_delete_hand_official.csv")
    
    df = pd.read_csv("D:/document/scrawl_instagram/data/processed/data_2_hai_gui.csv")
    
    df_random = df.sample(n=500,random_state  = 42)
    
    
    df_random
    
    # df = pd.read_csv("D:/document/scrawl_instagram/data/have_not_processing/csv/have_not_delete_hand_new_dataset_deleting_3.csv")
    df = pd.read_csv("D:/document/scrawl_instagram/data/processed/data_chia_deu_4_nam_official_9.csv")
    
    
    
    df = calculator_max_similarity_caption_with_popular_word("coronavirus")
    df = calculator_max_similarity_caption_with_popular_word('virus')
    df = calculator_max_similarity_caption_with_popular_word('pandemic')

    df = calculator_average_similarity_caption_with_popular_word('coronavirus')
    df = calculator_average_similarity_caption_with_popular_word('virus')
    df = calculator_average_similarity_caption_with_popular_word('pandemic')
    
    df = calculator_ratio_similarity_caption_with_popular_word('coronavirus', threshold=0.2)
    df = calculator_ratio_similarity_caption_with_popular_word('virus', threshold=0.2)
    df = calculator_ratio_similarity_caption_with_popular_word('pandemic', threshold=0.2)
    
    
    df = calculator_ratio_similarity_hashtags_with_popular_word('coronavirus', threshold=0.2)
    df = calculator_ratio_similarity_hashtags_with_popular_word('virus', threshold=0.2)
    df = calculator_ratio_similarity_hashtags_with_popular_word('pandemic', threshold=0.2)
    
   
    # df = calculator_ratio_similarity_caption_with_popular_word('coronavirus', threshold=0.5)
    # df = calculator_ratio_similarity_caption_with_popular_word('virus', threshold=0.5)
    # df = calculator_ratio_similarity_caption_with_popular_word('pandemic', threshold=0.5)
    
    df = calculator_max_similarity_hashtags_with_popular_word("coronavirus")
    df = calculator_max_similarity_hashtags_with_popular_word("virus")
    df = calculator_max_similarity_hashtags_with_popular_word("pandemic")
    
    df = calculator_average_similarity_with_hashtags_popular_word("coronavirus")
    df = calculator_average_similarity_with_hashtags_popular_word("virus")
    df = calculator_average_similarity_with_hashtags_popular_word("pandemic")
    
    
    
    # 0 -> 0.5
    # df = calculator_ratio_similarity_hashtags_with_popular_word("coronavirus",0.3)
    
    
    world bank child education coronavirus epidemic indian education school coronavirus    


    type(df.loc[12,'hashtags'])


def plot_histograms(column_name):


    df_temp = df
    years = [2020", "2021", "2022", "2023"]
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

plot_histograms('max_similarity_caption_coronavirus')  # Thay đổi tên tệp CSV tương ứng


df.columns


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

process_and_analyze_data_ratio('calculator_max_similarity_caption_with_coronavirus',0.7)  


df.columns

import pandas as pd
import matplotlib.pyplot as plt

def pie(column_name, threshold):
    df_temp = df
    years = [2020, 2021, 2022, 2023]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # fig.suptitle(f"Biểu đồ tròn thể hiện tần suất xuất hiện của các tỷ lệ phần trăm có gía trị tương đồng trên 0.3 của Hashtags và từ Coronavirus với threshold >= {threshold}")
    fig.suptitle(f"Biểu đồ Pie thể hiện phầm trăm của các tương đồng trung bình giữa Hashtags và từ Virus với threshold >= {threshold}")
    for i, year in enumerate(years):
        count_similarity_year = df_temp[df_temp['year'] == year]['year'].count()
        count_similarity_year_over_threshold = df_temp[(df_temp['year'] == year) & (df_temp[column_name] >= threshold)][column_name].count()
        ratio = count_similarity_year_over_threshold / count_similarity_year
        data = {
            'Category': [f'threshold >= {threshold}', 'Các Giá Trị Khác'],
            'Count': [count_similarity_year_over_threshold, count_similarity_year - count_similarity_year_over_threshold]
        }
        pie_df = pd.DataFrame(data)
        ax = axs[i // 2, i % 2]
        ax.pie(pie_df['Count'], labels=pie_df['Category'], autopct='%1.1f%%', startangle=90)
        ax.set_title(f"Year {year}")

    plt.show()
    
pie('average_similarity_caption_pandemic', 0.2)





def pie(column_name, threshold):
    df_temp = df
    years = [2020, 2021, 2022, 2023]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # fig.suptitle(f"Biểu đồ tròn thể hiện tần suất xuất hiện của các tỷ lệ phần trăm có gía trị tương đồng trên 0.3 của Hashtags và từ Coronavirus với threshold >= {threshold}")
    fig.suptitle(f"Biểu đồ Pie thể hiện phầm trăm của các tương đồng trung bình giữa Hashtags và từ Virus với threshold >= ")
    for i, year in enumerate(years):
        count_similarity_year = df_temp[df_temp['year'] == year]['year'].count()
        count_similarity_year_over_threshold = df_temp[(df_temp['year'] == year) & (df_temp['average_similarity_hashtags_coronavirus'] >= 0.3)]['average_similarity_hashtags_coronavirus'].count()
        ratio = count_similarity_year_over_threshold / count_similarity_year
        data = {
            'Category': [f'threshold >= ', 'Các Giá Trị Khác'],
            'Count': [count_similarity_year_over_threshold, count_similarity_year - count_similarity_year_over_threshold]
        }
        pie_df = pd.DataFrame(data)
        ax = axs[i // 2, i % 2]
        ax.pie(pie_df['Count'], labels=pie_df['Category'], autopct='%1.1f%%', startangle=90)
        ax.set_title(f"Year {year}")

    plt.show()
    
pie('average_similarity_caption_pandemic', 0.2)


df.columns






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

plot_histograms_by_year_max_similarity('average_similarity_hashtags_coronavirus') # Thay đổi tên tệp CSV tương ứng

df.columns


x = df.loc[0,'average_similarity_hashtags_coronavirus']
x










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
        caption_list = caption.split()
        hashtag_list = hashtag.split()
        
        max_similarity = -1.0  # Để lưu độ tương đồng lớn nhất tìm thấy
        most_similar_hashtag = None  # Để lưu hashtag có độ tương đồng lớn nhất
        most_similar_caption = None  # Để lưu hashtag có độ tương đồng lớn nhất
        # Lặp qua từng hashtag và tính toán độ tương đồng
        for caption in caption_list:
            print(index)
            for hashtag in hashtag_list:
                print(hashtag)
                
                if(caption in word_vectors and hashtag in word_vectors ):
                    text_caption = word_vectors[caption]
                    text_hashtag = word_vectors[hashtag]
                    similarity = cosine_similarity([text_caption],[text_hashtag])
                    
                    # Nếu độ tương đồng lớn hơn độ tương đồng lớn nhất đã tìm thấy
                    if similarity > max_similarity:
                        max_similarity = similarity
                        max_similarity = float(max_similarity[0][0])
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


def calculator_average_similarity_hashtags_with_caption():
    average_similarity_caption_hashtag_values = []
    
    # Lặp qua mỗi dòng trong DataFrame
    for index, row in df.iterrows():
        print(index)
        caption = row['caption']
        hashtag = row['hashtags']
        caption_list = caption.split()
        hashtag_list = hashtag.split()
    
        average_similarity = 0  # Để lưu độ tương đồng lớn nhất tìm thấy
        count = 0
        
        for caption in caption_list:
            for hashtag in hashtag_list:
                if(caption in word_vectors and hashtag in word_vectors ):
                    text_caption = word_vectors[caption]
                    text_hashtag = word_vectors[hashtag]
                    similarity = cosine_similarity([text_caption],[text_hashtag])
                    average_similarity += similarity
                    count += 1
        
        # Tính độ tương đồng trung bình (nếu có ít nhất một từ)
        if count > 0:
            average_similarity = average_similarity / count
            average_similarity = float(average_similarity[0][0])
        else:
            average_similarity = 0.0
        average_similarity_caption_hashtag_values.append(average_similarity)

    df['average_similarity_caption_hashtag_values'] = average_similarity_caption_hashtag_values
    
    
    return df


df = calculator_average_similarity_hashtags_with_caption()

# virus
# corronavirus
# isolation 
#  social distance
#  face mask

# comment : -> texxt -> max, tb 
# phan tram 

# dem ty le phan tram so tu` co do tuong quan > 0.3 trong tung caption


# Lưu DataFrame thành tệp CSV
df.to_csv('D:/document/scrawl_instagram/data/processed/data_1.csv', index=False)  # Thay 'ten_file.csv' bằng tên tệp bạn muốn lưu


## covid /covid 19-> coronavirus 
## text in text, hashtag with .... 6 tuwf
## tach' hashtag 
# plot từng từ / tổng
# threshold : 0.3 , 0.5, 0.7
# biểu đò phần trăm 









def calculator_average_max_similarity_each_word_in_caption_with_popular_word():
    list_average_similarity = []
    for index, row in df.iterrows():
        print(index)
        caption = row['caption']
        
        coronavirus_word_vector = word_vectors["coronavirus"]
        virus_word_vector = word_vectors["virus"]
        pandemic_word_vector = word_vectors["pandemic"]
        caption_words = caption.split()
        list_max = []

        
        for word in caption_words:
            if word in word_vectors:
                word_vector = word_vectors[word]
                
                similarity_coronavirus = cosine_similarity([word_vector], [coronavirus_word_vector])
                similarity_virus = cosine_similarity([word_vector], [virus_word_vector])
                similarity_pandemic = cosine_similarity([word_vector], [pandemic_word_vector])
                max_similarity = max(similarity_coronavirus,similarity_virus,similarity_pandemic)
        
   
                list_max.append(max_similarity)
        print(list_max)
        average_similarity = np.mean(list_max)
        list_average_similarity.append(average_similarity)
    
    df[f'calculator_average_similarity_caption_with_popular_word'] = list_average_similarity
    return df
df = calculator_average_max_similarity_each_word_in_caption_with_popular_word()



def calculator_average_max_similarity_each_word_in_hashtags_with_popular_word():
    list_average_similarity = []
    for index, row in df.iterrows():
        print(index)
        caption = row['hashtags']
        
        coronavirus_word_vector = word_vectors["coronavirus"]
        virus_word_vector = word_vectors["virus"]
        pandemic_word_vector = word_vectors["pandemic"]
        caption_words = caption.split()
        list_max = []

        
        for word in caption_words:
            if word in word_vectors:
                word_vector = word_vectors[word]
                
                similarity_coronavirus = cosine_similarity([word_vector], [coronavirus_word_vector])
                similarity_virus = cosine_similarity([word_vector], [virus_word_vector])
                similarity_pandemic = cosine_similarity([word_vector], [pandemic_word_vector])
                max_similarity = max(similarity_coronavirus,similarity_virus,similarity_pandemic)
        
   
                list_max.append(max_similarity)
        print(list_max)
        average_similarity = np.mean(list_max)
        list_average_similarity.append(average_similarity)
    
    df[f'calculator_average_similarity_hashtags_with_popular_word'] = list_average_similarity
    return df
df = calculator_average_max_similarity_each_word_in_hashtags_with_popular_word()
    
    

( 1.0 + 0.071 + 0.638 + 0.106 + 1.0 )/ 5

