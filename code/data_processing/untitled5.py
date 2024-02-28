import pandas as pd

def remove_short_hashtags(df):
    df['hashtags'] = df['hashtags'].astype(str)  # Chắc chắn rằng cột là kiểu str
    mask = df['hashtags'].apply(lambda x: len(x.split()) < 3)  # Kiểm tra độ dài của từng hashtag
    df = df[~mask]  # Loại bỏ các dòng theo điều kiện
    return df

# Tạo DataFrame ví dụ
data = {'hashtags': ['se om en', 'python programming', 'data science', 'ai']}
df = pd.DataFrame(data)

# In ra DataFrame trước khi xóa
print("Before removing short hashtags:")
print(df)

# Hàm riêng lẻ để xóa các dòng có độ dài của giá trị trong cột "hashtags" nhỏ hơn 3
df = remove_short_hashtags(df)

# In ra DataFrame sau khi xóa
print("\nAfter removing short hashtags:")
print(df)
