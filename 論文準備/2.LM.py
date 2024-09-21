import pandas as pd  # 數據處理庫
import nltk  # Natural Language Toolkit (NLTK) NLP庫，有很多 NLP 工具
from nltk.tokenize import word_tokenize # word_tokenize: 此函數可把一段文本切割為單詞 (tokenization)
from nltk.corpus import wordnet  # 語料庫（corpora）, wordnet: 語義詞典，用於提供詞的詞性、同義詞、反義詞等
from nltk.stem import WordNetLemmatizer # WordNetLemmatizer: 用於詞行還原
# 詞形還原：英文的名詞有複數型態，動詞則有過去式或進行式等，為了使單字標準化。使用 WordNetLemmatizer()將字詞變回原形。
#  1. 動詞 "running" 的詞根是 "run"。
#  2. 名詞 "better" 的詞根是 "good"。
    

# 下載 NLTK 資源
'''
NLTK的許多功能依賴於大型語言資源，例如語料庫 (corpora)、詞典和模型，這些資源因為體積較大或因為頻繁更新，通常不會內建於 NLTK 庫中。
因此，需要使用時，需要單獨下載這些資源。
'''
nltk.download('punkt') # 無監督學習模型，將文本按句子或單詞分割，不依賴詞典。
nltk.download('wordnet') # 大型語義詞典，用於詞形還原和語義分析(lemmatizer)

# 初始化詞形還原器
lemmatizer = WordNetLemmatizer() # 創建 class WordNetLemmatizer()的實例，這個實例中包含了詞形還原的功能。 



# 將 NLTK 的詞性標籤轉換為 WordNet 的詞性標籤
'''
詞性標籤(POS tags) 是對句子中的每個單詞進行語法分類的過程。標籤告訴我們每個單詞是名詞、動詞、形容詞還是副詞等。
    
    1. NLTK 的詞性標籤： 使用 NN(名詞)、VB(動詞)、JJ(形容詞)等。
    2. WordNet 的詞性標籤： 如 wordnet.NOUN(名詞)、wordnet.VERB(動詞)、wordnet.ADJ(形容詞)等。
    
為了讓 WordNetLemmatizer 知道每個單詞的詞性，需要一個從 NLTK tag 轉換到 WordNet tag。
'''
def get_wordnet_pos(tag):  # def a function with parameter: tag
    if tag.startswith('J'): # startswith(): 檢查是否以指定字母開頭，此處檢查 argument的開頭是否為 "J"
        return wordnet.ADJ # True: return WordNet's adj tag (wordnet.ADJ) 
    elif tag.startswith('V'): # 檢查 argument的開頭是否為 "V"
        return wordnet.VERB # True: return WordNet's verb tag (wordnet.VERB)
    elif tag.startswith('N'): # 檢查 argument的開頭是否為 "N"
        return wordnet.NOUN # True: return WordNet's noun tag (wordnet.NOUN)
    elif tag.startswith('R'): # 檢查 argument的開頭是否為 "R"
        return wordnet.ADV # True: return WordNet's adv tag (wordnet.ADV)
    else:
        return wordnet.NOUN  # 若非 "J", "V", "N", "R"，則 return wordnet.NOUN (其餘一律以名詞表示) 

    
    
# 改進的詞形還原函數
def lemmatize_words(words): #  def a function with parameter: words
    lemmatized = set() # 將一個空集合 asign to lemmatized
    for word in words: # 遍歷 words
        word = word.lower()  # 轉小寫 
        tagged = nltk.pos_tag([word]) # nltk.pos_tag(詞列表): 對 word 進行詞性標註，會返回一個 list [("單詞", "詞性")]
                                      # nltk.pos_tag() return 的值為用 list 包起來的 tuple，例如 [('run', 'VB'),('jump', 'VB')]
        wordnet_pos = get_wordnet_pos(tagged[0][1])  # 用剛剛 def function 把 return list 中第一個 tuple 中的 第二個 element 抓出來，也就是 tag 的部分，return wordnet 版本 tag
        lemmatized.add(lemmatizer.lemmatize(word, pos=wordnet_pos))  # 進行詞形還原
        # lemmatizer.lemmatize(): 對 word 進行詞形還原。會根據提供的詞性（wordnet_pos）將單詞還原成基本形式。
        # Ex. running -> wordnet_pos=wordnet_VERB -> run
        # 被還原的詞被加入 lemmatize set 中
    return lemmatized # retrun 全部被還原完畢的 set


def load_sentiment_words(file_path): 
    df = pd.read_excel(file_path) # 用 pandas的 read_excel 去讀取 file_path 的 excel 資料，並存到 df(dataframe) 中 
    positive_words = set(df.iloc[:, 1].dropna().astype(str))  # df.iloc[row_index, column_index]，這邊抓第一個 column 全部資料
    negative_words = set(df.iloc[:, 0].dropna().astype(str))  # dropna(): 去除 null, astype(str): 轉成 string，最後把結果轉成 set(可以去除重複值)
    return positive_words, negative_words # return 這兩個清乾淨的 set


def determine_sentiment(sentence, positive_words, negative_words):
    words = word_tokenize(sentence.lower()) # sentence轉小寫，並使用 word_tokenzie()切割成單詞
    has_positive = any(word in positive_words for word in words) # generator expression: return True or False, 檢查 words 中的每個 word 是否有在 positive_words 中 
    has_negative = any(word in negative_words for word in words) # any(): 只要其中有一個 True return True；不再檢查剩餘的單詞。

    if has_negative: # has_negative return True
        return 'Negative' 
    elif has_positive and not has_negative: # has_positive return True and has_negative return False
        return 'Positive'
    else:
        return 'Neutral'

# 載入詞庫，經過剛剛 def 的 function: load_sentiment_words()，會 return 2 個清洗乾淨的 set
positive_words, negative_words = load_sentiment_words('C:\\Users\\88697\\Downloads\\LM字典情緒詞庫.xlsx')  

# 還原詞庫，經過 def func: lemmatize_words(): return 完整還原完畢的 set
positive_words = lemmatize_words(positive_words)
negative_words = lemmatize_words(negative_words)

# 測試句子，經過 def func: determine_sentiment()，return 'Negative' or 'Positive' or 'Neutral'
test_sentence = "By applying voice intelligence to all external and internal communications, Dialpad is enabling organizations to sell more effectively, conduct more efficient meetings, personalize the customer experience, and make smarter business decisions automatically, in real-time, without installing new software."
sentiment = determine_sentiment(test_sentence, positive_words, negative_words)
print(f"{sentiment}")



# 讀取 CSV 文件，進行分析並保存結果
def analyze_csv(input_file, output_file): 
    df = pd.read_csv(input_file) # 用 pandas read_csv()讀取 csv 文件，並 asign to df(dataframe)
    df['LM label'] = df['sentence'].apply(lambda x: determine_sentiment(x, positive_words, negative_words)) # 把 sentence 欄位下的每一 row 去做 determine_sentiment(返回情緒)，並放入新欄位"LM label" 
    # lambda: anonymous function，(lambda parameter, return)
    # .apply(): 將某個函數應用到 DataFrame 或 Series 的每一個元素上，x 為 sentence欄位的每一 row 
    df.to_csv(output_file, index=False) # 把 sentence 與對應的情緒保存到另一個 CSV 文件中(output_file連結, 不用 index 欄位)

# 執行分析（替換路徑）
analyze_csv('C:\\Users\\88697\\Downloads\\AINews_label\\Total_data1.csv', 'C:\\Users\\88697\\Downloads\\AINews_label\\Total_data1.csv')



import pandas as pd 
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score # 指標函數
'''
1. Accuracy: 正確的預測數 / 總數據數
2. Precision： 預測正確陽 / 預測為陽
3. Recall: 預測正確陽 / 預測正確陽 + 預測錯誤陽
4. F1 Score: Precision 和 Recall 的調和平均數
'''


# 讀取 CSV 檔案
file_path = 'C:\\Users\\88697\\Downloads\\AINews_label\\Total_data1.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1') # 以 ISO-8859-1 編碼方式讀取csv文件

# 計算不匹配的次數
# 在 pandas，使用 Boolean comparison 對一整 column 進行操作時，這些運算會自動逐 row 應用到每個 element。
mismatches = data['researcher2 label'] != data['LM label'] # researcher2 label 應為人工label的部分，若 != LM label則為 mismatch (True)
mismatch_count = mismatches.sum() # 加總所有 mislable數量
total_data = data['researcher2 label'] == data['researcher2 label'] # 必為 True
data_count = total_data.sum() # 加總所有



# 計算準確率
accuracy = accuracy_score(data['researcher2 label'], data['LM label'])

'''
average='macro': 把每個類別的指標算出來後，再進行簡單的算術平均。
Ex. 
    三個類別 A、B、C 精確率分別為 0.90、0.70、0.80
    macro precision = (0.90 + 0.70 + 0.80) / 3 

不論這三個類別的數據量是否相同，最終結果都是對三個類別的平均。
'''
# 計算精確率
precision = precision_score(data['researcher2 label'], data['LM label'], average='macro')
# 計算召回率
recall = recall_score(data['researcher2 label'], data['LM label'], average='macro')
# 計算 F1 分數
f1 = f1_score(data['researcher2 label'], data['LM label'], average='macro')


print(f"Number of data: {data_count}")
print(f"Number of mismatches: {mismatch_count}")
print(f"Accuracy of the model predictions: {accuracy:.2f}") # .2f: 小數點後2位
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
