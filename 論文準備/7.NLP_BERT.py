'''	
1. 回顧：從詞袋 -> Word embedding -> 序列 -> transformer -> BERT
    
    早期 NLP 從簡單統計模型來處理文本(詞袋、TF-IDF) -> 無法捕捉上下文
    Word embedding 讓每個詞彙在一個連續的向量空間中表示，語義相似的詞在數學上靠得更近 -> 缺少注意力
    transformer 可以動態關注(注意)關聯性高的詞彙，從上下文(整個序列中)判斷 -> 但每個詞只對應一個固定的向量，無法根據上下文靈活改變 (BERT)


2. BERT (Bidirectional Encoder Representations from Transformers)
    是一種預訓練語言模型，使用 Transformers 架構，所以能夠根據文本的上下文學習單詞的語義。


3. BERT 的預訓練(pre-training)與微調(fine-tunning)

    1. pre-training：
        BERT 在一個非常大的無標記文本數據集上進行訓練，使用兩個NLP任務來學習語言的結構和規則。

            1. 掩蔽語言模型(Masked Language Model, MLM)：
                隨機掩蔽句子中的一些單詞，並要求模型根據上下文預測這些被掩蔽的單詞。
                這樣可以讓模型學會根據上下文理解單詞。

            2. 預測(Next Sentence Prediction, NSP)：
                給定一段句子，要求模型判斷第二句是否真的是緊跟在第一句之後。
                有助於模型學習句子之間的邏輯關係。


    2. fine-tunning：
        進行預訓練之後，可以針對具體的NLP任務，進行微調，而不需要從頭開始訓練。


4. NLP 任務介紹：
    
    1. 句子邊界檢測 (Sentence Boundary Detection, SBD)
        確定文本中何處開始和結束一個句子。這對於文本處理非常關鍵，因為正確的句子分割是許多NLP任務的基礎。

    2. 問答 (Question Answering, QA)
        旨在讓機器能夠理解自然語言問題並提供精確的答案。

    3. 命名實體識別 (Named Entity Recognition, NER)
        從非結構化文本中提取信息並將其分類到預定義的命名實體，如地點 (LOC)、組織 (ORG) 和人物 (PER)

    4. 文本分類 (Text classification)
        例如情緒分析(Sentiment Analysis) -> 輸入文本（新聞、文章）中分析情緒信息。

    5. 文本摘要 (text summarization)
        從文件生成簡潔摘要。







'''