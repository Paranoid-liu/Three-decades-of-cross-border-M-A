import csv

import numpy as np
import pandas as pd

from bertopic import BERTopic
from nltk.corpus import stopwords

df = pd.read_csv('E:/jiaying/ber/data_0711.csv', engine='python')

stopwords = set(stopwords.words('english'))
stopwords = stopwords.union(
    {"et", "al", "named", "etc", "ie", "eg", "id", "even", "would", "soon", "I", "a", "this", "paper", "article"
        , "acquisition", "acquisitions", "cross-border", "cross-border m&a", "cross-border m&as", "m-and-a"
        , "mas", "study", "authors", "author", "significant", "find", "cross",  "data", "also", "model", "two",
     "like", "results", "result", "analysis", "may", "analysts", "acquisition", "acquisitions",
     "mergers", "merger", "mas", "ma", "cross", "border", "crossborder", "findings", "new", "countries", "evidence",
     "literature", "new", "review", "using", "level", "country", "approach", "sample", "research", "theory", "firm",
     "three", "field"})


def lower_split(text, min_len=2, max_len=15):
    temp = text.lower().translate(str.maketrans('', '', r"""!"#$%&'()*+,./:;<=>?@[\]^_`{|}~""")).split()
    return [token for token in temp if
            min_len <= len(token) <= max_len and not token.startswith('_') and not token.isdigit()]


def tokenize(text):
    # 分词，去除停词
    return [token for token in lower_split(text) if token not in stopwords]


df['content'] = df['Data'].apply(tokenize)

docs = df['content'].astype('str')
topic_model = BERTopic(verbose=True, embedding_model="all-MiniLM-L6-v2", calculate_probabilities=True,
                       diversity=0.9, top_n_words=15)

topics, probs = topic_model.fit_transform(docs)
print(topic_model.get_topic_info())

probability_threshold = 0.01
new_topics = [np.argmax(prob) if max(prob) >= probability_threshold else -1 for prob in probs]
topic_model.update_topics(docs, new_topics)
documents = pd.DataFrame({"Document": docs, "Topic": new_topics})
topic_model._update_topic_size(documents)

topics_reduced, probs_reduced = topic_model.reduce_topics(docs, new_topics, probs, nr_topics=9)
topic_model.update_topics(docs, topics_reduced)

visualize_topics = topic_model.visualize_topics()
# save trained_model
topic_model.save("my_model")

# 可视化结果保存至html中，可以动态显示信息
visualize_topics.write_html('visualize_topics.html')
print(topic_model.get_topics())
print(topic_model.get_topic_info())

# load trained_model
# topic_model = BERTopic.load("E:/jiaying/ber/0728/my_model")
# visualize_topics.write_html('visualize_topics.html')
# print(topic_model.get_topics())
# print(topic_model.get_topic_info())

#  print result_csv
fina_li = []
with open('E:/jiaying/ber/data_0711.csv', "r", encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    i = 0  # if row[20] in author_H[row[1]]:
    for row in reader:
        row.append(topics_reduced[i])
        fina_li.append(row)
        i = i + 1

with open("topic_result.csv", "w", newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["No", "Title", "Year", "Abstract", "Author Keywords", "Index Keywords", "Data", "Topic"])
    writer.writerows(fina_li)

