############
# embedding 1
############

import openai
import pandas as pd
import numpy as np

# api key 호출
from openai import OpenAI
client = OpenAI()

questions = [
    "1. 무슨 일을 하는 데 있어, 흥미나 재미를 거의 느끼지 못한다",
    "2. 기분이 쳐지거나, 우울하거나, 희망이 없다고 느낀다",
    "3. 잠들기 어렵거나, 계속 잠들기 힘들거나, 혹은 너무 많이 잔다",
    "4. 피곤하다고 느끼거나 기운이 거의 없다",
    "5. 식욕이 거의 없거나, 아니면 너무 많이 먹는다",
    "6. 내 자신이 싫거나, 자신을 실패자라고 여기거나, 자신이 내 자신이나 가족을 실망시킨다고 생각한다",
    "7. 무슨 일에(신문 읽기, TV 보기 등) 집중하기가 어렵다",
    "8. 움직임이나 말이 너무 느려 남들이 알아차릴 정도이다. 아니면, 안절부절 못하거나 가만히 있지 못하여 보통 때보다 더 많이 돌아다닌다",
    "9. 차라리 죽었으면 더 낫겠다는 생각을 하거나, 어떻게 해서든지 자해를 하려는 생각을 한다"
]

# embedding function
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# generating embedding dataframe(df) for questions 
df = pd.DataFrame(questions, columns=['text'])
df['embedding'] = df['text'].apply(lambda x: get_embedding(x))
df['question'] = [f"Q11_{i}" for i in range(1, 10)]

# 'data' is a dataframe with summated scores of samples for Q11 (edited from "DATA_국민건강 관련 인식 및 관리방안에 대한 조사_250717_시군구 코드 변경.xlsx") 
# generating example data
# values = [
#     ["F3B5E1EC-99C6-4A31-ABC4-23476B4F71BE", 51, 4, 1, 1, 1, 1, 3, 2, 0, 1, 0, 0, 9,  "5~9"],
#     ["9F3DD358-283C-4FD1-8149-D9CF7F2EF545", 32, 2, 2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 3,  "0~4"],
#     ["5C66EC1A-EEEB-4BD2-A138-CD69B905BAB0", 52, 4, 2, 1, 1, 0, 0, 0, 1, 1, 1, 0, 5,  "5~9"],
#     ["083C8C3E-20EC-40CB-883C-B2A686A4F12E", 75, 6, 1, 1, 0, 1, 1, 1, 2, 0, 1, 1, 8,  "5~9"],
#     ["1446A134-9D71-4E0B-A95C-D000FF4D9955", 42, 3, 2, 1, 2, 2, 1, 2, 2, 2, 0, 1, 13, "10~19"],
# ]
# columns = [
#     "ID", "SQ3A", "SQ3B", "SQ4",
#     "Q11_1", "Q11_2", "Q11_3", "Q11_4", "Q11_5",
#     "Q11_6", "Q11_7", "Q11_8", "Q11_9",
#     "score", "score_group"
# ]
# data = pd.DataFrame(values, columns=columns)

# Q11 answer matrix of 'data'
q11_cols = [f"Q11_{i}" for i in range(1, 10)]
responses = data[q11_cols].values  

# question embedding matrix
emb_matrix = np.vstack(df['embedding'].values)

# weighted embedding
weighted_embeddings = responses @ emb_matrix  
data['embedding'] = weighted_embeddings.tolist()
