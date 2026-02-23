############
# embedding 3
############

import pandas as pd
import numpy as np
from openai import OpenAI

questions = {
    1: "무슨 일을 하는 데 있어, 흥미나 재미를 거의 느끼지 못하는가?",
    2: "기분이 쳐지거나, 우울하거나, 희망이 없다고 느끼는가?",
    3: "잠들기 어렵거나, 계속 잠들기 힘들거나, 혹은 너무 많이 자는가?",
    4: "피곤하다고 느끼거나 기운이 거의 없는가?",
    5: "식욕이 거의 없거나, 아니면 너무 많이 먹는가?",
    6: "내 자신이 싫거나, 자신을 실패자라고 여기거나, 자신이 내 자신이나 가족을 실망시킨다고 생각하나?",
    7: "무슨 일에(신문 읽기, TV 보기 등) 집중하기가 어려운가?",
    8: "움직임이나 말이 너무 느려 남들이 알아차릴 정도이거나, 안절부절 못하거나 가만히 있지 못하여 보통 때보다 더 많이 돌아다니는가?",
    9: "차라리 죽었으면 더 낫겠다는 생각을 하거나, 어떻게 해서든지 자해를 하려는 생각을 하는가?"
}

answer_map = {
    0: "전혀 아니다",
    1: "며칠 동안 그렇다",
    2: "일주일 이상 그렇다",
    3: "거의 매일 그렇다"
}

## 'data' is a dataframe with summated scores of samples for Q11 (edited from "DATA_국민건강 관련 인식 및 관리방안에 대한 조사_250717_시군구 코드 변경.xlsx") 
## Refer 'survey_emb1.py' for data format

# answers (scores) to texts
for i in range(1, 10):
    q_col = f"Q11_{i}"   # original answer column
    txt_col = f"txt_{i}"    # transformed text column

    data[txt_col] = data[q_col].apply(
        lambda x: f"{questions[i]} {answer_map.get(x, '')}."
    )

# api key 호출
client = OpenAI()

# embedding function
def embed_text_list(text_list, model="text-embedding-3-large", batch_size=128):
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]

        response = client.embeddings.create(
            model=model,
            input=batch
        )

        batch_emb = [item.embedding for item in response.data]
        embeddings.extend(batch_emb)

    return np.array(embeddings)

# texts to embeddings
for i in range(1, 10):
    col_txt = f"txt_{i}"
    col_emb = f"emb_{i}"

    data[col_emb] = list(embed_text_list(data[col_txt].tolist()))

# emb1 + ... + emb9 = embedding
emb_cols = [f"emb_{i}" for i in range(1, 10)]
data["embedding"] = data[emb_cols].apply(
    lambda row: sum(row.values),
    axis=1
)
