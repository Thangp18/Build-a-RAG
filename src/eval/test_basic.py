from ragas import evaluate
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevance
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv(override=True)

mock_data = {
    "question": [
        "RAG là viết tắt của từ gì?",  # Câu 1: Trả lời siêu chuẩn
        "Trái đất quay quanh hành tinh nào?", # Câu 2: Bịa đặt thông tin (Hallucination)
        "Python là gì?"               # Câu 3: Trả lời lạc đề (Irrelevant)
    ],
    "contexts": [
        ["RAG là viết tắt của Retrieval-Augmented Generation, một kỹ thuật tối ưu LLM."],
        ["Trái đất quay quanh Mặt trời và là hành tinh thứ 3 trong hệ Mặt trời."],
        ["Python là một ngôn ngữ lập trình phổ biến, dễ học và mạnh mẽ."]
    ],
    "answer": [
        "RAG là viết tắt của Retrieval-Augmented Generation.", # Rất đúng!
        "Trái đất quay quanh Mặt trăng.",                      # Bịa đặt! Mặt trời mới đúng.
        "Hôm nay trời rất đẹp và tôi thích đi chơi."           # Lạc đề hoàn toàn!
    ],
    "ground_truth": [
        "RAG là Retrieval-Augmented Generation.",
        "Trái đất quay quanh Mặt trời.",
        "Python là ngôn ngữ lập trình."
    ]
}

# datasets
dataset = Dataset.from_dict(mock_data)

#llm
llm = ChatOpenRouter(model="openrouter/owl-alpha")

#eval
eval = evaluate(
    dataset=dataset,
    metrics=[faithfulness],
    llm=llm
)

df = eval.to_pandas()
cols_show = ["question", "answer", "faithfulness", "answer_relevance"]
print(df[cols_to_show].to_string(index=False))
print("\n📊 Kết quả chấm điểm:")
print(eval)