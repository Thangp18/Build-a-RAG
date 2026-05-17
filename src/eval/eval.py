import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from ragas import evaluate
# Use ragas.metrics (NOT ragas.metrics.collections) — collections only supports InstructorLLM (OpenAI-style),
# while standard ragas.metrics supports any LangChain LLM including Gemini.
from ragas.metrics import (
    Faithfulness,        # Độ trung thực – câu trả lời có phù hợp với context không?
    AnswerRelevancy,     # Độ liên quan – câu trả lời có đúng trọng tâm câu hỏi không?
    ContextPrecision,    # Độ chính xác retrieval – context có xếp hạng phần liên quan lên đầu không?
    ContextRecall,       # Độ bao phủ retrieval – context có chứa đủ thông tin để trả lời không?
    SemanticSimilarity,  # Độ tương đồng ngữ nghĩa – câu trả lời có gần với ground truth không?
)
from datasets import Dataset
import math
from langchain_openai import ChatOpenAI 
from langchain_huggingface import HuggingFaceEmbeddings
from src.api.agent_state import get_agent
from dotenv import load_dotenv

load_dotenv()

def get_data(agent, question: str):
    input_msg= {
        "messages": [{
            "role": "user", "content": question
        }]
    }
    response = agent.invoke(input= input_msg)

    #Trích xuất context từ toolmsg
    context = []
    for msg in response.get("messages", []):
        msg_type = msg.__class__.__name__
        if msg_type == "ToolMessage":
            context.append(msg.content)
        
    if not context:
        context = ["Không dùng tool để truy xuất ngữ cảnh"]
        
    #Trích xuất answer
    answer = response["messages"][-1].content
    return context, answer

#eval
def eval():
    print("🚀 Khởi động Cỗ máy Đánh giá Tự động RAGAS...")
    agent = get_agent()
    if not agent:
        print("❌ Lỗi khởi tạo Agent.")
        return

    evaluation_set = [
        {
            "question": "Một khó khăn lớn của mô hình AI khi xử lý giao thông Việt Nam là gì?",
            "ground_truth": "Một khung hình camera giờ cao điểm có thể chứa hàng trăm xe máy đan xen, gây hiện tượng che khuất và bounding box chồng chéo."
        },
        {
            "question": "Tỷ lệ xe mô tô và xe gắn máy chiếm bao nhiêu trong cơ cấu phương tiện tại Việt Nam?",
            "ground_truth": "Tỷ lệ xe mô tô/gắn máy chiếm hơn 90% cơ cấu phương tiện trên đường."
        },
        {
            "question": "Hệ thống camera AI tại Việt Nam có thể phát hiện những lỗi vi phạm nào?",
            "ground_truth": "Hệ thống có thể phát hiện các lỗi như dừng xe đè vạch người đi bộ, đi sai làn đường, không đội mũ bảo hiểm và vượt đèn đỏ."
        },
        {
            "question": "Doanh số toàn thị trường ô tô Việt Nam năm 2025 đạt bao nhiêu xe?",
            "ground_truth": "Năm 2025 doanh số toàn thị trường đạt hơn 604.000 xe."
        },
        {
            "question": "Hãng xe nào dẫn đầu thị trường ô tô Việt Nam năm 2025?",
            "ground_truth": "VinFast dẫn đầu thị trường với hơn 175.000 xe điện được giao thành công."
        },
        {
            "question": "Chính sách hỗ trợ nào đang thúc đẩy xe điện tại Việt Nam?",
            "ground_truth": "Ô tô điện chạy pin được miễn 100% lệ phí trước bạ."
        },
        {
            "question": "Mẫu xe bán chạy nhất của VinFast năm 2024 là gì?",
            "ground_truth": "VF 5 là mẫu xe bán chạy nhất của VinFast với 32.000 xe."
        }
    ]

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    for idx, item in enumerate(evaluation_set):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"👉 Câu {idx+1}: {question}")

        #lấy dữ liệu
        context, answer = get_data(agent, question)
        questions.append(question)
        answers.append(answer)
        contexts_list.append(context)
        ground_truths.append(ground_truth)

    #Đóng gói dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    })

    llm = ChatOpenAI(
        model="openai/gpt-oss-120b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
    )

    EMBEDDING_MODEL = HuggingFaceEmbeddings(model="AITeamVN/Vietnamese_Embedding")

    # Định nghĩa bộ metrics đánh giá toàn diện (instantiate as objects)
    # Newer RAGAS versions require llm/embeddings to be passed at metric construction time
    metrics = [
        Faithfulness(),       # Độ trung thực: câu trả lời có phù hợp với context không?
        AnswerRelevancy(),    # Độ liên quan: câu trả lời có đúng trọng tâm câu hỏi không?
        ContextPrecision(),   # Độ chính xác retrieval: context có xếp hạng phần liên quan lên đầu không?
        ContextRecall(),      # Độ bao phủ retrieval: context có chứa đủ thông tin để trả lời không?
        SemanticSimilarity(), # Độ tương đồng ngữ nghĩa: câu trả lời có gần với ground truth không?
    ]

    print(f"\n📊 Đang đánh giá với {len(metrics)} metrics: {[m.name for m in metrics]}")
    print("⏳ Quá trình này có thể mất vài phút...\n")

    #eval
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=EMBEDDING_MODEL
    )

    # In và xuất báo cáo
    print("\n" + "="*60)
    print("🏆 KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG")
    print("="*60)

    # Tóm tắt từng metric theo dạng bảng với progress bar
    metric_descriptions = {
        "faithfulness":      "Độ trung thực (tránh hallucination)",
        "answer_relevancy":  "Độ liên quan của câu trả lời",
        "context_precision": "Độ chính xác của retrieval",
        "context_recall":    "Độ bao phủ của retrieval",
        "answer_similarity": "Độ tương đồng với ground truth",
    }
    result_dict = result.to_pandas().mean(numeric_only=True).to_dict()

    print(f"\n{'Metric':<25} {'Điểm':>6}   {'Trực quan':<22}  Ý nghĩa")
    print("-" * 85)
    valid_scores = []
    for metric_name, score in result_dict.items():
        desc = metric_descriptions.get(metric_name, "")
        if math.isnan(score):
            print(f"{metric_name:<25} {'N/A':>6}   [{'?'*20}]  {desc} ⚠️ Lỗi khi tính")
        else:
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"{metric_name:<25} {score:>6.4f}   [{bar}]  {desc}")
            valid_scores.append(score)
    print("-" * 85)

    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    print(f"\n🎯 Điểm trung bình tổng hợp: {avg_score:.4f}")

    # Đánh giá mức độ
    if avg_score >= 0.8:
        grade = "✅ Xuất sắc – Pipeline hoạt động tốt"
    elif avg_score >= 0.6:
        grade = "🟡 Trung bình khá – Có thể cải thiện thêm"
    elif avg_score >= 0.4:
        grade = "🟠 Cần cải thiện – Xem xét lại chunking / retrieval"
    else:
        grade = "🔴 Kém – Cần xem xét lại toàn bộ pipeline"
    print(f"📋 Xếp loại hệ thống: {grade}")
    print("=" * 60)

    # Lưu vào file CSV để làm tài liệu đính kèm CV
    df = result.to_pandas()
    output_path = "src/eval/result/final_report.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 Đã xuất báo cáo chi tiết ra file: {output_path}")

if __name__ == "__main__":
    eval()