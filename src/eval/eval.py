import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from ragas import evaluate
from ragas.metrics import faithfulness
from datasets import Dataset
# from langchain_openrouter import ChatOpenRouter
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
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

    #call llm 
    # llm = ChatNVIDIA(model="moonshotai/kimi-k2.6", api_key= os.getenv("NVIDIA_API_KEY"))
    llm = ChatGoogleGenerativeAI(
        model= "gemini-2.5-flash-lite",
        api_key= os.getenv("GOOGLE_API_KEY")
    )
    # llm = ChatGroq(
    #     model= "llama-3.1-8b-instant",
    #     api_key= os.getenv("GROQ_API_KEY")
    # )

    #eval
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness],
        llm=llm
    )

    # In và xuất báo cáo
    print("\n" + "="*40)
    print("🏆 KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG")
    print("="*40)
    print(result)
    
    # Lưu vào file Excel/CSV để làm tài liệu đính kèm CV
    df = result.to_pandas()
    output_path = "src/eval/result/final_report.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 Đã xuất báo cáo chi tiết ra file: {output_path}")
if __name__ == "__main__":
    eval()