import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# ================= 0. 路径与环境变量 =================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")   # 关键修复1: 加载 .env，不再依赖系统环境变量

# ================= 1. 定义工具 =================

@tool
def search_master_literature(keyword: str) -> str:
    """当需要查阅文献综述、看过的论文核心结论、或者是找某一类机制（如'能隙'、'杂质'、'水相'）时使用此工具。"""
    try:
        df = pd.read_csv(BASE_DIR / "master_literature.csv", encoding="utf-8-sig")
        mask = df.apply(lambda row: row.astype(str).str.contains(keyword, case=False).any(), axis=1)
        results = df[mask]
        if results.empty:
            return f"主库中没有找到关于 '{keyword}' 的记录。请尝试换个词。"
        output = f"在主库中找到 {len(results)} 篇相关文献：\n"
        for _, row in results.head(3).iterrows():
            title = row.get('文献标题', '未知标题')
            conclusion = row.get('核心结论', '未记录') if pd.notna(row.get('核心结论')) else '未记录'
            mechanism = row.get('关键机制/设计策略', '未记录') if pd.notna(row.get('关键机制/设计策略')) else '未记录'
            output += f"- 《{title}》\n  机制: {mechanism}\n  结论: {conclusion}\n\n"
        return output
    except Exception as e:
        return f"读取主库失败: {str(e)}"

@tool
def search_detailed_data(guest_name: str) -> str:
    """当需要查询具体的发光波长(nm)、寿命(us/ms/s)、量子产率(%)、掺杂浓度等具体物理参数时使用此工具。输入客体分子名称（如'PCA', 'ICz'）。"""
    try:
        df = pd.read_csv(BASE_DIR / "rtp_records.csv", encoding="utf-8-sig")
        results = df[df['guest_name'].astype(str).str.contains(guest_name, case=False)]
        if results.empty:
            return f"细节数据库中没有找到 '{guest_name}' 的数据。"
        output = f"找到 '{guest_name}' 的相关数据记录：\n"
        for _, row in results.head(3).iterrows():
            output += f"- 客体: {row['guest_name']} | 宿主: {row['host']}, 掺杂: {row['doping_wt_pct']} wt%\n"
            output += f"  磷光峰: {row['lambda_phos_nm']} nm, 寿命: {row['tau_p_s']} s, 量子产率: {row['PLQY_pct']}%\n"
        return output
    except Exception as e:
        return f"读取细节库失败: {str(e)}"

tools = [search_master_literature, search_detailed_data]

# ================= 2. 初始化模型 =================
llm = ChatOpenAI(
    model="glm-5.3-flash",
    api_key=os.getenv("ZAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.3,
)

# ================= 3. 构建 Prompt =================
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个严谨的文献数据助理。你有两个知识库：
    1. 文献主库（包含综述文献、核心机制、结论），使用 search_master_literature 查询。
    2. 细节数据库（包含具体光谱参数），使用 search_detailed_data 查询。
    请基于工具返回的结果回答用户。如果用户问具体的数值，请优先查细节库；如果问研究思路，请查主库。不要自己编造数据。"""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# ================= 4. 创建 Agent（关键修复2: 不再使用 memory= 参数） =================
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 关键修复3: 用显式列表维护多轮对话历史，代替已废弃的 ConversationBufferMemory
chat_history = []

# ================= 5. 测试交互 =================
if __name__ == "__main__":
    print("=== 具备双库检索能力的文献助理已启动 ===")
    while True:
        user_input = input("\n你想查什么: ").strip()
        if user_input.lower() in ("quit", "exit", ""):
            break
        print("Agent 正在查阅...")
        try:
            response = agent_executor.invoke(
                {"input": user_input, "chat_history": chat_history}
            )
            answer = response["output"]
            print(f"\n🤖 助理回复:\n{answer}")
            chat_history.extend([("human", user_input), ("ai", answer)])
            chat_history = chat_history[-10:]  # 只保留最近5轮，防止上下文爆炸
        except Exception as e:
            print(f"\n❌ 报错啦: {e}")
