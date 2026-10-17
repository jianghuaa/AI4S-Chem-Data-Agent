import os
import pandas as pd
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# ================= 1. 定义工具 =================
@tool
def read_markdown_paper(file_path: str) -> str:
    """读取本地的 Markdown (.md) 格式论文或笔记。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content[:6000] + "\n...[截断]..." if len(content) > 6000 else content
    except Exception as e:
        return f"读取文件失败: {str(e)}"

@tool
def search_master_literature(keyword: str) -> str:
    """查阅本地文献核心结论或机制时使用。"""
    try:
        df = pd.read_csv("master_literature.csv", encoding='utf-8-sig')
        mask = df.apply(lambda row: row.astype(str).str.contains(keyword, case=False).any(), axis=1)
        results = df[mask]
        if results.empty: return f"本地主库中没有找到 '{keyword}' 的记录。请尝试联网。"
        output = f"在主库中找到 {len(results)} 篇相关文献：\n"
        for _, row in results.head(2).iterrows():
            output += f"- 《{row.get('文献标题')}》 核心结论: {row.get('核心结论')}\n"
        return output
    except Exception as e: return str(e)

@tool
def search_detailed_data(guest_name: str) -> str:
    """查询具体的光谱参数。"""
    try:
        df = pd.read_csv("rtp_records.csv", encoding='utf-8-sig')
        results = df[df['guest_name'].astype(str).str.contains(guest_name, case=False)]
        if results.empty: return f"本地无 '{guest_name}' 数据，请尝试联网。"
        output = f"找到 '{guest_name}' 数据：\n"
        for _, row in results.head(2).iterrows():
            output += f"- 磷光峰: {row['lambda_phos_nm']} nm, 寿命: {row['tau_p_s']} s\n"
        return output
    except Exception as e: return str(e)

@tool
def search_web(query: str) -> str:
    """联网搜索最新文献或资讯。"""
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key: return "无 TAVILY_API_KEY 变量。"
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, search_depth="advanced", max_results=2)
        results = response.get('results', [])
        if not results: return "网络未找到。"
        return "网络搜索结果：\n" + "\n".join([f"- {r['title']}: {r['content'][:200]}" for r in results])
    except Exception as e: return str(e)

tools = [read_markdown_paper, search_master_literature, search_detailed_data, search_web]

# ================= 2. 初始化模型 =================
llm = ChatOpenAI(
    model="glm-5.3-flash",
    api_key=os.getenv("ZAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.5, # 稍微调高一点温度，让导师人设更生动
    max_retries=3
)

# ================= 3. 构建硬核导师 Prompt =================
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个极其严谨、犀利且富有启发性的硬核科研导师。
    你的目标不是直接给学生答案，而是通过**苏格拉底式提问**，引导学生自己思考，找出逻辑漏洞。

    【规则】
    1. 只问 2-3 个直击灵魂的入门级问题，不要要求学生算任何复杂的数据。
    2. 提问方向：实验的对照组是什么？机制的本质是什么？证据的边界在哪里？
    3. 如果学生说“我看的是综述/我找不到原文”，立刻承认“是我要求的层级放错了”，并调整提问（从实验细节转向宏观机制）。
    4. 敏锐捕捉学生自己的洞见，顺着洞见追问本质。
    5. 你可以使用工具阅读学生的本地 md 文件，并根据文件内容进行提问。"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# ================= 4. 运行循环 =================
if __name__ == "__main__":
    print("=== 硬核科研导师 Agent 已上线 (输入 quit 退出) ===")
    print("")
    
    chat_history = []
    
    while True:
        user_input = input("\n你: ")
        if user_input.lower() in ['quit', 'exit']: break
            
        print("导师正在思考...")
        try:
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            answer = response['output']
            print(f"\n🧑‍🏫 导师:\n{answer}\n")
            
            chat_history.extend([HumanMessage(content=user_input), AIMessage(content=answer)])
            if len(chat_history) > 10: chat_history = chat_history[-10:]
        except Exception as e:
            print(f"\n❌ 报错: {e}")