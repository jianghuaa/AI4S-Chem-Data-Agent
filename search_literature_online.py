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
def search_master_literature(keyword: str) -> str:
    """查阅本地文献核心结论或机制时使用。"""
    try:
        df = pd.read_csv("master_literature.csv", encoding='utf-8-sig')
        mask = df.apply(lambda row: row.astype(str).str.contains(keyword, case=False).any(), axis=1)
        results = df[mask]
        if results.empty:
            return f"本地主库中没有找到关于 '{keyword}' 的记录。请尝试使用 search_web 去互联网搜索。"
        
        output = f"在主库中找到 {len(results)} 篇相关文献：\n"
        for _, row in results.head(3).iterrows():
            output += f"- 《{row.get('文献标题', '未知标题')}》\n  核心结论: {row.get('核心结论', '未记录')}\n"
        return output
    except Exception as e:
        return f"读取主库失败: {str(e)}"

@tool
def search_detailed_data(guest_name: str) -> str:
    """查询本地数据库中具体的光谱参数（波长、寿命、量子产率）时使用。"""
    try:
        df = pd.read_csv("rtp_records.csv", encoding='utf-8-sig')
        results = df[df['guest_name'].astype(str).str.contains(guest_name, case=False)]
        if results.empty:
            return f"本地细节库中没有找到 '{guest_name}' 的数据。请尝试使用 search_web 去互联网搜索。"
        
        output = f"找到 '{guest_name}' 的相关数据记录：\n"
        for _, row in results.head(3).iterrows():
            output += f"- 宿主: {row['host']}, 掺杂: {row['doping_wt_pct']} wt%, 磷光峰: {row['lambda_phos_nm']} nm, 寿命: {row['tau_p_s']} s\n"
        return output
    except Exception as e:
        return f"读取细节库失败: {str(e)}"

@tool
def search_web(query: str) -> str:
    """当本地知识库没有数据，或者用户询问最新文献、互联网资讯时使用此工具进行联网搜索。输入要搜索的关键词。"""
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "未配置 TAVILY_API_KEY 环境变量，无法联网。"
        
        client = TavilyClient(api_key=api_key)
        # search_depth="advanced" 会去抓取网页正文，返回给大模型
        response = client.search(query=query, search_depth="advanced", max_results=3)
        
        results = response.get('results', [])
        if not results:
            return f"互联网上也没有找到关于 '{query}' 的信息。"
            
        output = "通过联网搜索找到以下信息：\n"
        for res in results:
            output += f"- 标题: {res['title']}\n  摘要: {res['content']}\n  链接: {res['url']}\n\n"
        return output
    except Exception as e:
        return f"联网搜索失败: {str(e)}"

tools = [search_master_literature, search_detailed_data, search_web]

# ================= 2. 初始化模型 =================
llm = ChatOpenAI(
    model="glm-5.3-flash",
    api_key=os.getenv("ZAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.3,
    max_retries=3
)

# ================= 3. 构建 Prompt =================
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个严谨的文献数据助理。
    你有三个工具：
    1. 查本地主库 (search_master_literature)
    2. 查本细节库 (search_detailed_data)
    3. 联网搜索 (search_web)
    规则：优先查本地库。如果本地库返回“没有找到”，你**必须**调用 search_web 去互联网上搜索，以给用户最准确的答案。绝不要自己编造数据。"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# ================= 4. 连续对话循环 =================
if __name__ == "__main__":
    print("=== 具备联网搜索能力的文献助理已启动 ===")
    chat_history = []
    
    while True:
        user_input = input("\n你想查什么: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        print("Agent 正在查阅 (先查本地，再查网络)...")
        try:
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            answer = response['output']
            print(f"\n🤖 助理回复:\n{answer}")
            
            chat_history.extend([HumanMessage(content=user_input), AIMessage(content=answer)])
            if len(chat_history) > 10: chat_history = chat_history[-10:]
        except Exception as e:
            print(f"\n❌ 报错啦: {e}")