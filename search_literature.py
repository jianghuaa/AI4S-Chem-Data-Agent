import os
import pandas as pd
import time
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# ================= 1. 定义工具 =================
@tool
def search_master_literature(keyword: str) -> str:
    """查阅文献核心结论或机制时使用。"""
    try:
        df = pd.read_csv("master_literature.csv", encoding='utf-8-sig')
        mask = df.apply(lambda row: row.astype(str).str.contains(keyword, case=False).any(), axis=1)
        results = df[mask]
        if results.empty:
            return f"主库中没有找到关于 '{keyword}' 的记录。"
        
        output = f"在主库中找到 {len(results)} 篇相关文献：\n"
        for _, row in results.head(3).iterrows():
            title = row.get('文献标题', '未知标题')
            conclusion = row.get('核心结论', '未记录') if pd.notna(row.get('核心结论')) else '未记录'
            output += f"- 《{title}》\n  核心结论: {conclusion}\n"
        return output
    except Exception as e:
        return f"读取主库失败: {str(e)}"

@tool
def search_detailed_data(guest_name: str) -> str:
    """查询具体的光谱参数（波长、寿命、量子产率）时使用。"""
    try:
        df = pd.read_csv("rtp_records.csv", encoding='utf-8-sig')
        results = df[df['guest_name'].astype(str).str.contains(guest_name, case=False)]
        if results.empty:
            return f"细节库中没有找到 '{guest_name}' 的数据。"
        
        output = f"找到 '{guest_name}' 的相关数据记录：\n"
        for _, row in results.head(3).iterrows():
            output += f"- 宿主: {row['host']}, 掺杂: {row['doping_wt_pct']} wt%\n"
            output += f"  磷光峰: {row['lambda_phos_nm']} nm, 寿命: {row['tau_p_s']} s, PLQY: {row['PLQY_pct']}%\n"
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
    max_retries=3  # 增加网络重试次数，防止Connection error
)

# ================= 3. 构建 Prompt =================
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个严谨的文献数据助理。请基于工具返回的结果回答用户。如果问具体数值，查细节库；如果问研究思路，查主库。不要自己编造数据。"),
    MessagesPlaceholder(variable_name="chat_history"), # 用于注入历史记录
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ================= 4. 创建 Agent (不加自带memory) =================
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# ================= 5. 连续对话循环 (手动管理记忆) =================
if __name__ == "__main__":
    print("=== 具备手动记忆的双库文献助理已启动 (输入 'quit' 退出) ===")
    chat_history = [] # 初始化历史列表
    
    while True:
        user_input = input("\n你想查什么: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        print("Agent 正在查阅...")
        try:
            # 把当前问题和历史记录一起传给 Agent
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            answer = response['output']
            print(f"\n🤖 助理回复:\n{answer}")
            
            # 将本轮对话加入历史记录
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))
            
            # 限制历史记录长度，防止Token爆炸
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
                
        except Exception as e:
            print(f"\n❌ 报错啦: {e}")
