"""
智能搜索助手 - 基于 LangGraph + Tavily API 的真实搜索系统
1. 理解用户需求
2. 使用Tavily API真实搜索信息  
3. 生成基于搜索结果的回答
"""

import asyncio
import os
from pathlib import Path
from typing import TypedDict, Annotated, Optional, cast

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr
from tavily import TavilyClient

# 加载环境变量：优先读取与本脚本同目录的 .env（推荐做法），
# 找不到时再回退到 python-dotenv 默认的“向上逐级查找”逻辑。
_ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_FILE if _ENV_FILE.exists() else None)

# 读取配置：统一 strip，避免 .env 中多余的空格/换行导致鉴权失败
LLM_API_KEY = (os.getenv("LLM_API_KEY") or "").strip()
LLM_MODEL_ID = (os.getenv("LLM_MODEL_ID") or "gpt-4o-mini").strip()
LLM_BASE_URL = (os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1").strip()
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
TAVILY_API_KEY = (os.getenv("TAVILY_API_KEY") or "").strip()

# 缺少 LLM Key 时直接给出可操作的提示，而不是等到调用接口时抛出晦涩的 401
if not LLM_API_KEY:
    raise SystemExit(
        "错误：未配置 LLM_API_KEY。\n"
        "  1) 复制模板：cp code/chapter6/Langgraph/.env.example code/chapter6/Langgraph/.env\n"
        "  2) 在 code/chapter6/Langgraph/.env 中填入 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_ID\n"
        "  或直接执行：export LLM_API_KEY=\"你的APIKey\""
    )

# 定义状态结构
class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str        # 用户查询（原始提问，生成答案时引用）
    search_query: str      # 优化后的搜索查询
    search_results: str    # Tavily搜索结果
    final_answer: str      # 最终答案
    step: str              # 当前步骤

# 初始化模型
# 注意：部分 OpenAI 兼容网关在“非流式”模式下会把结果包一层
# {"success": true, "data": {...}}，OpenAI SDK 无法解析（choices 会是 None）。
# 因此这里统一开启 streaming，由 LangChain 在内部把增量拼接成完整回复。
llm = ChatOpenAI(
    model=LLM_MODEL_ID,
    api_key=SecretStr(LLM_API_KEY),
    base_url=LLM_BASE_URL,
    temperature=0.7,
    streaming=True,
    timeout=LLM_TIMEOUT,
)

# Tavily 客户端：延迟到真正搜索时再创建，
# 这样即使没有配置 TAVILY_API_KEY，也能走“基于已有知识回答”的降级分支
tavily_client: Optional[TavilyClient] = None


def get_tavily_client() -> TavilyClient:
    """创建（并复用）Tavily 客户端；未配置 TAVILY_API_KEY 时抛出异常。"""
    global tavily_client
    if tavily_client is None:
        if not TAVILY_API_KEY:
            raise RuntimeError("未配置 TAVILY_API_KEY，无法调用 Tavily 搜索接口")
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    return tavily_client


def _content_to_text(content: object) -> str:
    """把消息内容统一转成纯文本。

    新版 LangChain 的 ``content`` 可能是分段列表（例如 [{"type": "text", "text": "..."}]），
    这里统一拍平成字符串，避免后续 str 操作出现类型或运行时错误。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return "" if content is None else str(content)


def _format_recent_history(messages: list, limit: int = 2, max_chars: int = 200) -> str:
    """把最近几轮对话整理成简短文本，用于理解“那后天呢”这类追问。"""
    lines = []
    for msg in messages[:-1][-limit:]:   # 排除本轮提问，只取历史
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        content = _content_to_text(msg.content).strip().replace("\n", " ")
        lines.append(f"{role}：{content[:max_chars]}")
    return "\n".join(lines)


def _extract_search_query(response_text: str, fallback: str) -> str:
    """从 LLM 输出中解析搜索关键词，解析失败时退回用户原始查询。"""
    for label in ("搜索词：", "搜索关键词：", "搜索词:", "搜索关键词:"):
        if label in response_text:
            # 只取该行的内容，避免把后面的说明文字一起当作关键词
            query = response_text.split(label, 1)[1].splitlines()[0].strip()
            query = query.strip("「」“”\"' ")
            if query:
                return query
    return fallback

async def understand_query_node(state: SearchState) -> dict:
    """步骤1：理解用户查询并生成搜索关键词"""

    # 获取最新的用户消息（多轮对话时取最近的一条 HumanMessage）
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = _content_to_text(msg.content)
            break

    history_text = _format_recent_history(state["messages"])
    history_block = f"\n对话历史（供理解追问时参考）：\n{history_text}\n" if history_text else ""

    understand_prompt = f"""分析用户的查询："{user_message}"
{history_block}
请完成两个任务：
1. 简洁总结用户想要了解什么
2. 生成最适合搜索的关键词（中英文均可，要精准）

格式：
理解：[用户需求总结]
搜索词：[最佳搜索关键词]"""

    response = await llm.ainvoke([SystemMessage(content=understand_prompt)])

    # 解析LLM的输出，提取搜索关键词（解析失败时退回用户原始查询）
    response_text = _content_to_text(response.content)
    search_query = _extract_search_query(response_text, user_message)

    return {
        # 保存用户的原始提问，生成答案时把它作为“用户问题”
        "user_query": user_message,
        "search_query": search_query,
        "step": "understood",
        "messages": [AIMessage(content=f"我理解您的需求：{response_text}")]
    }

async def tavily_search_node(state: SearchState) -> dict:
    """步骤2：使用Tavily API进行真实搜索"""

    search_query = state["search_query"]

    try:
        print(f"🔍 正在搜索: {search_query}")

        # 调用Tavily搜索API（同步 SDK 放进线程执行，避免阻塞事件循环）
        response = await asyncio.to_thread(
            get_tavily_client().search,
            query=search_query,
            search_depth="basic",
            include_answer=True,
            include_raw_content=False,
            max_results=5,
        )
        
        # 处理搜索结果
        search_results = ""
        
        # 优先使用Tavily的综合答案
        if response.get("answer"):
            search_results = f"综合答案：\n{response['answer']}\n\n"
        
        # 添加具体的搜索结果
        if response.get("results"):
            search_results += "相关信息：\n"
            for i, result in enumerate(response["results"][:3], 1):
                title = result.get("title", "")
                content = result.get("content", "")
                url = result.get("url", "")
                search_results += f"{i}. {title}\n{content}\n来源：{url}\n\n"
        
        if not search_results:
            search_results = "抱歉，没有找到相关信息。"
        
        return {
            "search_results": search_results,
            "step": "searched",
            "messages": [AIMessage(content=f"✅ 搜索完成！找到了相关信息，正在为您整理答案...")]
        }
        
    except Exception as e:
        error_msg = f"搜索时发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        
        return {
            "search_results": f"搜索失败：{error_msg}",
            "step": "search_failed",
            "messages": [AIMessage(content="❌ 搜索遇到问题，我将基于已有知识为您回答")]
        }

async def generate_answer_node(state: SearchState) -> dict:
    """步骤3：基于搜索结果生成最终答案"""

    history_text = _format_recent_history(state["messages"])
    history_block = f"\n对话历史：\n{history_text}\n" if history_text else ""

    # 检查是否有搜索结果
    if state["step"] == "search_failed":
        # 如果搜索失败，基于LLM知识回答
        fallback_prompt = f"""搜索API暂时不可用，请基于您的知识回答用户的问题：

用户问题：{state['user_query']}
{history_block}
请提供一个有用的回答，并说明这是基于已有知识的回答。"""

        response = await llm.ainvoke([SystemMessage(content=fallback_prompt)])

        response_text = _content_to_text(response.content)
        return {
            "final_answer": response_text,
            "step": "completed",
            "messages": [AIMessage(content=response_text)]
        }
    
    # 基于搜索结果生成答案
    answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案：

用户问题：{state['user_query']}
{history_block}
搜索结果：
{state['search_results']}

请要求：
1. 综合搜索结果，提供准确、有用的回答
2. 如果是技术问题，提供具体的解决方案或代码
3. 引用重要信息的来源
4. 回答要结构清晰、易于理解
5. 如果搜索结果不够完整，请说明并提供补充建议"""

    response = await llm.ainvoke([SystemMessage(content=answer_prompt)])

    response_text = _content_to_text(response.content)
    return {
        "final_answer": response_text,
        "step": "completed",
        "messages": [AIMessage(content=response_text)]
    }

# 构建搜索工作流
def create_search_assistant():
    workflow = StateGraph(SearchState)
    
    # 添加三个节点
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)
    
    # 设置线性流程
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)
    
    # 编译图
    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

async def main():
    """主函数：运行智能搜索助手"""
    
    # 没有 Tavily Key 时不退出程序：搜索节点会失败，并触发“基于已有知识回答”的降级分支
    if not TAVILY_API_KEY:
        print("⚠️  未配置 TAVILY_API_KEY：本次将无法联网搜索，会基于模型已有知识回答")
        print("   如需联网搜索，请参照 .env.example 在 code/chapter6/Langgraph/.env 中配置\n")

    app = create_search_assistant()
    
    print("🔍 智能搜索助手启动！")
    print("我会使用Tavily API为您搜索最新、最准确的信息")
    print("支持各种问题：新闻、技术、知识问答等")
    print("(输入 'quit' 退出)\n")

    # 全程复用同一个 thread_id：配合 InMemorySaver 让助手记住上下文，支持连续追问。
    # （原先每轮提问都会新建 thread_id，历史记忆随之丢失，多轮对话等同于重新开始）
    config: RunnableConfig = {"configurable": {"thread_id": "search-session-1"}}

    while True:
        user_input = input("🤔 您想了解什么: ").strip()
        
        if user_input.lower() in ['quit', 'q', '退出', 'exit']:
            print("感谢使用！再见！👋")
            break
        
        if not user_input:
            continue
        
        try:
            print("\n" + "="*60)
            
            # 只提交本轮的新消息，其余状态（历史对话）由 checkpointer 自动恢复。
            # LangGraph 支持“部分状态更新”，这里用 cast 说明该字典就是图状态的输入，
            # 避免类型检查器把缺少后续节点才写入的字段当成错误。
            initial_state = cast(SearchState, {"messages": [HumanMessage(content=user_input)]})
            async for output in app.astream(initial_state, config=config):
                for node_name, node_output in output.items():
                    if "messages" in node_output and node_output["messages"]:
                        latest_message = node_output["messages"][-1]
                        if isinstance(latest_message, AIMessage):
                            if node_name == "understand":
                                print(f"🧠 理解阶段: {latest_message.content}")
                            elif node_name == "search":
                                print(f"🔍 搜索阶段: {latest_message.content}")
                            elif node_name == "answer":
                                print(f"\n💡 最终回答:\n{latest_message.content}")
            
            print("\n" + "="*60 + "\n")
        
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            print("请重新输入您的问题。\n")

if __name__ == "__main__":
    asyncio.run(main())