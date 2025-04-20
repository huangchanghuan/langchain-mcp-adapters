import logging
import os
import ssl

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import asyncio
from dotenv import load_dotenv
from httpx import Client, AsyncClient
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()
# 解析并输出结果
def print_optimized_result(agent_response):
    """
    解析代理响应并输出优化后的结果。
    :param agent_response: 代理返回的完整响应
    """
    messages = agent_response.get("messages", [])
    steps = []  # 用于记录计算步骤
    final_answer = None  # 最终答案

    for message in messages:
        if hasattr(message, "additional_kwargs") and "tool_calls" in message.additional_kwargs:
            # 提取工具调用信息
            tool_calls = message.additional_kwargs["tool_calls"]
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]
                steps.append(f"调用工具: {tool_name}({tool_args})")
        elif message.type == "tool":
            # 提取工具执行结果
            # 提取工具执行结果
            tool_name = message.name
            tool_result = message.content
            steps.append(f"{tool_name} 的结果是: {tool_result}")
        elif message.type == "ai":
            # 提取最终答案
            final_answer = message.content

    # 打印优化后的结果
    print("\n计算过程:")
    for step in steps:
        print(f"- {step}")
    if final_answer:
        print(f"\n最终答案: {final_answer}")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# 禁用 SSL 验证（不推荐用于生产环境）
os.environ['REQUESTS_CA_BUNDLE'] = ""
ssl._create_default_https_context = ssl._create_unverified_context

# 创建 ChatOpenAI 实例时传递 verify=False
model = ChatOpenAI(
    base_url="https://api.siliconflow.cn/v1/",
    model="deepseek-ai/DeepSeek-V3",
    # base_url="https://ide.tools.cmic.site/app/tiangong/openai-service/8080/v1/",
    # model="deepseek-v3",
    api_key="sk-lrhdwxwgnkrmqviiuzoqifucbpdddevpnucrvfehabiaxxmj",
    http_client=Client(verify=False, proxy="http://localhost:8888"),
    http_async_client=AsyncClient(verify=False, proxy="http://localhost:8888"),
)



async def main():
    async with MultiServerMCPClient({
        "playwright": {
            "url": "http://localhost:3001/sse",
            "transport": "sse"
        }
    }) as client:
        agent = create_react_agent(model, client.get_tools())


        response =await agent.ainvoke({"messages": "每次调用一次工具，等待结果，再根据结果进行下一步工具调用，不能有任何借口，完成以下任务：打开网址www.baidu.com,搜索system_fingerprint，然后点击进第一个搜索结果，返回结果"})

        print_optimized_result(response)

if __name__ == "__main__":
    asyncio.run(main())


