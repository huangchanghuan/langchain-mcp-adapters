import json
import logging
import os
import ssl
from typing import AsyncIterable

from langchain.tools.json import tool
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import asyncio
from dotenv import load_dotenv
from httpx import Client, AsyncClient
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import AIMessageChunk


load_dotenv()


# 解析并流式输出结果
async def stream_optimized_result(agent_response_stream: AsyncIterable) -> None:
    """
    流式解析代理响应并输出优化后的结果。
    :param agent_response_stream: 代理返回的流式响应
    """
    steps = []  # 用于记录计算步骤
    final_answer_chunks = []  # 收集最终答案的片段

    print("\n计算过程:")
    async for chunk in agent_response_stream:
        if isinstance(chunk, AIMessageChunk):
            # 处理AI消息块
            if chunk.content:
                # 如果是最终答案的流式输出
                print(chunk.content, end="", flush=True)
                final_answer_chunks.append(chunk.content)
        elif hasattr(chunk, "additional_kwargs") and "tool_calls" in chunk.additional_kwargs:
            # 提取工具调用信息
            tool_calls = chunk.additional_kwargs["tool_calls"]
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]
                step = f"- 调用工具: {tool_name}({tool_args})"
                print(step)
                steps.append(step)
        elif hasattr(chunk, "type") and chunk.type == "tool":
            # 提取工具执行结果
            tool_name = chunk.name
            tool_result = chunk.content
            step = f"- {tool_name} 的结果是: {tool_result}"
            print(step)
            steps.append(step)

    # 打印最终答案（如果是以流式方式收集的）
    if final_answer_chunks:
        final_answer = "".join(final_answer_chunks)
        print(f"\n\n最终答案: {final_answer}")


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
    # base_url="https://ide.tools.cmic.site/app/tiangong/openai-service/8080/v1/",
    # model="deepseek-v3",
    # api_key="sk-lrhdwxwgnkrmqviiuzoqifucbpdddevpnucrvfehabiaxxmj",
    # base_url="https://api.siliconflow.cn/v1/",
    # model="Qwen/Qwen2.5-32B-Instruct",
    # api_key="sk-lrhdwxwgnkrmqviiuzoqifucbpdddevpnucrvfehabiaxxmj",
    base_url="https://api.hunyuan.cloud.tencent.com/v1/",
    model="hunyuan-pro",
    api_key="sk-VMGNgGpim4ztGDiQtx9Gzm6S92PcJeaabchVrqcmc2DLIb59",
    http_client=Client(verify=False, proxy="http://localhost:8888"),
    http_async_client=AsyncClient(verify=False, proxy="http://localhost:8888"),
    streaming=True  # 启用流式响应
)


async def main():
    async with MultiServerMCPClient({
        "playwright": {
            "url": "http://localhost:3001/sse",
            "transport": "sse"
        }
    }) as client:
        agent = create_react_agent(
            model,
            client.get_tools(),
        )

        # 使用流式调用
        response_stream = agent.astream({
            "messages": "每次调用一次工具（注意工具的参数格式，你经常把json格式返回错误），等待结果，再根据结果进行下一步工具调用，不能有任何借口，完成以下任务：打开网址http://www.baidu.com,搜索system_fingerprint，然后点击进第一个搜索结果，返回结果"
        })

        async for chunk in response_stream:
            print(chunk)
if __name__ == "__main__":
    asyncio.run(main())