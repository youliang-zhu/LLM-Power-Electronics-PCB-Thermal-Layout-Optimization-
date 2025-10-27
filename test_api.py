import dashscope

# 直接输入 API Key
dashscope.api_key = "sk-dcff60c19e3245e387fdfa95958dfd2e"

# 调用 deepseek-r1 模型
response = dashscope.Generation.call(
    model="deepseek-r1-distill-llama-70b",
    messages=[{"role": "user", "content": "9.9和9.11谁大"}],
    result_format="message"
)

# 打印返回结果
if response and response.get("output") and response["output"].get("choices"):
    print("思考过程：")
    print(response["output"]["choices"][0].get("reasoning_content", "No reasoning content available"))

    print("最终答案：")
    print(response["output"]["choices"][0]["message"]["content"])
else:
    print("API 调用失败，返回结果:", response)
