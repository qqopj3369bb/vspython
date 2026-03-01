import os
import json
import requests

class AliyunDashScopeAgent:
    def __init__(self):
      
        
        self.api_key = "sk-2974ba547b034c4b8b9eac9ee813457e" 
        self.model_name = "qwen-plus"
        self.dashscope_api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        
        # 系统提示词（严格约束输出格式）
        self.system_prompt = """
        你的唯一任务是解析用户的出行路径查询请求，提取出发地和目的地。
        输出格式必须是严格的JSON字符串,不允许任何额外内容:
        1. 有效查询（包含出发地和目的地）：{"start":"出发地","end":"目的地"}
        2. 无效查询（无明确地点/非出行查询）：{"error":"请输入如'从A到B怎么走'的出行查询"}
        禁止输出JSON以外的任何文字、表情、符号!
        """

    def call_llm(self, user_input):
        """调用百炼LLM解析用户意图"""
        # 打印当前使用的 API Key（前8位+后8位，用于验证是否填对）
        print(f"当前使用的API Key:{self.api_key[:8]}...{self.api_key[-8:]}")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "input": {
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input}
                ]
            },
            "parameters": {
                "temperature": 0.0,
                "max_tokens": 100,
                "result_format": "text"
            }
        }
        
        try:
            response = requests.post(
                url=self.dashscope_api_url,
                headers=headers,
                json=payload,
                timeout=10
            )
            
            print(f"LLM原始响应: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                llm_output = result.get("output", {}).get("text", "").strip()
                llm_output = llm_output.replace("```json", "").replace("```", "").strip()
                
                try:
                    return json.loads(llm_output)
                except json.JSONDecodeError:
                    return {"error": f"解析失败,LLM返回:{llm_output}"}
            else:
                return {"error": f"API调用失败:{response.status_code} - {response.text}"}
                
        except Exception as e:
            return {"error": f"调用异常：{str(e)}"}

    def call_mcp_route(self, start, end):
        """查询出行路径"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": f"""
                        请查询从【{start}】到【{end}】的出行路径，要求：
                        1. 包含步行、公交、驾车三种方式
                        2. 每种方式说明距离和预计耗时
                        3. 用简洁的中文列出，不要多余内容
                        """
                    }
                ]
            },
            "parameters": {
                "temperature": 0.1,
                "max_tokens": 500
            }
        }
        
        try:
            response = requests.post(self.dashscope_api_url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                result = response.json()
                return result.get("output", {}).get("text", "未查询到路径信息")
            else:
                return f"路径查询失败：{response.status_code} - {response.text}"
        except Exception as e:
            return f"查询异常：{str(e)}"

    def run(self):
        """启动命令行交互"""
        print("===== 阿里云百炼 - 出行查询Agent =====")
        print(" 输入示例：从杭州西湖到杭州东站怎么走？")
        print(" 输入'exit'退出程序\n")
        
        while True:
            user_input = input("你：")
            if user_input.lower() == "exit":
                print("Agent:再见!")
                break
            
            print("Agent:正在理解你的请求...")
            llm_result = self.call_llm(user_input)
            
            if "error" in llm_result:
                print(f"Agent:{llm_result['error']}\n")
                continue
            
            start = llm_result.get("start")
            end = llm_result.get("end")
            if not start or not end:
                print("Agent:未识别到出发地或目的地,请明确输入！\n")
                continue
            
            print(f"Agent:正在查询从【{start}】到【{end}】的出行路径...")
            route_result = self.call_mcp_route(start, end)
            
            print(f"Agent:查询结果:\n{route_result}\n")

if __name__ == "__main__":
    agent = AliyunDashScopeAgent()
    agent.run()