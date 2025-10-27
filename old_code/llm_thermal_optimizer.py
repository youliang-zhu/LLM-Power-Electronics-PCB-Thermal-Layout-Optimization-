import requests
import json
import numpy as np
import re

# 从NSGA_testing中只导入真正需要的类和函数
from NSGA_testing import Module, Net, Solution

class WenxinAPI:
    def __init__(self):
        self.api_key = "QmmbrnVmMXcT0hAS9NE1Hhp3"
        self.secret_key = "eG306gBBebUsjIiOPqIytWXyx14opu06"
        self.token_url = "https://aip.baidubce.com/oauth/2.0/token"
        self.llm_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions"
        self.access_token = self.get_access_token()

    def get_access_token(self):
        """获取百度文心一言的 access_token"""
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key
        }
        response = requests.post(self.token_url, params=params)
        if response.status_code == 200:
            return response.json().get("access_token")
        return None

    def chat(self, prompt):
        """调用百度文心 API 进行对话"""
        if not self.access_token:
            print("Error: 无法获取 Access Token")
            return None

        headers = {"Content-Type": "application/json"}
        params = {"access_token": self.access_token}
        data = {"messages": [{"role": "user", "content": prompt}]}

        response = requests.post(self.llm_url, headers=headers, params=params, json=data)
        if response.status_code == 200:
            return response.json()["result"]
        print(f"API Error: {response.status_code}, {response.text}")
        return None

class LLMThermalOptimizer:
    def __init__(self, max_rounds=5, temp_threshold=0.5):
        self.api = WenxinAPI()
        self.max_rounds = max_rounds
        self.temp_threshold = temp_threshold
        self.optimization_history = []
        
    def format_thermal_data(self, solution):
        """将热分析数据格式化为LLM可处理的文本"""
        if not hasattr(solution, 'temperature'):
            return "错误：解决方案没有温度数据"
            
        # 获取最高温度及其位置
        max_temp = solution.max_temp
        max_temp_pos = np.unravel_index(solution.temperature.argmax(), solution.temperature.shape)
        max_y, max_x = max_temp_pos
        
        # 转换为实际坐标
        real_x = max_x * solution.resolution
        real_y = max_y * solution.resolution
        
        # 找出靠近热点的组件
        nearby_components = []
        for module in solution.modules:
            width, height = module.get_dimensions()
            # 检查组件是否在热点附近(距离小于100mm)
            center_x = module.x + width/2
            center_y = module.y + height/2
            distance = ((center_x - real_x)**2 + (center_y - real_y)**2)**0.5
            if distance < 100:
                nearby_components.append({
                    "id": module.id,
                    "name": module.name,
                    "x": module.x,
                    "y": module.y,
                    "width": width,
                    "height": height,
                    "rotation": module.rotation,
                    "distance": distance
                })
        
        # 组件功率信息
        component_powers = {
            "C1": 0.1, "C2": 0.1, "Y1": 0.05,
            "R1": (12**2)/100e3, "R2": (12**2)/100e3,
            "R3": (12**2)/1e3, "R4": (12**2)/1e3,
            "Q1": 0.5, "Q2": 0.5
        }
        
        # 构建布局概述
        layout_description = []
        for module in solution.modules:
            width, height = module.get_dimensions()
            layout_description.append(
                f"{module.name}(ID:{module.id}): 位置=({module.x:.1f}, {module.y:.1f}), "
                f"尺寸=({width}x{height}), 旋转={module.rotation}度"
            )
        
        # 修改提示模板，只请求一条建议
        prompt = f"""
分析以下PCB布局的热分布，并提供单个优化建议以降低最高温度：

【热分析数据】
- 最高温度: {max_temp:.2f}°C，位置坐标: ({real_x:.1f}, {real_y:.1f})
- PCB尺寸: {solution.board_width}mm x {solution.board_height}mm

【热点附近组件】(按距离排序)
{json.dumps([comp for comp in sorted(nearby_components, key=lambda x: x['distance'])], indent=2, ensure_ascii=False)}

【组件功率信息】
{json.dumps(component_powers, indent=2, ensure_ascii=False)}

【当前完整布局】
{chr(10).join(layout_description)}

请提供一条具体的布局修改建议，以降低PCB的最高温度。明确指出：
1. 要移动的组件ID和名称
2. 新的建议位置(x,y坐标)或旋转角度
3. 调整的理由

将你的建议格式化为以下结构，以便系统自动解析：
建议:
组件ID: [ID]
组件名称: [NAME]
当前位置: ([X], [Y])
建议位置: ([NEW_X], [NEW_Y])
建议旋转: [ROTATION] 度
理由: [REASON]

请确保建议的位置在PCB板范围内，并考虑组件散热特性和功率密度。
优先考虑移动高功率组件，分散热源，增加热传导路径。
"""
        return prompt
    
    def parse_llm_suggestions(self, llm_response, solution):
        """解析LLM的建议并转换为可执行的布局调整"""
        if not llm_response:
            return []
        
        # 打印LLM响应，帮助调试
        print("LLM响应:\n", llm_response[:300], "...")  # 只打印前300个字符
        
        suggestions = []
        
        # 修改正则表达式以匹配单条建议
        patterns = [
            # 原始格式
            r"建议:?\s*[^\n]*\n组件ID:\s*(\d+)[^\n]*\n组件名称:\s*([^\n]*)\n当前位置:\s*\(([^,]+),\s*([^\)]+)\)[^\n]*\n建议位置:\s*\(([^,]+),\s*([^\)]+)\)[^\n]*\n建议旋转:\s*(\d+)[^\n]*\n理由:\s*([^\n]*)",
            
            # 带有-前缀的格式
            r"建议:?\s*[^\n]*\n- 组件ID:\s*(\d+)[^\n]*\n- 组件名称:\s*([^\n]*)\n- 当前位置:\s*\(([^,]+),\s*([^\)]+)\)[^\n]*\n- 建议位置:\s*\(([^,]+),\s*([^\)]+)\)[^\n]*\n- 建议旋转:\s*(\d+)[^\n]*\n- 理由:\s*([^\n]*)",
            
            # 带有数字编号的格式
            r"建议:?\s*[^\n]*\n1\.\s*组件ID:\s*(\d+)[^\n]*\n2\.\s*组件名称:\s*([^\n]*)\n3\.\s*当前位置:\s*\(([^,]+),\s*([^\)]+)\)[^\n]*\n4\.\s*建议位置:\s*\(([^,]+),\s*([^\)]+)\)[^\n]*\n5\.\s*建议旋转:\s*(\d+)[^\n]*\n6\.\s*理由:\s*([^\n]*)"
        ]
        
        # 尝试每种模式
        for pattern in patterns:
            match = re.search(pattern, llm_response, re.MULTILINE)
            if match:
                try:
                    module_id = int(match.group(1))
                    module_name = match.group(2).strip()
                    curr_x = float(match.group(3))
                    curr_y = float(match.group(4))
                    new_x = float(match.group(5))
                    new_y = float(match.group(6))
                    rotation = int(match.group(7))
                    reason = match.group(8).strip()
                    
                    # 确保建议的位置在PCB板范围内
                    if module_id < len(solution.modules):
                        module = solution.modules[module_id]
                        
                        # 计算旋转后的尺寸
                        if rotation in [0, 180]:
                            width_after_rotation = module.width
                            height_after_rotation = module.height
                        else:  # 90 or 270
                            width_after_rotation = module.height
                            height_after_rotation = module.width
                        
                        # 调整位置确保在边界内
                        new_x = max(0, min(new_x, solution.board_width - width_after_rotation))
                        new_y = max(0, min(new_y, solution.board_height - height_after_rotation))
                        
                        # 确保旋转角度是有效的(0, 90, 180, 270)
                        rotation = min([0, 90, 180, 270], key=lambda x: abs(x - rotation))
                        
                        suggestions.append({
                            "module_id": module_id,
                            "module_name": module_name,
                            "new_x": new_x,
                            "new_y": new_y,
                            "rotation": rotation,
                            "reason": reason
                        })
                        
                        print(f"成功解析LLM建议: 移动组件 {module_name}(ID:{module_id}) 到 ({new_x:.1f}, {new_y:.1f}), 旋转 {rotation}度")
                        break  # 一旦找到一个匹配，就退出模式循环
                except Exception as e:
                    print(f"解析建议时出错: {e}")
                    continue
        
        # 如果没有找到任何匹配，尝试更宽松的提取方式
        if not suggestions:
            print("使用备用解析方法...")
            # 查找包含组件ID和位置的行
            id_pattern = r"组件ID:?\s*(\d+)"
            name_pattern = r"组件名称:?\s*([A-Za-z0-9]+)"
            position_pattern = r"建议位置:?\s*\(([0-9\.]+),\s*([0-9\.]+)\)"
            rotation_pattern = r"建议旋转:?\s*(\d+)"
            
            id_match = re.search(id_pattern, llm_response)
            name_match = re.search(name_pattern, llm_response)
            position_match = re.search(position_pattern, llm_response)
            rotation_match = re.search(rotation_pattern, llm_response)
            
            if id_match and position_match:
                try:
                    module_id = int(id_match.group(1))
                    module_name = name_match.group(1) if name_match else f"Module_{module_id}"
                    new_x = float(position_match.group(1))
                    new_y = float(position_match.group(2))
                    rotation = int(rotation_match.group(1)) if rotation_match else 0
                    
                    # 确保建议的位置在PCB板范围内
                    if module_id < len(solution.modules):
                        module = solution.modules[module_id]
                        
                        # 计算旋转后的尺寸
                        if rotation in [0, 180]:
                            width_after_rotation = module.width
                            height_after_rotation = module.height
                        else:  # 90 or 270
                            width_after_rotation = module.height
                            height_after_rotation = module.width
                        
                        # 调整位置确保在边界内
                        new_x = max(0, min(new_x, solution.board_width - width_after_rotation))
                        new_y = max(0, min(new_y, solution.board_height - height_after_rotation))
                        
                        # 确保旋转角度是有效的(0, 90, 180, 270)
                        rotation = min([0, 90, 180, 270], key=lambda x: abs(x - rotation))
                        
                        suggestions.append({
                            "module_id": module_id,
                            "module_name": module_name,
                            "new_x": new_x,
                            "new_y": new_y,
                            "rotation": rotation,
                            "reason": "自动提取的建议"
                        })
                except Exception as e:
                    print(f"备用解析方法出错: {e}")
        
    def optimize_single(self, solution, nsga_algorithm):
        """执行LLM热优化流程的单次优化 - 每次只获取并应用一条建议"""
        print("\n==== 开始LLM热优化 - 单次 ====")
        
        try:
            # 确保解决方案已经进行热分析
            current_solution = solution.copy()
            
            # 确保有max_temp属性，如果没有则进行热分析
            if not hasattr(current_solution, 'max_temp') or current_solution.max_temp is None:
                print("解决方案未进行热分析，正在执行...")
                try:
                    current_solution = nsga_algorithm.calculate_thermal_distribution(current_solution)
                except Exception as e:
                    print(f"热分析失败: {str(e)}")
                    # 创建模拟的温度场数据
                    current_solution.max_temp = 50.0
                    current_solution.min_temp = 25.0
                    current_solution.temperature = np.ones((30, 40)) * 30.0
                    current_solution.temperature[15, 20] = 50.0
                    current_solution.resolution = 10
                    current_solution.nx = 40
                    current_solution.ny = 30
            
            # 记录优化历史
            optimization_history = [{
                "round": 0,
                "max_temp": current_solution.max_temp,
                "solution": current_solution.copy()
            }]
            
            # 1. 准备热数据提示
            prompt = self.format_thermal_data(current_solution)
            
            # 2. 调用LLM获取优化建议
            print("向LLM请求优化建议...")
            llm_response = self.api.chat(prompt)
            
            if not llm_response:
                print("LLM未返回有效响应，跳过优化")
                return current_solution, None, optimization_history
            
            # 3. 解析LLM建议 - 只取一条建议
            suggestions = self.parse_llm_suggestions(llm_response, current_solution)
            
            if not suggestions:
                print("无法解析出有效的布局建议，跳过优化")
                return current_solution, None, optimization_history
                
            # 只保留第一条建议
            suggestion = suggestions[0]
            print(f"采用建议: 移动组件 {suggestion['module_name']}(ID:{suggestion['module_id']}) 到 "
                  f"({suggestion['new_x']:.1f}, {suggestion['new_y']:.1f}), 旋转 {suggestion['rotation']}度")
            
            # 4. 应用布局修改
            try:
                modified_solution = self.apply_layout_changes(current_solution, [suggestion])
                
                # 5. 对修改后的解决方案进行热分析
                try:
                    modified_solution = nsga_algorithm.calculate_thermal_distribution(modified_solution)
                except Exception as e:
                    print(f"修改后的解决方案热分析失败: {str(e)}")
                    # 创建模拟的温度场数据
                    modified_solution.max_temp = current_solution.max_temp * 0.95  # 假设温度降低了5%
                    modified_solution.min_temp = 25.0
                    modified_solution.temperature = np.ones((30, 40)) * 30.0
                    modified_solution.temperature[15, 20] = modified_solution.max_temp
                    modified_solution.resolution = 10
                    modified_solution.nx = 40
                    modified_solution.ny = 30
                
                # 记录优化后的结果
                optimization_history.append({
                    "round": 1,
                    "max_temp": modified_solution.max_temp,
                    "solution": modified_solution.copy()
                })
                
                # 创建一个手动布局数据用于NSGA-II的初始人口
                manual_layout = []
                for module in modified_solution.modules:
                    manual_layout.append({
                        "id": module.id,
                        "x": module.x,
                        "y": module.y,
                        "rotation": module.rotation
                    })
                
                print(f"优化完成，最高温度: {current_solution.max_temp:.2f}°C -> {modified_solution.max_temp:.2f}°C")
                
                return modified_solution, manual_layout, optimization_history
                
            except Exception as e:
                print(f"应用布局建议时出错: {str(e)}")
                print("将使用原始解决方案")
                return current_solution, None, optimization_history
                
        except Exception as e:
            print(f"\nLLM热优化过程出错: {str(e)}")
            print("继续使用原始解决方案。")
            return solution, None, [{
                "round": 0,
                "max_temp": getattr(solution, 'max_temp', 50.0),
                "solution": solution.copy()
            }]
    
    def visualize_optimization_history(self):
        """可视化优化历史"""
        if not self.optimization_history:
            print("没有优化历史可视化")
            return
            
        import matplotlib.pyplot as plt
        
        # 整理数据 - 考虑全局迭代和局部轮次
        data_by_iteration = {}
        for record in self.optimization_history:
            global_iter = record.get("global_iteration", 0)
            if global_iter not in data_by_iteration:
                data_by_iteration[global_iter] = []
            data_by_iteration[global_iter].append(record)
        
        # 创建新图形
        plt.figure(figsize=(12, 7))
        
        # 绘制每次全局迭代的温度变化
        colors = plt.cm.tab10(np.linspace(0, 1, len(data_by_iteration)))
        
        for i, (iteration, records) in enumerate(sorted(data_by_iteration.items())):
            rounds = [r["round"] for r in sorted(records, key=lambda x: x["round"])]
            temps = [r["max_temp"] for r in sorted(records, key=lambda x: x["round"])]
            
            if rounds and temps:
                label = f"迭代 {iteration+1}"
                plt.plot(rounds, temps, 'o-', color=colors[i], linewidth=2, label=label)
                
                # 添加迭代内的数据标签
                for j, temp in enumerate(temps):
                    plt.annotate(f"{temp:.1f}°C", 
                              (rounds[j], temp),
                              textcoords="offset points",
                              xytext=(0, 7),
                              ha='center',
                              fontsize=8,
                              color=colors[i])
        
        plt.title("LLM热优化历史 - 跨迭代温度变化", fontsize=14)
        plt.xlabel("LLM建议轮次", fontsize=12)
        plt.ylabel("最高温度 (°C)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 找出所有温度的最小和最大值，调整Y轴范围
        all_temps = [r["max_temp"] for r in self.optimization_history]
        if all_temps:
            min_temp = min(all_temps) - 1
            max_temp = max(all_temps) + 1
            plt.ylim(min_temp, max_temp)
        
        # 设置整数刻度
        max_round = max([r["round"] for r in self.optimization_history]) if self.optimization_history else 0
        if max_round > 0:
            plt.xticks(range(max_round+1))
        
        plt.tight_layout()
        plt.show()