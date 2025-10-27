import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
from functools import wraps

from library.layout_visulization import *
from library.pcb_layout_model import *
from library.llm_thermal_optimizer import *
from library.NSGA_algorithm import *
from library.thermal_calculation import *

from test import *

def main():
    """主函数 - 运行PCB布局优化，添加LLM热优化迭代过程"""
    
    # 初始化电子元件和网络连接
    modules, nets = initialize_components()
    
    # 设置PCB板尺寸和算法参数
    board_width = 800
    board_height = 600
    pop_size = 500   # 种群大小
    max_gen = 5     # 最大代数

    # 创建热分析器
    thermal_calculator = ThermalCalculator(board_width, board_height)
    
    # 添加手动布局
    manual_layout = [
        {"id": 0, "x": 638, "y": 299, "rotation": 0},   # C1
        {"id": 1, "x": 316, "y": 301, "rotation": 0},   # C2
        {"id": 2, "x": 95, "y": 263, "rotation": 90},   # Y1
        {"id": 3, "x": 197, "y": 108, "rotation": 270}, # R1
        {"id": 4, "x": 700, "y": 106, "rotation": 270}, # R2
        {"id": 5, "x": 524, "y": 104, "rotation": 270}, # R3
        {"id": 6, "x": 346, "y": 104, "rotation": 270}, # R4
        {"id": 7, "x": 276, "y": 479, "rotation": 0},   # Q1
        {"id": 8, "x": 657, "y": 473, "rotation": 0}    # Q2
    ]
    
    # 创建NSGA-II实例
    nsga2 = NSGA2(pop_size, max_gen, board_width, board_height)
    
    # 查看手动布局的效果
    check_manual_layout(board_width, board_height, modules, nets, manual_layout)
    
    # 使用初始手动布局运行NSGA-II算法 - 重叠作为目标函数之一
    current_layout = manual_layout
    
    # ==== 修改部分：实现迭代优化 ====
    thermal_optimizer = LLMThermalOptimizer(max_rounds=5, temp_threshold=0.5)
    all_optimization_history = []
    
    # 主迭代循环 - 运行5轮NSGA-II + LLM优化
    initial_solution = None
    final_solution = None
    
    for iteration in range(3):
        print(f"\n\n{'='*20} 迭代优化循环 {iteration+1}/5 {'='*20}")
        
        # 使用当前布局运行NSGA-II
        print(f"\n[迭代 {iteration+1}] 运行NSGA-II算法...")
        nsga_pop_size = 500 if iteration == 0 else 250  # 第一轮使用更大种群
        nsga_max_gen = 5 if iteration == 0 else 3      # 后续轮次减少代数
        
        # 创建新的NSGA-II实例，针对当前迭代适当调整参数
        nsga2 = NSGA2(nsga_pop_size, nsga_max_gen, board_width, board_height)
        
        # 运行NSGA-II，使用当前布局作为种群初始化的起点
        pareto_front = nsga2.run(modules, nets, current_layout)
        
        # 检查是否找到解决方案
        if not pareto_front:
            print(f"[迭代 {iteration+1}] NSGA-II未找到可行的解决方案，迭代中止。")
            break
        
        # 选择最佳折衷解
        def weighted_sum(solution, weights=[0.35, 0.3, 0.05, 0.3]):
            obj_max = [max(sol.objectives[i] for sol in pareto_front) for i in range(4)]
            obj_min = [min(sol.objectives[i] for sol in pareto_front) for i in range(4)]
            
            norm_obj = [(solution.objectives[i] - obj_min[i]) / (obj_max[i] - obj_min[i] + 1e-10)
                        for i in range(4)]
            
            return sum(w * obj for w, obj in zip(weights, norm_obj))
        
        best_compromise = min(pareto_front, key=lambda x: weighted_sum(x))
        
        # 对最佳折衷解进行热分析
        print(f"\n[迭代 {iteration+1}] 对NSGA最佳折衷解进行热分析...")
        try:
            best_compromise = thermal_calculator.calculate_thermal_distribution(best_compromise)     
            print("6666666666666666666666666666666666666666666666666666666")       
            # 保存第一次迭代的解决方案作为初始解
            if iteration == 0:
                initial_solution = best_compromise
            
            # 如果这是最后一次迭代，保存为最终解并跳过LLM优化
            if iteration == 2:
                print("这是最后一轮迭代，跳过LLM优化步骤。")
                final_solution = best_compromise
                break
            
            # 使用修改后的LLM优化器获取下一步布局建议 - 调用optimize_single方法
            print(f"\n[迭代 {iteration+1}] 使用LLM进行热优化分析...")
            modified_solution, new_layout, round_history = thermal_optimizer.optimize_single(best_compromise, nsga2, thermal_calculator)
            
            # 如果是最后一轮，保存最终解
            if iteration == 2:
                final_solution = modified_solution
            
            # 添加本轮历史记录
            for record in round_history:
                if record.get("round") > 0:  # 只添加新记录
                    record["global_iteration"] = iteration
                    all_optimization_history.append(record)
            
            # 如果LLM提供了新布局，使用它作为下一轮NSGA-II的输入
            if new_layout:
                current_layout = new_layout
                print(f"[迭代 {iteration+1}] LLM提供了新布局，将用于下一轮NSGA-II")
            else:
                # 如果LLM没有提供有效建议，使用当前最佳解的布局
                current_layout = []
                for module in best_compromise.modules:
                    current_layout.append({
                        "id": module.id,
                        "x": module.x,
                        "y": module.y,
                        "rotation": module.rotation
                    })
                print(f"[迭代 {iteration+1}] 使用当前最佳解的布局作为下一轮输入")
        
        except Exception as e:
            print(f"[迭代 {iteration+1}] 出错: {str(e)}")
            print("计算有误！热分析计算失败，无法继续优化。")
            break
    
    # ==== 结束迭代优化 ====
    
    # 显示优化结果
    print("\n\n" + "="*50)
    print("迭代优化完成! 最终结果:")
    
    if initial_solution is None or final_solution is None:
        print("优化过程未能完成，未找到有效解决方案。")
        return
        
    # 确保初始解和最终解都有温度数据
    print(not hasattr(initial_solution, 'max_temp')==True, "\n")
    print(initial_solution.max_temp==None)
    if not hasattr(initial_solution, 'max_temp') or initial_solution.max_temp is None:
        print("初始解缺少温度数据，无法完成结果对比。")
        return
        
    if not hasattr(final_solution, 'max_temp') or final_solution.max_temp is None:
        print("最终解缺少温度数据，无法完成结果对比。")
        return
    
    print("\n=== 优化前后对比 ===")
    print(f"初始最高温度: {initial_solution.max_temp:.2f}°C")
    print(f"最终最高温度: {final_solution.max_temp:.2f}°C")
    temp_improvement = initial_solution.max_temp - final_solution.max_temp
    print(f"温度降低: {temp_improvement:.2f}°C")
    print(f"改善百分比: {(temp_improvement / initial_solution.max_temp * 100):.2f}%")
    
    # 目标函数对比
    print("\n目标函数对比:")
    print(f"总线长 - 初始: {initial_solution.objectives[0]:.2f}, 最终: {final_solution.objectives[0]:.2f}")
    print(f"交叉点 - 初始: {initial_solution.objectives[1]:.2f}, 最终: {final_solution.objectives[1]:.2f}")
    print(f"面积利用率 - 初始: {1-initial_solution.objectives[2]:.2f}, 最终: {1-final_solution.objectives[2]:.2f}")
    print(f"重叠面积 - 初始: {initial_solution.objectives[3]:.2f}, 最终: {final_solution.objectives[3]:.2f}")
    
    # 可视化最终布局
    print("\n显示最终优化布局...")
    visualizer = PCBVisualizer()
    visualizer.visualize_layout(final_solution, nsga2.min_distance)
    components_data = convert_solution_to_component_data(final_solution)
    print(components_data)
    # thermal_visualize(components_data, final_solution)

def initialize_components():
    # 使用提供的尺寸初始化组件
    modules = [
        Module(0, 400, 300, name="C1", width=230, height=130, rotation=0),  # 电容1
        Module(1, 450, 300, name="C2", width=230, height=130, rotation=0),  # 电容2
        Module(2, 300, 100, name="Y1", width=154, height=54, rotation=0),  # 晶振
        Module(3, 350, 200, name="R1", width=232, height=60, rotation=0),  # 电阻1
        Module(4, 400, 200, name="R2", width=232, height=60, rotation=0),  # 电阻2
        Module(5, 350, 250, name="R3", width=232, height=60, rotation=0),  # 电阻3
        Module(6, 400, 250, name="R4", width=232, height=60, rotation=0),  # 电阻4
        Module(7, 200, 200, 'semicircle_arch', name="Q1", width=79, height=98, rotation=0),  # 晶体管1
        Module(8, 200, 300, 'semicircle_arch', name="Q2", width=79, height=98, rotation=0)   # 晶体管2
    ]

    # 为组件添加引脚（相对偏移）
    # C1引脚 - 确保引脚在元件内部
    modules[0].add_pin(-110, 0, 0)  # 第1个引脚, 左侧
    modules[0].add_pin(110, 0, 1)   # 第2个引脚, 右侧
    
    # C2引脚
    modules[1].add_pin(-110, 0, 0)  # 第1个引脚, 左侧
    modules[1].add_pin(110, 0, 1)   # 第2个引脚, 右侧
    
    # Y1引脚
    modules[2].add_pin(-70, 0, 0)   # 第1个引脚, 顶部中间偏左
    modules[2].add_pin(70, 0, 1)    # 第2个引脚, 顶部中间偏右
    
    # R1引脚
    modules[3].add_pin(-110, 0, 0)  # 第1个引脚, 左侧
    modules[3].add_pin(110, 0, 1)   # 第2个引脚, 右侧
    
    # R2引脚
    modules[4].add_pin(-110, 0, 0)  # 第1个引脚, 左侧
    modules[4].add_pin(110, 0, 1)   # 第2个引脚, 右侧
    
    # R3引脚
    modules[5].add_pin(-110, 0, 0)  # 第1个引脚, 左侧
    modules[5].add_pin(110, 0, 1)   # 第2个引脚, 右侧
    
    # R4引脚
    modules[6].add_pin(-110, 0, 0)  # 第1个引脚, 左侧
    modules[6].add_pin(110, 0, 1)   # 第2个引脚, 右侧
    
    # Q1引脚 - 三个引脚均匀分布
    modules[7].add_pin(0, -30, 0)   # 第1个引脚, 底部
    modules[7].add_pin(-25, 15, 1)  # 第2个引脚, 左上
    modules[7].add_pin(25, 15, 2)   # 第3个引脚, 右上
    
    # Q2引脚
    modules[8].add_pin(0, -30, 0)   # 第1个引脚, 底部
    modules[8].add_pin(-25, 15, 1)  # 第2个引脚, 左上
    modules[8].add_pin(25, 15, 2)   # 第3个引脚, 右上

    # 创建网络连接 - 每个连接指定模块ID和引脚ID
    nets = [
        Net(0, [(4, 1), (6, 1), (5, 1), (3, 1), (2, 0)], name="12V", weight=2.0),
        Net(1, [(2, 1), (8, 0), (7, 0)], name="GND", weight=2.0),
        Net(2, [(4, 0), (8, 1)], name="NetQ2_2"),
        Net(3, [(3, 0), (1, 0), (7, 1)], name="NetC2_1"),
        Net(4, [(6, 0), (0, 1), (1, 1), (8, 2)], name="NetC1_2"),
        Net(5, [(5, 0), (0, 0), (7, 2)], name="NetC1_1")
    ]
    
    return modules, nets

def check_manual_layout(board_width, board_height, modules, nets, manual_layout):
     # 查看手动布局的效果
    print("\n手动布局的效果:")
    # 创建一个Solution对象并应用手动布局
    manual_solution = Solution(modules, nets, board_width, board_height)
    for item in manual_layout:
        module_id = item["id"]
        if module_id < len(manual_solution.modules):
            manual_solution.modules[module_id].x = item["x"]
            manual_solution.modules[module_id].y = item["y"]
            manual_solution.modules[module_id].rotation = item["rotation"]
    
    # 计算手动布局的目标函数
    manual_solution.calc_objectives()
    
    # 输出目标函数值
    print(f"总线长: {manual_solution.objectives[0]:.2f}")
    print(f"连线交叉点数: {manual_solution.objectives[1]:.2f}")
    print(f"面积利用率: {1 - manual_solution.objectives[2]:.2f}")
    print(f"重叠面积: {manual_solution.objectives[3]:.2f}")
    
    # 检查布局是否在PCB板内
    is_in_board = all(
        m.x >= 0 and m.y >= 0 and 
        m.x + m.get_dimensions()[0] <= board_width and 
        m.y + m.get_dimensions()[1] <= board_height 
        for m in manual_solution.modules
    )
    print(f"所有组件都在PCB板内: {'是' if is_in_board else '否'}")

    return True

if __name__ == "__main__":
    main()