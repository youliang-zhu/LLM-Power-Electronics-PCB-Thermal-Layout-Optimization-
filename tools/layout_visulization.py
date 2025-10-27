import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from functools import wraps
import io

class PCBVisualizer:
    """PCB布局可视化类，专注于显示最终布局和热分布结果"""
    
    def __init__(self, save_path=None):
        """
        初始化PCB可视化器
        
        参数:
            save_path (str, optional): 保存图像的路径，如果为None则只显示不保存
        """
        self.save_path = save_path
        
    def visualize_layout(self, solution, min_distance=10):
        """
        可视化PCB布局 - 显示组件、连线和连线交叉点
        
        参数:
            solution: 包含模块和网络的解决方案对象
            min_distance: 组件间最小距离
        """
        # 创建图形和坐标轴
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # 绘制PCB板边界
        board = patches.Rectangle((0, 0), solution.board_width, solution.board_height, 
                                  linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(board)
        
        # 使用鲜明的颜色方案
        module_colors = plt.cm.tab20(np.linspace(0, 1, len(solution.modules)))
        net_colors = plt.cm.Set2(np.linspace(0, 1, len(solution.nets)))
        
        # 存储引脚位置信息 - 用于后续绘制连线
        pin_positions = {}  # 格式: {(module_id, pin_id): (x, y, module_name)}
        
        # 第一次遍历：绘制所有模块
        for i, module in enumerate(solution.modules):
            width, height = module.get_dimensions()
            
            # 矩形模块
            rect = patches.Rectangle((module.x, module.y), width, height, 
                                    linewidth=1, edgecolor='black', facecolor=module_colors[i], 
                                    alpha=0.7)
            ax.add_patch(rect)
            
            # 添加模块标签
            center_x = module.x + width/2
            center_y = module.y + height/2
            ax.text(center_x, center_y, f"{module.name}\n({module.id})", 
                   ha='center', va='center', fontsize=10, weight='bold')
        
        # 第二次遍历：绘制所有引脚
        for i, module in enumerate(solution.modules):
            # 收集所有引脚位置
            for pin in module.pins:
                x_offset, y_offset, pin_id = pin
                pin_pos = module.get_pin_position(pin_id)
                if pin_pos:
                    pin_x, pin_y = pin_pos
                    # 使用更大的点和更鲜明的颜色标注引脚
                    ax.plot(pin_x, pin_y, 'o', markersize=7, color='red')
                    
                    # 添加引脚ID标签
                    ax.text(pin_x+5, pin_y+5, str(pin_id), fontsize=9, 
                          ha='center', va='center', color='black', weight='bold',
                          bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
                    
                    # 存储引脚位置及其所属模块
                    pin_positions[(module.id, pin_id)] = (pin_x, pin_y, module.name)
        
        # 存储所有线段用于检测和标记交叉点
        all_segments = []
        
        # 第三次遍历：绘制网络连接
        for net_idx, net in enumerate(solution.nets):
            # 为每个网络使用固定的颜色和线型
            line_color = net_colors[net_idx % len(net_colors)]
            line_style = ['-', '--', '-.', ':'][net_idx % 4]
            
            # 获取线段
            wire_segments = net.get_wire_segments(solution.modules)
            
            # 绘制每个线段
            for segment in wire_segments:
                (x1, y1), (x2, y2) = segment
                # 绘制连线
                ax.plot([x1, x2], [y1, y2], 
                    line_style, color=line_color, alpha=0.8, linewidth=2+net.weight/2)
                
                # 在连线中间添加网络名称标签
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                # 计算标签放置角度，使其沿连线方向
                angle = np.degrees(np.arctan2(y2-y1, x2-x1))
                if angle > 90 or angle < -90:
                    angle += 180  # 确保文字不会上下颠倒
                
                # 添加带背景的网络标签
                bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec=line_color, alpha=0.8)
                ax.text(mid_x, mid_y, net.name, fontsize=8, ha='center', va='center',
                    rotation=angle, bbox=bbox_props, rotation_mode='anchor')
                
            # 收集所有线段用于后续交叉点检测
            for segment in wire_segments:
                all_segments.append((net.id, segment))
        
        # 检测并标记交叉点
        crossings = []
        for i in range(len(all_segments)):
            net1_id, seg1 = all_segments[i]
            for j in range(i + 1, len(all_segments)):
                net2_id, seg2 = all_segments[j]
                
                # 同一网络内的线段交叉不计算
                if net1_id == net2_id:
                    continue
                
                if self.check_segments_intersection(seg1, seg2):
                    # 计算交点坐标
                    intersection_point = self.calculate_intersection_point(seg1, seg2)
                    if intersection_point:
                        crossings.append(intersection_point)
        
        # 标记交叉点
        for x, y in crossings:
            ax.plot(x, y, 'X', markersize=10, color='red', markeredgewidth=2)
        
        # 添加交叉点总数标注
        ax.text(10, solution.board_height - 20, f"Total Crossings: {len(crossings)}", 
               fontsize=12, color='red', weight='bold',
               bbox=dict(facecolor='white', alpha=0.8))
        
        # 设置图表参数
        plt.xlim(0, solution.board_width)
        plt.ylim(0, solution.board_height)
        plt.title('PCB Layout Visualization', fontsize=14)
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        
        # 保存图像（如果指定了路径）
        if self.save_path:
            try:
                save_filename = f"{self.save_path}/pcb_layout.png"
                plt.savefig(save_filename, dpi=300, bbox_inches='tight')
                print(f"PCB布局图已保存至: {save_filename}")
            except Exception as e:
                print(f"保存PCB布局图时出错: {str(e)}")
        
        plt.show()
    
    def visualize_pareto_front(self, pareto_front):
        """
        可视化帕累托前沿
        
        参数:
            pareto_front: 帕累托最优解列表
        """
        if not pareto_front:
            print("No solutions found.")
            return
        
        objectives = np.array([solution.objectives for solution in pareto_front])
        
        # 创建目标函数名称标签
        obj_names = ["Wire Length", "Wire Crossings", "Area Utilization", "Overlap"]
        
        plt.figure(figsize=(16, 12))
        
        # 计算所需的子图数量和布局
        num_pairs = len(objectives[0]) * (len(objectives[0]) - 1) // 2
        rows = int(np.ceil(np.sqrt(num_pairs)))
        cols = int(np.ceil(num_pairs / rows))
        
        plot_idx = 1
        # 生成所有目标函数对的散点图
        for i in range(len(objectives[0])):
            for j in range(i + 1, len(objectives[0])):
                plt.subplot(rows, cols, plot_idx)
                plt.scatter(objectives[:, i], objectives[:, j])
                plt.xlabel(obj_names[i])
                plt.ylabel(obj_names[j])
                plt.title(f'{obj_names[i]} vs {obj_names[j]}')
                plt.grid(True)
                plot_idx += 1
        
        plt.tight_layout()
        
        # 保存图像（如果指定了路径）
        if self.save_path:
            try:
                save_filename = f"{self.save_path}/pareto_front.png"
                plt.savefig(save_filename, dpi=300, bbox_inches='tight')
                print(f"帕累托前沿图已保存至: {save_filename}")
            except Exception as e:
                print(f"保存帕累托前沿图时出错: {str(e)}")
        
        plt.show()
    
    def visualize_optimization_history(self, optimization_history):
        """
        可视化优化历史
        
        参数:
            optimization_history: 优化历史记录列表
        """
        if not optimization_history:
            print("没有优化历史可视化")
            return
            
        # 整理数据 - 考虑全局迭代和局部轮次
        data_by_iteration = {}
        for record in optimization_history:
            global_iter = record.get("global_iteration", 0)
            if global_iter not in data_by_iteration:
                data_by_iteration[global_iter] = []
            data_by_iteration[global_iter].append(record)
        
        # 创建新图形
        plt.figure(figsize=(12, 7))
        
        # 绘制每次全局迭代的温度变化
        colors = plt.cm.tab10(np.linspace(0, 1, len(data_by_iteration)))
        
        for i, (iteration, records) in enumerate(sorted(data_by_iteration.items())):
            sorted_records = sorted(records, key=lambda x: x["round"])
            rounds = [r["round"] for r in sorted_records]
            temps = [r["max_temp"] for r in sorted_records]
            
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
        all_temps = [r["max_temp"] for r in optimization_history]
        if all_temps:
            min_temp = min(all_temps) - 1
            max_temp = max(all_temps) + 1
            plt.ylim(min_temp, max_temp)
        
        # 设置整数刻度
        max_round = max([r["round"] for r in optimization_history]) if optimization_history else 0
        if max_round > 0:
            plt.xticks(range(max_round+1))
        
        plt.tight_layout()
        
        # 保存图像（如果指定了路径）
        if self.save_path:
            try:
                save_filename = f"{self.save_path}/optimization_history.png"
                plt.savefig(save_filename, dpi=300, bbox_inches='tight')
                print(f"优化历史图已保存至: {save_filename}")
            except Exception as e:
                print(f"保存优化历史图时出错: {str(e)}")
        
        plt.show()
    
    @staticmethod
    def check_segments_intersection(seg1, seg2):
        """
        检查两个线段是否相交
        
        参数:
            seg1, seg2: 线段，每个线段由两个点表示 ((x1,y1), (x2,y2))
            
        返回:
            bool: 如果线段相交则为True，否则为False
        """
        def ccw(A, B, C):
            """判断三点的顺序是否为逆时针"""
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        # 提取线段端点
        A, B = seg1
        C, D = seg2
        
        # 使用叉积判断线段是否相交
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    
    @staticmethod
    def calculate_intersection_point(seg1, seg2):
        """
        计算两条线段的交点坐标
        
        参数:
            seg1, seg2: 线段，每个线段由两个点表示 ((x1,y1), (x2,y2))
            
        返回:
            tuple or None: 交点坐标 (x,y) 或者None（如果不相交）
        """
        (x1, y1), (x2, y2) = seg1
        (x3, y3), (x4, y4) = seg2
        
        # 使用线性代数求解两直线交点
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if abs(denom) > 1e-10:  # 避免除以零
            ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
            intersection_x = x1 + ua * (x2 - x1)
            intersection_y = y1 + ua * (y2 - y1)
            
            return (intersection_x, intersection_y)
        return None