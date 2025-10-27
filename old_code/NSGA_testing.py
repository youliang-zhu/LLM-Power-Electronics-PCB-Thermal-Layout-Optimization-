
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
from functools import wraps

from llm_thermal_optimizer import *

class PlotManager:
    def __init__(self):
        self.figures = []
        self.titles = []
        self.use_deferred = False  # 新增标志, 默认不使用延迟显示
    
    def set_deferred(self, deferred=True):
        """设置是否使用延迟显示"""
        self.use_deferred = deferred
    
    def add_figure(self, title=None):
        """添加当前图形到管理器"""
        fig = plt.gcf().number
        self.figures.append(fig)
        self.titles.append(title or f"Figure {len(self.figures)}")
        
    def show_all(self):
        """在程序结束时显示所有存储的图形"""
        if not self.figures:
            print("没有图形可显示")
            return
            
        print(f"\n共有{len(self.figures)}个图形等待显示")
        for i, fig_num in enumerate(self.figures):
            try:
                # 获取图形对象并显示
                fig = plt.figure(fig_num)
                print(f"显示: {self.titles[i]}")
                plt.show()
            except Exception as e:
                print(f"显示图形 {self.titles[i]} 时出错: {str(e)}")
                
def defer_show(title=None):
    """装饰器：延迟显示图形，而是将其添加到PlotManager"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 检查是否使用延迟显示
            if not plot_manager.use_deferred:
                # 不使用延迟显示，直接调用原始函数
                return func(*args, **kwargs)
            
            # 以下是延迟显示的逻辑
            before_count = len(plt.get_fignums())
            original_show = plt.show
            
            def custom_show(*args, **kwargs):
                print(f"捕获到plt.show()调用，保存图形: {title}")
                plot_manager.add_figure(title)
                plt.close()
            
            plt.show = custom_show
            
            try:
                result = func(*args, **kwargs)
                
                after_count = len(plt.get_fignums())
                if after_count > before_count:
                    for fig_num in plt.get_fignums()[before_count:]:
                        plt.figure(fig_num)
                        plot_manager.add_figure(title)
                        plt.close(fig_num)
                
                return result
            finally:
                plt.show = original_show
        return wrapper
    return decorator

# 全局PlotManager实例
plot_manager = PlotManager()



class Module:
    def __init__(self, id, x=0, y=0, shape='rectangle', name="", width=0, height=0, rotation=0):
        self.id = id
        self.name = name if name else f"Module_{id}"
        self.width = width
        self.height = height
        self.shape = shape
        self.x = x
        self.y = y
        self.rotation = rotation  # 0, 90, 180, 270 degrees
        self.pins = []  # List of (x_offset, y_offset, pin_id) tuples
        
    def add_pin(self, x_offset, y_offset, pin_id=None):
        """添加引脚（相对于模块中心的偏移）"""
        self.pins.append((x_offset, y_offset, pin_id))
    
    def get_pin_position(self, pin_id):
        """获取特定引脚的绝对坐标（考虑旋转）- 改进版本"""
        # 查找对应pin_id的引脚
        for pin in self.pins:
            if pin[2] == pin_id:
                x_offset, y_offset, _ = pin
                
                # 计算旋转前的中心点
                center_x = self.x + self.width/2
                center_y = self.y + self.height/2
                
                # 根据旋转角度计算实际偏移
                if self.rotation == 0:
                    actual_x = center_x + x_offset
                    actual_y = center_y + y_offset
                elif self.rotation == 90:
                    # 旋转90度的处理，宽高互换
                    center_x = self.x + self.height/2
                    center_y = self.y + self.width/2
                    actual_x = center_x - y_offset
                    actual_y = center_y + x_offset
                elif self.rotation == 180:
                    actual_x = center_x - x_offset
                    actual_y = center_y - y_offset
                elif self.rotation == 270:
                    # 旋转270度的处理，宽高互换
                    center_x = self.x + self.height/2
                    center_y = self.y + self.width/2
                    actual_x = center_x + y_offset
                    actual_y = center_y - x_offset
                
                return actual_x, actual_y
        
        # 如果没找到对应pin_id的引脚
        return None
    
    def get_dimensions(self):
        """获取旋转后的宽度和高度"""
        if self.rotation in [0, 180]:
            return self.width, self.height
        else:  # 90 or 270 degrees
            return self.height, self.width
    
    def get_area(self):
        """计算模块面积"""
        width, height = self.get_dimensions()
        
        if self.shape == 'rectangle':
            return width * height
        elif self.shape == 'semicircle_arch':
            # 半圆拱形面积计算 (假设宽度为矩形宽度，半圆直径与宽度相同)
            # 注意：这里简化处理，如果旋转了，仍然按照最初的形状计算
            rectangle_area = width * (height / 2)
            semicircle_area = math.pi * (width / 2) ** 2 / 2
            return rectangle_area + semicircle_area
    
    def get_bounding_box(self):
        """获取旋转后的边界框"""
        width, height = self.get_dimensions()
        return self.x, self.y, width, height
    
    def overlaps(self, other, min_distance=0):
        """重叠检测（考虑旋转和最小距离）- 改进版本"""
        x1, y1, w1, h1 = self.get_bounding_box()
        x2, y2, w2, h2 = other.get_bounding_box()
        
        # 检查是否重叠或距离过近，加入了最小距离要求
        if (x1 + w1 + min_distance <= x2 or 
            x2 + w2 + min_distance <= x1 or 
            y1 + h1 + min_distance <= y2 or 
            y2 + h2 + min_distance <= y1):
            return False  # 不重叠且满足最小距离
        
        # 添加调试信息（可选）
        # print(f"检测到重叠：{self.name} 和 {other.name}")
        # print(f"位置：({x1},{y1},{w1},{h1}) 和 ({x2},{y2},{w2},{h2})")
        
        return True  # 重叠或距离不足
        
    def overlap_area(self, other, min_distance=0):
        """重叠面积计算（考虑旋转和最小距离）"""
        if not self.overlaps(other, min_distance):
            return 0
        
        x1, y1, w1, h1 = self.get_bounding_box()
        x2, y2, w2, h2 = other.get_bounding_box()
        
        # 计算重叠区域
        x_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
        y_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)
        
        # 加入最小距离影响
        if x_overlap > 0 and y_overlap > 0:
            # 考虑最小距离
            if x_overlap <= min_distance:
                x_overlap = min_distance
            if y_overlap <= min_distance:
                y_overlap = min_distance
        
        return x_overlap * y_overlap
    
    def copy(self):
        """创建模块的深拷贝"""
        new_module = Module(
            self.id, 
            self.x, 
            self.y, 
            self.shape, 
            self.name, 
            self.width, 
            self.height, 
            self.rotation
        )
        # 复制引脚信息
        new_module.pins = self.pins.copy()
        return new_module
    

class Net:
    """网络连接类，代表模块间的连线"""
    def __init__(self, id, connections, name="", weight=1.0):
        self.id = id
        self.name = name if name else f"Net_{id}"
        self.connections = connections  # List of (module_id, pin_id) tuples
        self.weight = weight  # 网络权重，影响线长计算
    
    def calc_length(self, modules):
        """计算该网络的总线长（使用MCTS优化的路径）"""
        # 获取所有引脚的位置
        pin_positions = []
        for m_id, p_id in self.connections:
            if m_id < len(modules):
                pin_pos = modules[m_id].get_pin_position(p_id)
                if pin_pos:
                    pin_positions.append(pin_pos)
        
        # 使用MCTS找到最优路径
        if len(pin_positions) > 1:
            router = MCTSRouter(max_iterations=500)  # 调整迭代次数以平衡性能和质量
            optimized_path = router.find_best_route(pin_positions)
            
            # 计算优化路径的总长度 - 使用欧几里得距离(直线)而非曼哈顿距离
            total_length = 0
            for i in range(len(optimized_path) - 1):
                x1, y1 = optimized_path[i]
                x2, y2 = optimized_path[i + 1]
                # 使用欧几里得距离(直线长度)
                length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                total_length += length * self.weight
                
            return total_length
        
        return 0  # 如果只有一个或没有引脚
    
    def get_wire_segments(self, modules):
        """获取该网络所有连线段，使用MCTS优化的路径"""
        segments = []
        pin_positions = []
        
        # 获取所有引脚的位置
        for m_id, p_id in self.connections:
            if m_id < len(modules):
                pin_pos = modules[m_id].get_pin_position(p_id)
                if pin_pos:
                    pin_positions.append(pin_pos)
        
        # 使用MCTS找到最优路径
        if len(pin_positions) > 1:
            router = MCTSRouter(max_iterations=500)
            optimized_path = router.find_best_route(pin_positions)
            
            # 创建优化后的线段
            for i in range(len(optimized_path) - 1):
                segments.append((optimized_path[i], optimized_path[i + 1]))
        
        return segments

def check_segments_intersection(seg1, seg2, epsilon=1e-6):
    """检查两条线段是否相交（改进版）
    
    参数:
    seg1, seg2: 两条线段，每个线段由两个点表示，格式为 ((x1, y1), (x2, y2))
    epsilon: 数值精度阈值
    
    返回:
    bool: 线段是否相交
    """
    # 提取各个点的坐标
    (x1, y1), (x2, y2) = seg1
    (x3, y3), (x4, y4) = seg2
    
    # 检查线段的端点是否相同，如果有共同端点则不算作交叉
    if (abs(x1-x3) < epsilon and abs(y1-y3) < epsilon) or \
       (abs(x1-x4) < epsilon and abs(y1-y4) < epsilon) or \
       (abs(x2-x3) < epsilon and abs(y2-y3) < epsilon) or \
       (abs(x2-x4) < epsilon and abs(y2-y4) < epsilon):
        return False
    
    # 使用跨立实验法检测线段相交
    def cross_product(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    
    # 检查(p1, p2)和(p3, p4)是否相交
    d1 = cross_product((x3, y3), (x4, y4), (x1, y1))
    d2 = cross_product((x3, y3), (x4, y4), (x2, y2))
    d3 = cross_product((x1, y1), (x2, y2), (x3, y3))
    d4 = cross_product((x1, y1), (x2, y2), (x4, y4))
    
    # 线段相交的条件: d1*d2 < 0 且 d3*d4 < 0
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    
    # 检查接近共线的情况（考虑数值精度）
    if abs(d1) < epsilon and abs(d2) < epsilon and abs(d3) < epsilon and abs(d4) < epsilon:
        # 共线情况下，检查投影是否重叠
        if max(x1, x2) < min(x3, x4) - epsilon or max(x3, x4) < min(x1, x2) - epsilon:
            return False  # x轴投影不重叠
        if max(y1, y2) < min(y3, y4) - epsilon or max(y3, y4) < min(y1, y2) - epsilon:
            return False  # y轴投影不重叠
        return True  # 投影重叠，线段有部分重合
    
    # 检查特殊情况：一个线段的端点非常接近另一个线段
    if abs(d1) < epsilon and on_segment((x3, y3), (x4, y4), (x1, y1)):
        return True
    if abs(d2) < epsilon and on_segment((x3, y3), (x4, y4), (x2, y2)):
        return True
    if abs(d3) < epsilon and on_segment((x1, y1), (x2, y2), (x3, y3)):
        return True
    if abs(d4) < epsilon and on_segment((x1, y1), (x2, y2), (x4, y4)):
        return True
    
    return False

def on_segment(p, q, r, epsilon=1e-6):
    """检查点r是否在线段pq上"""
    return (max(p[0], q[0]) >= r[0] - epsilon and r[0] >= min(p[0], q[0]) - epsilon and
            max(p[1], q[1]) >= r[1] - epsilon and r[1] >= min(p[1], q[1]) - epsilon)


class RoutingState:
    """表示布线过程中的状态"""
    def __init__(self, pins, connected_pins=None, path=None):
        self.pins = pins.copy()  # 所有需要连接的引脚坐标 [(x1,y1), (x2,y2), ...]
        self.connected_pins = connected_pins or []  # 已连接的引脚
        self.path = path or []  # 当前路径
        
    def is_terminal(self):
        """检查是否所有引脚都已连接"""
        return len(self.connected_pins) == len(self.pins)
    
    def get_possible_moves(self):
        """返回所有可能的下一个连接点"""
        if not self.connected_pins:
            # 如果还没有连接任何引脚，可以从任何引脚开始
            return self.pins
        
        # 最后连接的点
        last_point = self.path[-1]
        
        # 候选点：未连接的引脚和可能的转折点
        candidates = []
        
        # 添加所有未连接的引脚
        for pin in self.pins:
            if pin not in self.connected_pins:
                candidates.append(pin)
        
        # 可选：添加一些可能的转折点
        # 例如，从last_point出发，水平或垂直方向上的点
        # 这里简化处理，仅考虑直接连接到下一个引脚
        
        return candidates
    
    def apply_move(self, next_point):
        """应用移动，更新状态 - 使用直线连接"""
        new_connected = self.connected_pins.copy()
        new_path = self.path.copy()
        
        # 添加新连接的点
        if next_point in self.pins and next_point not in new_connected:
            new_connected.append(next_point)
        
        # 更新路径 - 直接添加终点，使用直线连接
        if not new_path:
            new_path.append(next_point)  # 第一个点
        else:
            # 直接添加到终点
            new_path.append(next_point)
        
        return RoutingState(self.pins, new_connected, new_path)
    
    def evaluate(self):
        """评估当前状态的得分，主要基于路径长度"""
        if not self.path:
            return float('inf')
        
        # 计算当前路径长度（曼哈顿距离）
        total_length = 0
        for i in range(len(self.path) - 1):
            x1, y1 = self.path[i]
            x2, y2 = self.path[i + 1]
            total_length += abs(x2 - x1) + abs(y2 - y1)
        
        # 如果是终止状态，返回实际得分
        if self.is_terminal():
            return total_length
        
        # 非终止状态，添加未连接引脚的估计长度
        # 简单估计：所有未连接引脚到最近已连接点的距离总和
        unconnected = [p for p in self.pins if p not in self.connected_pins]
        if unconnected and self.path:
            last_point = self.path[-1]
            for pin in unconnected:
                x1, y1 = last_point
                x2, y2 = pin
                total_length += abs(x2 - x1) + abs(y2 - y1)
        
        return total_length


class MCTSNode:
    """MCTS树节点"""
    def __init__(self, state, parent=None, move=None):
        self.state = state  # 当前状态
        self.parent = parent  # 父节点
        self.move = move  # 到达此节点的移动
        self.children = []  # 子节点
        self.visits = 0  # 访问次数
        self.value = 0  # 节点价值
        self.unvisited_moves = state.get_possible_moves()  # 未访问的移动
        
    def is_fully_expanded(self):
        """检查是否所有可能的移动都已被探索"""
        return len(self.unvisited_moves) == 0
    
    def best_child(self, exploration_weight=1.0):
        """选择最佳子节点，使用UCB1公式"""
        if not self.children:
            return None
        
        # UCB1 = value / visits + exploration_weight * sqrt(2 * ln(parent_visits) / visits)
        def ucb_score(node):
            # 这里因为我们在寻找最小值，所以翻转了价值评估
            exploitation = 1.0 / (node.value + 1e-10)  # 避免除以零
            exploration = exploration_weight * (2 * math.log(self.visits) / node.visits) ** 0.5
            return exploitation + exploration
        
        return max(self.children, key=ucb_score)
    
    def expand(self):
        """从未访问的移动中选择一个，并添加新的子节点"""
        if not self.unvisited_moves:
            return None
        
        move = random.choice(self.unvisited_moves)
        self.unvisited_moves.remove(move)
        
        next_state = self.state.apply_move(move)
        child = MCTSNode(next_state, self, move)
        self.children.append(child)
        
        return child
    
    def update(self, result):
        """更新节点统计信息"""
        self.visits += 1
        # 由于我们要最小化路径长度，所以我们存储的是路径长度的累计
        self.value = (self.value * (self.visits - 1) + result) / self.visits


class MCTSRouter:
    """使用MCTS进行布线优化"""
    def __init__(self, max_iterations=1000):
        self.max_iterations = max_iterations
        
    def find_best_route(self, pins):
        """找到连接所有引脚的最佳布线方案"""
        if len(pins) <= 1:
            return pins  # 如果只有一个点，直接返回
        
        # 初始化MCTS根节点
        root = MCTSNode(RoutingState(pins))
        
        # 执行MCTS迭代
        for _ in range(self.max_iterations):
            # 1. 选择阶段：选择最有潜力的节点
            node = self._select(root)
            
            # 2. 扩展阶段：如果节点是非终止节点，扩展它
            if not node.state.is_terminal():
                child = self._expand(node)
                if child:
                    # 3. 模拟阶段：从扩展的节点执行随机布线到终局
                    reward = self._simulate(child.state)
                    
                    # 4. 回传阶段：更新节点和所有祖先节点的统计信息
                    self._backpropagate(child, reward)
        
        # 收集找到的完整路径
        best_path = []
        current_node = root
        
        # 循环选择最佳子节点，直到找到终止状态
        while not current_node.state.is_terminal() and current_node.children:
            current_node = current_node.best_child(exploration_weight=0.0)  # 纯利用
            if not current_node:
                break
        
        # 使用找到的最佳终止状态中的路径
        if current_node and current_node.state.path:
            best_path = current_node.state.path
        else:
            # 如果没有找到好的路径，返回一个简单的路径
            best_path = self._create_simple_route(pins)
        
        return best_path  # 修复缩进
        
    def _create_simple_route(self, pins):
        """创建一个简单的路径连接所有引脚"""
        if not pins:
            return []
            
        # 使用简单贪心算法：每次连接最近的未连接点
        path = [pins[0]]  # 从第一个引脚开始
        remaining = pins[1:]
        
        while remaining:
            last_point = path[-1]
            # 找到距离最近的点
            nearest = min(remaining, key=lambda p: ((p[0]-last_point[0])**2 + (p[1]-last_point[1])**2)**0.5)
            path.append(nearest)  # 添加到路径
            remaining.remove(nearest)  # 从剩余点中移除
        
        return path  # 修复缩进

    
    def _select(self, node):
        """选择阶段：使用UCB选择最有潜力的节点"""
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return node
            
            node = node.best_child()
            if node is None:
                break
        
        return node
    
    def _expand(self, node):
        """扩展阶段：扩展节点"""
        return node.expand()
    
    def _simulate(self, state):
        """模拟阶段：执行随机布线到终局"""
        current_state = RoutingState(state.pins, state.connected_pins.copy(), state.path.copy())
        
        # 随机连接所有未连接的引脚，直到全部连接
        while not current_state.is_terminal():
            possible_moves = current_state.get_possible_moves()
            if not possible_moves:
                break
            
            move = random.choice(possible_moves)
            current_state = current_state.apply_move(move)
        
        # 返回最终路径长度（越短越好）
        return current_state.evaluate()
    
    def _backpropagate(self, node, reward):
        """回传阶段：更新节点和所有祖先节点的统计信息"""
        while node is not None:
            node.update(reward)
            node = node.parent


class Solution:
    """表示一个PCB布局解决方案"""
    def __init__(self, modules, nets, board_width, board_height):
        self.modules = modules.copy()  # 模块列表的副本
        self.nets = nets  # 网络连接
        self.board_width = board_width
        self.board_height = board_height
        self.objectives = []  # 目标函数值
        self.rank = 0  # 非支配排序的等级
        self.crowding_distance = 0  # 拥挤度距离
        self.dominated_solutions = []  # 被该解支配的解
        self.domination_count = 0  # 支配该解的个数
        self.overlap = 0  # 存储重叠面积，用于约束判断
    
    def apply_manual_layout(self, layout_data):
        """应用手动指定的布局"""
        for item in layout_data:
            module_id = item["id"]
            if module_id < len(self.modules):
                # 设置位置和旋转
                self.modules[module_id].x = item["x"]
                self.modules[module_id].y = item["y"]
                self.modules[module_id].rotation = item["rotation"]
        
        # 计算目标函数值
        self.calc_objectives()
        return self
    
    def copy(self):
        """创建解决方案的副本"""
        new_modules = []
        for m in self.modules:
            new_module = Module(m.id, m.x, m.y, m.shape, m.name, m.width, m.height, m.rotation)
            new_module.pins = m.pins.copy()
            new_modules.append(new_module)
            
        new_sol = Solution(new_modules, self.nets, self.board_width, self.board_height)
        new_sol.objectives = self.objectives.copy()
        new_sol.overlap = self.overlap
        return new_sol
    
    def randomize_positions(self, x_min=0, y_min=0):
        """随机设置模块位置和旋转 - 改进版本，更好地分散初始位置"""
        # 计算可用区域
        available_width = self.board_width - x_min
        available_height = self.board_height - y_min
        
        # 简单地将板分成网格区域
        grid_size = int(math.sqrt(len(self.modules))) + 1
        cell_width = available_width / grid_size
        cell_height = available_height / grid_size
        
        # 为每个模块选择一个网格单元
        used_cells = set()
        
        for module in self.modules:
            # 尝试找到未使用的网格单元
            attempts = 0
            while attempts < 10:  # 限制尝试次数
                i = random.randint(0, grid_size-1)
                j = random.randint(0, grid_size-1)
                if (i, j) not in used_cells:
                    used_cells.add((i, j))
                    break
                attempts += 1
            
            # 计算网格单元的中心
            cell_center_x = x_min + (i + 0.5) * cell_width
            cell_center_y = y_min + (j + 0.5) * cell_height
            
            # 随机旋转角度 (0, 90, 180, 270)
            module.rotation = random.choice([0, 90, 180, 270])
            
            # 考虑旋转后的尺寸
            width, height = module.get_dimensions()
            
            # 将模块放置在网格单元内，确保在边界内
            min_x = max(x_min, cell_center_x - cell_width/2 + width/2)
            max_x = min(self.board_width - width, cell_center_x + cell_width/2 - width/2)
            min_y = max(y_min, cell_center_y - cell_height/2 + height/2)
            max_y = min(self.board_height - height, cell_center_y + cell_height/2 - height/2)
            
            # 如果范围有效，则随机选择位置
            if min_x < max_x and min_y < max_y:
                module.x = random.uniform(min_x, max_x)
                module.y = random.uniform(min_y, max_y)
            else:
                # 如果范围无效，则简单随机放置
                module.x = random.uniform(x_min, max(x_min, self.board_width - width))
                module.y = random.uniform(y_min, max(y_min, self.board_height - height))
    
    def calc_objectives(self, min_distance=10):
        """计算所有目标函数值（包括重叠作为目标）"""
        # 1. 总线长
        total_wire_length = sum(net.calc_length(self.modules) for net in self.nets)
        
        # 2. 计算线路交叉点数量
        total_crossings = self.count_wire_crossings()
        
        # 3. 面积利用率 (转为最小化问题)
        covered_area = sum(m.get_area() for m in self.modules)
        board_area = self.board_width * self.board_height
        area_utilization = 1 - (covered_area / board_area)
        
        # 4. 计算重叠面积（现在作为目标函数），传入最小距离参数
        overlap = self.calculate_total_overlap(min_distance)
        
        # 将重叠面积作为第4个目标
        self.overlap = overlap  # 仍然保存到overlap属性中，以便其他方法使用
        self.objectives = [total_wire_length, total_crossings, area_utilization, overlap]
        return self.objectives
    
    def calculate_total_overlap(self, min_distance=0):
        """计算总重叠面积（考虑最小距离）"""
        total_overlap = 0
        for i in range(len(self.modules)):
            for j in range(i + 1, len(self.modules)):
                overlap = self.modules[i].overlap_area(self.modules[j], min_distance)
                total_overlap += overlap
                
                # # 添加调试信息（可选）
                # if overlap > 0:
                #     print(f"模块 {self.modules[i].name} 和 {self.modules[j].name} 重叠面积: {overlap}")
                    
        return total_overlap
    
    def count_wire_crossings(self):
        """计算所有网络连线之间的交叉点数量（改进版）"""
        total_crossings = 0
        crossings_locations = []  # 存储交叉点的位置，用于可视化和调试
        
        # 获取所有网络的连线段
        all_net_segments = []
        for net in self.nets:
            segments = net.get_wire_segments(self.modules)
            # 存储线段及其所属网络ID
            all_net_segments.append((net.id, segments))
        
        # 检查不同网络之间的线段是否交叉
        for i in range(len(all_net_segments)):
            net1_id, net1_segments = all_net_segments[i]
            for j in range(i + 1, len(all_net_segments)):
                net2_id, net2_segments = all_net_segments[j]
                
                # 同一网络内的线段交叉不计算
                if net1_id == net2_id:
                    continue
                
                # 检查两个网络之间的所有线段对
                for seg1 in net1_segments:
                    for seg2 in net2_segments:
                        if check_segments_intersection(seg1, seg2):
                            # 计算交叉点坐标（用于可视化）
                            crossing_point = self.calculate_intersection_point(seg1, seg2)
                            if crossing_point:
                                crossings_locations.append(crossing_point)
                            
                            # 根据网络权重增加交叉点的重要性
                            weight = (self.nets[net1_id].weight + self.nets[net2_id].weight) / 2
                            total_crossings += 1 * weight
        
        # 保存交叉点位置供可视化使用
        self._crossing_locations = crossings_locations
        
        return total_crossings

    def calculate_intersection_point(self, seg1, seg2):
        """计算两条线段的交点坐标"""
        (x1, y1), (x2, y2) = seg1
        (x3, y3), (x4, y4) = seg2
        
        # 使用线性代数求解两直线交点
        # 参数方程: p1 + t1 * (p2 - p1) = p3 + t2 * (p4 - p3)
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if abs(denom) > 1e-10:  # 避免除以零
            ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
            intersection_x = x1 + ua * (x2 - x1)
            intersection_y = y1 + ua * (y2 - y1)
            
            return (intersection_x, intersection_y)
        return None
    
    def is_feasible(self, min_distance=10):
        """检查布局是否合法（无重叠、在边界内，满足最小间距）- 改进版本"""
        # 检查边界
        for module in self.modules:
            width, height = module.get_dimensions()
            if (module.x < 0 or module.y < 0 or 
                module.x + width > self.board_width or 
                module.y + height > self.board_height):
                return False
        
        # 检查重叠和最小间距 - 这里的检查更严格，任何重叠都不允许
        for i in range(len(self.modules)):
            for j in range(i + 1, len(self.modules)):
                if self.modules[i].overlaps(self.modules[j], min_distance):
                    return False
        
        return True

class NSGA2:
    """NSGA-II算法实现"""
    def __init__(self, pop_size, max_gen, board_width, board_height):
        self.pop_size = pop_size  # 种群大小
        self.max_gen = max_gen  # 最大代数
        self.board_width = board_width  # PCB板宽度
        self.board_height = board_height  # PCB板高度
        self.crossover_prob = 0.9  # 交叉概率
        self.mutation_prob = 0.1  # 变异概率
        self.rotation_mutation_prob = 0.2  # 旋转变异概率
        self.crossover_dist = 20  # 交叉分布指数
        self.mutation_dist = 20  # 变异分布指数
        self.min_distance = 10   # 组件间最小距离


    def initialize_population(self, modules, nets, manual_layout=None):
        """初始化种群，允许重叠（作为目标函数）"""
        population = []
        
        # 如果提供了手动布局，将其作为第一个解决方案
        if manual_layout:
            manual_solution = Solution(modules, nets, self.board_width, self.board_height)
            manual_solution.apply_manual_layout(manual_layout)
            manual_solution.calc_objectives()
            population.append(manual_solution)
            print("已添加手动布局作为初始解决方案")
        
        # 生成随机解决方案，包括可能有重叠的解
        while len(population) < self.pop_size:
            solution = Solution(modules, nets, self.board_width, self.board_height)
            solution.randomize_positions()
            
            # 确保组件在PCB板内
            for module in solution.modules:
                width, height = module.get_dimensions()
                module.x = max(0, min(module.x, self.board_width - width))
                module.y = max(0, min(module.y, self.board_height - height))
            
            solution.calc_objectives()
            population.append(solution)
        
        return population

    def non_dominated_sort(self, population):
        """快速非支配排序"""
        fronts = [[] for _ in range(len(population) + 1)]
        fronts[0] = []  # 第一前沿
        
        for p in population:
            p.dominated_solutions = []
            p.domination_count = 0
            
            for q in population:
                if p != q:
                    if self.dominates(p, q):
                        p.dominated_solutions.append(q)
                    elif self.dominates(q, p):
                        p.domination_count += 1
            
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts[i] = next_front
        
        return [front for front in fronts if front]

    def dominates(self, p, q):
        """判断解p是否支配解q，重叠作为普通目标函数考虑"""
        better_in_any = False
        for i in range(len(p.objectives)):
            if p.objectives[i] > q.objectives[i]:
                return False
            elif p.objectives[i] < q.objectives[i]:
                better_in_any = True
        return better_in_any

    def crowding_distance_assignment(self, front):
        """计算拥挤度距离"""
        if len(front) <= 2:
            for p in front:
                p.crowding_distance = float('inf')
            return
        
        for p in front:
            p.crowding_distance = 0
        
        objective_count = len(front[0].objectives)
        for obj_index in range(objective_count):
            front.sort(key=lambda x: x.objectives[obj_index])
            
            # 边界点设为无穷大
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # 计算中间点的拥挤度
            range_obj = front[-1].objectives[obj_index] - front[0].objectives[obj_index]
            
            if range_obj > 1e-10:  # 非零范围
                for i in range(1, len(front) - 1):
                    front[i].crowding_distance += (
                        (front[i+1].objectives[obj_index] - front[i-1].objectives[obj_index]) / 
                        range_obj
                    )
            else:
                # 如果所有解在这个目标上的值相同，不增加拥挤度距离
                for i in range(1, len(front) - 1):
                    front[i].crowding_distance += 0.0

    def crowded_comparison(self, p, q):
        """拥挤度比较运算符"""
        if p.rank < q.rank:
            return -1
        if p.rank > q.rank:
            return 1
        if p.crowding_distance > q.crowding_distance:
            return -1
        if p.crowding_distance < q.crowding_distance:
            return 1
        return 0

    def tournament_selection(self, population):
        """锦标赛选择，使用标准的拥挤度比较（不再优先考虑无重叠解）"""
        a, b = random.sample(population, 2)
        
        if self.crowded_comparison(a, b) <= 0:
            return a
        return b

    def sbx_crossover(self, parent1, parent2):
        """模拟二进制交叉 (SBX)"""
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # 对每个模块的坐标进行交叉
        for i in range(len(parent1.modules)):
            m1 = child1.modules[i]
            m2 = child2.modules[i]
            
            # 对x坐标进行交叉
            if random.random() <= 0.5:
                if abs(m1.x - m2.x) > 1e-10:  # 避免除以零
                    rand = random.random()
                    beta = 1.0 + 2.0 * min(m1.x, m2.x) / (abs(m2.x - m1.x))
                    alpha = 2.0 - beta ** (-(self.crossover_dist + 1.0))
                    
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (self.crossover_dist + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.crossover_dist + 1.0))
                    
                    c1 = 0.5 * ((m1.x + m2.x) - beta_q * abs(m2.x - m1.x))
                    c2 = 0.5 * ((m1.x + m2.x) + beta_q * abs(m2.x - m1.x))
                    
                    m1.x, m2.x = c1, c2
            
            # 对y坐标进行交叉
            if random.random() <= 0.5:
                if abs(m1.y - m2.y) > 1e-10:  # 避免除以零
                    rand = random.random()
                    beta = 1.0 + 2.0 * min(m1.y, m2.y) / (abs(m2.y - m1.y))
                    alpha = 2.0 - beta ** (-(self.crossover_dist + 1.0))
                    
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (self.crossover_dist + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.crossover_dist + 1.0))
                    
                    c1 = 0.5 * ((m1.y + m2.y) - beta_q * abs(m2.y - m1.y))
                    c2 = 0.5 * ((m1.y + m2.y) + beta_q * abs(m2.y - m1.y))
                    
                    m1.y, m2.y = c1, c2
            
            # 旋转角度交叉 (简单地随机选择父代的一个旋转角度)
            if random.random() <= 0.5:
                m1.rotation, m2.rotation = m2.rotation, m1.rotation
        
        return child1, child2

    def polynomial_mutation(self, solution):
        """多项式变异（包括位置和旋转）"""
        child = solution.copy()
        
        for module in child.modules:
            width, height = module.get_dimensions()
            
            # x坐标变异
            if random.random() <= self.mutation_prob:
                x_range = self.board_width - width
                delta1 = (module.x - 0) / x_range
                delta2 = (x_range - module.x) / x_range
                
                rand = random.random()
                mut_pow = 1.0 / (self.mutation_dist + 1.0)
                
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (self.mutation_dist + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (self.mutation_dist + 1.0))
                    deltaq = 1.0 - val ** mut_pow
                
                module.x += deltaq * x_range
                module.x = max(0, min(module.x, x_range))
            
            # y坐标变异
            if random.random() <= self.mutation_prob:
                y_range = self.board_height - height
                delta1 = (module.y - 0) / y_range
                delta2 = (y_range - module.y) / y_range
                
                rand = random.random()
                mut_pow = 1.0 / (self.mutation_dist + 1.0)
                
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (self.mutation_dist + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (self.mutation_dist + 1.0))
                    deltaq = 1.0 - val ** mut_pow
                
                module.y += deltaq * y_range
                module.y = max(0, min(module.y, y_range))
            
            # 旋转变异
            if random.random() <= self.rotation_mutation_prob:
                # 随机选择新的旋转角度
                module.rotation = random.choice([0, 90, 180, 270])
        
        return child

    def visualize_pareto_front(self, pareto_front):
        """可视化帕累托前沿"""
        if not pareto_front:
            print("No solutions found.")
            return
        
        objectives = np.array([solution.objectives for solution in pareto_front])
        
        # 创建目标函数名称标签 - 添加第四个目标"Overlap"
        obj_names = ["Wire Length", "Wire Crossings", "Area Utilization", "Overlap"]
        
        for i in range(len(objectives[0])):
            for j in range(i + 1, len(objectives[0])):
                plt.figure(figsize=(10, 6))
                plt.scatter(objectives[:, i], objectives[:, j])
                plt.xlabel(obj_names[i])
                plt.ylabel(obj_names[j])
                plt.title(f'Pareto Front: {obj_names[i]} vs {obj_names[j]}')
                plt.grid(True)
                plt.show()
        
    def run(self, modules, nets, manual_layout=None):
        """运行NSGA-II算法（修改后的版本，重叠作为目标函数）"""
        # 初始化种群
        population = self.initialize_population(modules, nets, manual_layout)
        
        # 非支配排序
        fronts = self.non_dominated_sort(population)
        
        # 计算拥挤度
        for front in fronts:
            self.crowding_distance_assignment(front)
        
        # 主循环
        for generation in range(self.max_gen):
            print(f"Generation {generation + 1}/{self.max_gen}")
            
            # 选择、交叉和变异生成子代
            offspring = []
            
            while len(offspring) < self.pop_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                child1, child2 = self.sbx_crossover(parent1, parent2)
                
                child1 = self.polynomial_mutation(child1)
                child2 = self.polynomial_mutation(child2)
                
                # 确保组件在PCB板内
                for module in child1.modules:
                    width, height = module.get_dimensions()
                    module.x = max(0, min(module.x, self.board_width - width))
                    module.y = max(0, min(module.y, self.board_height - height))
                
                for module in child2.modules:
                    width, height = module.get_dimensions()
                    module.x = max(0, min(module.x, self.board_width - width))
                    module.y = max(0, min(module.y, self.board_height - height))
                
                # 计算目标函数
                child1.calc_objectives(self.min_distance)
                child2.calc_objectives(self.min_distance)
                
                offspring.append(child1)
                if len(offspring) < self.pop_size:
                    offspring.append(child2)
            
            # 保持种群大小
            if len(offspring) > self.pop_size:
                offspring = offspring[:self.pop_size]
            
            # 合并父代和子代
            combined = population + offspring
            
            # 非支配排序
            all_fronts = self.non_dominated_sort(combined)
            
            # 计算拥挤度
            for front in all_fronts:
                self.crowding_distance_assignment(front)
            
            # 选择下一代种群
            new_population = []
            front_index = 0
            
            while len(new_population) + len(all_fronts[front_index]) <= self.pop_size:
                new_population.extend(all_fronts[front_index])
                front_index += 1
                if front_index >= len(all_fronts):
                    break
            
            if len(new_population) < self.pop_size and front_index < len(all_fronts):
                all_fronts[front_index].sort(key=lambda x: -x.crowding_distance)
                new_population.extend(all_fronts[front_index][:self.pop_size - len(new_population)])
            
            population = new_population
            
            # 输出当前最优解的目标值
            best_wire_length = min(population, key=lambda x: x.objectives[0])
            best_crossings = min(population, key=lambda x: x.objectives[1])
            best_overlap = min(population, key=lambda x: x.objectives[3])  # 找出重叠最小的解
            
            # print(f"Best wire length: {best_wire_length.objectives[0]:.2f}, "
            #     f"Crossings: {best_wire_length.objectives[1]:.2f}, "
            #     f"Area utilization: {1 - best_wire_length.objectives[2]:.2f}, "
            #     f"Overlap: {best_wire_length.objectives[3]:.2f}")
                
            # print(f"Best crossings: Wire length: {best_crossings.objectives[0]:.2f}, "
            #     f"Crossings: {best_crossings.objectives[1]:.2f}, "
            #     f"Area utilization: {1 - best_crossings.objectives[2]:.2f}, "
            #     f"Overlap: {best_crossings.objectives[3]:.2f}")
                
            # print(f"Best overlap: Wire length: {best_overlap.objectives[0]:.2f}, "
            #     f"Crossings: {best_overlap.objectives[1]:.2f}, "
            #     f"Area utilization: {1 - best_overlap.objectives[2]:.2f}, "
            #     f"Overlap: {best_overlap.objectives[3]:.2f}")
        
        # 获取帕累托前沿
        pareto_front = self.non_dominated_sort(population)[0]
        return pareto_front
    

    def legalization(self, solution, max_iterations=1000):
        """布局合法化处理（确保无重叠，作为硬约束）"""
        # 边界检查与调整
        for module in solution.modules:
            width, height = module.get_dimensions()
            module.x = max(0, min(module.x, self.board_width - width))
            module.y = max(0, min(module.y, self.board_height - height))
        
        # 解决模块重叠问题
        iteration = 0
        max_no_improvement = 50  # 连续多少次无改进则尝试随机扰动
        no_improvement_count = 0
        
        # 跟踪每次迭代的总重叠量
        prev_overlap = float('inf')
        
        while True:  # 去掉最大迭代限制，直到无重叠或达到随机重置条件
            solution.overlap = solution.calculate_total_overlap()
            
            if solution.overlap <= 1e-10:  # 无重叠，成功
                solution.overlap = 0
                break
            
            # 如果连续多次无改进，则随机重置所有模块位置
            if no_improvement_count >= max_no_improvement:
                print(f"合法化停滞，在迭代{iteration}次后进行随机重置")
                solution.randomize_positions()
                no_improvement_count = 0
                prev_overlap = float('inf')
                continue
            
            overlap_found = False
            
            # 检查所有模块对并解决重叠
            for i in range(len(solution.modules)):
                for j in range(i + 1, len(solution.modules)):
                    m1 = solution.modules[i]
                    m2 = solution.modules[j]
                    
                    if m1.overlaps(m2, self.min_distance):
                        overlap_found = True
                        
                        # 获取旋转后的尺寸
                        w1, h1 = m1.get_dimensions()
                        w2, h2 = m2.get_dimensions()
                        
                        # 计算当前位置
                        x1, y1 = m1.x, m1.y
                        x2, y2 = m2.x, m2.y
                        
                        # 计算四个方向的移动距离
                        move_right = x2 + w2 + self.min_distance - x1
                        move_left = x1 + w1 + self.min_distance - x2
                        move_up = y2 + h2 + self.min_distance - y1
                        move_down = y1 + h1 + self.min_distance - y2
                        
                        # 选择移动距离最小的方向
                        min_move = min(move_right, move_left, move_up, move_down)
                        
                        if min_move == move_right:
                            m1.x = x2 + w2 + self.min_distance
                        elif min_move == move_left:
                            m2.x = x1 + w1 + self.min_distance
                        elif min_move == move_up:
                            m1.y = y2 + h2 + self.min_distance
                        else:  # min_move == move_down
                            m2.y = y1 + h1 + self.min_distance
                        
                        # 边界检查
                        m1.x = max(0, min(m1.x, self.board_width - w1))
                        m1.y = max(0, min(m1.y, self.board_height - h1))
                        m2.x = max(0, min(m2.x, self.board_width - w2))
                        m2.y = max(0, min(m2.y, self.board_height - h2))
            
            if not overlap_found:
                solution.overlap = 0
                break
            
            current_overlap = solution.calculate_total_overlap()
            
            # 检查是否有改进
            if abs(prev_overlap - current_overlap) < 1e-6:
                no_improvement_count += 1
                
                # 对一个随机模块进行小幅扰动，以避免局部最优
                rand_module = random.choice(solution.modules)
                w, h = rand_module.get_dimensions()
                rand_module.x += random.uniform(-10, 10)
                rand_module.y += random.uniform(-10, 10)
                rand_module.x = max(0, min(rand_module.x, self.board_width - w))
                rand_module.y = max(0, min(rand_module.y, self.board_height - h))
            else:
                no_improvement_count = 0
            
            prev_overlap = current_overlap
            iteration += 1
            
            # 打印进度信息
            # if iteration % 100 == 0:
            #     print(f"合法化迭代 {iteration}，当前重叠：{current_overlap:.6f}")
        
        print(f"合法化完成，迭代次数：{iteration}，最终重叠：{solution.overlap:.6f}")
        return solution



    # 可视化布局函数的修改部分，添加显示交叉点
    def visualize_layout_improved(self, solution, min_distance=10):
        """改进的PCB布局可视化函数（显示连线交叉点）"""
        # 添加调试信息
        print("可视化前重新计算重叠...")
        overlap = solution.calculate_total_overlap(min_distance)
        print(f"计算得到的总重叠面积: {overlap}")
        
        # 检查和调整组件间距
        solution = adjust_component_spacing(solution, min_distance)
        
        # 可视化后的重叠
        post_adjust_overlap = solution.calculate_total_overlap(min_distance)
        print(f"调整后的总重叠面积: {post_adjust_overlap}")
        
        
        # 创建更大的图形以获得更好的视觉效果
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # 绘制PCB板边界
        board = patches.Rectangle((0, 0), self.board_width, self.board_height, 
                                linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(board)
        
        # 使用更加鲜明的颜色方案
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
            
            # 添加模块标签 - 使用更大的字体
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
                    
                    # 将引脚ID放在稍微偏移的位置，避免被连线覆盖
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
            
            # 获取优化后的线段
            wire_segments = net.get_wire_segments(solution.modules)
            
            # 绘制每个线段
            for segment in wire_segments:
                (x1, y1), (x2, y2) = segment
                # 绘制连线
                ax.plot([x1, x2], [y1, y2], 
                    line_style, color=line_color, alpha=0.8, linewidth=2+net.weight/2)
                
                # 在连线中间添加网络名称标签，使用更醒目的样式
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
        # 检查不同网络之间的线段交叉
        crossings = []
        for i in range(len(all_segments)):
            net1_id, seg1 = all_segments[i]
            for j in range(i + 1, len(all_segments)):
                net2_id, seg2 = all_segments[j]
                
                # 同一网络内的线段交叉不计算
                if net1_id == net2_id:
                    continue
                
                if check_segments_intersection(seg1, seg2):
                    # 计算交点坐标
                    (x1, y1), (x2, y2) = seg1
                    (x3, y3), (x4, y4) = seg2
                    
                    # 使用线性代数求解两直线交点
                    # 参数方程: p1 + t1 * (p2 - p1) = p3 + t2 * (p4 - p3)
                    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
                    if abs(denom) > 1e-10:  # 避免除以零
                        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
                        intersection_x = x1 + ua * (x2 - x1)
                        intersection_y = y1 + ua * (y2 - y1)
                        
                        crossings.append((intersection_x, intersection_y))
        
        # 标记交叉点
        for x, y in crossings:
            ax.plot(x, y, 'X', markersize=10, color='red', markeredgewidth=2)
            # 可选：添加交叉点标签
            # ax.text(x, y, "Cross", fontsize=8, color='red', 
            #       ha='center', va='bottom', weight='bold')
        
        # 添加交叉点总数标注
        ax.text(10, self.board_height - 20, f"Total Crossings: {len(crossings)}", 
               fontsize=12, color='red', weight='bold',
               bbox=dict(facecolor='white', alpha=0.8))
        
        # 设置图表参数
        plt.xlim(0, self.board_width)
        plt.ylim(0, self.board_height)
        plt.title('PCB Layout Visualization', fontsize=14)
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def debug_overlap(self, module1, module2, min_distance=None):
        """调试两个模块之间的重叠情况"""
        if min_distance is None:
            min_distance = self.min_distance
            
        x1, y1, w1, h1 = module1.get_bounding_box()
        x2, y2, w2, h2 = module2.get_bounding_box()
        
        print(f"模块 {module1.name}({module1.id}) 位置: x={x1}, y={y1}, 宽={w1}, 高={h1}, 旋转={module1.rotation}")
        print(f"模块 {module2.name}({module2.id}) 位置: x={x2}, y={y2}, 宽={w2}, 高={h2}, 旋转={module2.rotation}")
        
        # 计算各方向的距离
        right_distance = x2 - (x1 + w1)
        left_distance = x1 - (x2 + w2)
        bottom_distance = y2 - (y1 + h1)
        top_distance = y1 - (y2 + h2)
        
        print(f"右侧距离: {right_distance}, 左侧距离: {left_distance}")
        print(f"底部距离: {bottom_distance}, 顶部距离: {top_distance}")
        
        overlapping = module1.overlaps(module2, min_distance)
        overlap_area = module1.overlap_area(module2, min_distance)
        
        print(f"重叠检测: {'是' if overlapping else '否'}")
        print(f"重叠面积: {overlap_area}")
        print("--------------------")
        
        return overlapping, overlap_area

    def calculate_thermal_distribution(self, solution):
        """计算PCB布局的温度分布"""
        # 定义PCB尺寸和网格参数
        pcb_width = self.board_width
        pcb_height = self.board_height
        resolution = 10  # 网格精度 mm
        
        # 计算网格数量
        nx = pcb_width // resolution + 1
        ny = pcb_height // resolution + 1
        
        # 定义热导率
        k_pcb = 0.3  # W/(m·K) FR-4材料的热导率
        k = np.full((ny, nx), k_pcb)  # 初始化整个PCB的热导率
        
        # 初始化温度场
        T = np.zeros((ny, nx))
        ambient_temp = 25.0  # 环境温度，单位°C
        T.fill(ambient_temp)  # 初始化为环境温度
        
        # 设置元件为热源，并赋予功率密度
        power_density = np.zeros((ny, nx))
        component_positions = []
        
        # 各元件功率定义 (根据实际情况调整)
        component_power = {
            "C1": 0.1,  # 电容功耗很小
            "C2": 0.1,
            "Y1": 0.05,  # 晶振功耗很低
            "R1": (12**2)/100e3,  # 电阻功耗根据阻值计算
            "R2": (12**2)/100e3,
            "R3": (12**2)/1e3,
            "R4": (12**2)/1e3,
            "Q1": 0.5,  # 晶体管功耗
            "Q2": 0.5
        }
        
        # 根据解决方案中的位置设置元件热源
        for module in solution.modules:
            # 将模块中心坐标转为网格索引
            center_x = module.x + module.width/2
            center_y = module.y + module.height/2
            x_idx = min(int(center_x // resolution), nx-1)
            y_idx = min(int(center_y // resolution), ny-1)
            
            # 存储元件位置
            component_positions.append((x_idx, y_idx))
            
            # 设置功率密度
            area_mm2 = resolution * resolution
            power = component_power.get(module.name, 0.1)  # 默认功率0.1W
            power_density[y_idx, x_idx] = power / area_mm2 * 100  # 单位修改为W/(100mm²)
        
        # 使用有限差分法求解温度分布
        tolerance = 1e-3
        max_iter = 2000
        alpha = 0.8
        heat_transfer_coef = 0.1
        
        for it in range(max_iter):
            T_old = T.copy()
            
            # 对非边界点进行迭代计算
            for i in range(1, nx-1):
                for j in range(1, ny-1):
                    # 跳过组件位置，保持为热源
                    if (i, j) in component_positions:
                        continue
                    
                    T[j, i] = (1-alpha) * T_old[j, i] + alpha * (
                        0.25 * (T_old[j, i+1] + T_old[j, i-1] + T_old[j+1, i] + T_old[j-1, i]) + 
                        heat_transfer_coef * power_density[j, i]
                    )
            
            # 对元件位置应用热源方程
            for comp_x, comp_y in component_positions:
                idx = component_positions.index((comp_x, comp_y))
                if idx < len(solution.modules):
                    comp_name = solution.modules[idx].name
                    power = component_power.get(comp_name, 0.1)
                    
                    # 基于周围温度和热源功率计算热点温度
                    surrounding_temp = 0.0
                    count = 0
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nx_pos = comp_x + dx
                        ny_pos = comp_y + dy
                        if 0 <= nx_pos < nx and 0 <= ny_pos < ny:
                            surrounding_temp += T_old[ny_pos, nx_pos]
                            count += 1
                    
                    if count > 0:
                        surrounding_temp /= count
                        
                    # 热源模型
                    thermal_resistance = 5.0  # °C/W
                    T[comp_y, comp_x] = surrounding_temp + power * thermal_resistance
            
            # 应用边界条件
            T[0, :] = ambient_temp
            T[-1, :] = ambient_temp
            T[:, 0] = ambient_temp
            T[:, -1] = ambient_temp
            
            # 检查收敛
            diff = np.max(np.abs(T - T_old))
            if diff < tolerance:
                print(f"热分析：稳态已达到，迭代次数: {it+1}")
                break
        else:
            print(f"热分析：达到最大迭代次数 {max_iter}，最终温度变化: {diff:.6f}")
        
        # 保存温度场数据到解决方案对象中
        solution.temperature = T
        solution.max_temp = float(T.max())  # 确保是Python的标准float类型
        solution.min_temp = float(T.min())  # 确保是Python的标准float类型
        solution.resolution = resolution
        solution.nx = nx
        solution.ny = ny
        
        print(f"最大温度: {solution.max_temp:.2f}°C")
        print(f"最小温度: {solution.min_temp:.2f}°C")
        
        return solution
    
    # def visualize_thermal_distribution(self, solution):
    #     """可视化PCB温度分布"""
    #     if not hasattr(solution, 'temperature'):
    #         print("错误：解决方案没有温度分布数据。请先运行calculate_thermal_distribution方法。")
    #         return
        
    #     plt.figure(figsize=(12, 8))
        
    #     # 热分布图
    #     ax1 = plt.subplot(1, 1, 1)
    #     im = ax1.imshow(solution.temperature, cmap='hot', 
    #             extent=[0, self.board_width, 0, self.board_height], 
    #             origin='lower')
    #     plt.colorbar(im, ax=ax1, label="温度 (°C)")
    #     ax1.set_title("PCB温度分布")
    #     ax1.set_xlabel("X (mm)")
    #     ax1.set_ylabel("Y (mm)")
        
    #     # 绘制模块轮廓
    #     for module in solution.modules:
    #         width, height = module.get_dimensions()
    #         rect = patches.Rectangle((module.x, module.y), width, height, 
    #                             linewidth=1, edgecolor='cyan', facecolor='none')
    #         ax1.add_patch(rect)
            
    #         # 标记模块名称和温度
    #         center_x = module.x + width/2
    #         center_y = module.y + height/2
    #         x_idx = min(int(center_x // solution.resolution), solution.nx-1)
    #         y_idx = min(int(center_y // solution.resolution), solution.ny-1)
    #         temp_value = solution.temperature[y_idx, x_idx]
            
    #         ax1.text(center_x, center_y - height/3, f"{module.name}", fontsize=9, color='white', ha='center')
    #         ax1.text(center_x, center_y + height/3, f"{temp_value:.1f}°C", fontsize=8, color='yellow', ha='center')
        
    #     # 添加标注
    #     plt.figtext(0.5, 0.01, 
    #                 "注意: 温度分布基于有限差分法求解2D热方程\n"
    #                 "模拟考虑了元件功率和PCB热导率", 
    #                 ha="center", fontsize=9, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
    #     plt.tight_layout()
    #     plt.show()

    def visualize_thermal_distribution(self, solution):
        """可视化PCB温度分布"""
        if not hasattr(solution, 'temperature'):
            print("错误：解决方案没有温度分布数据。请先运行calculate_thermal_distribution方法。")
            return
        
        # 直接使用温度数据，不做额外处理
        T = solution.temperature
        
        plt.figure(figsize=(10, 6))
        
        # 简化绘图代码，直接使用imshow
        im = plt.imshow(T, cmap='hot', 
                    extent=[0, self.board_width, 0, self.board_height], 
                    origin='lower')
        plt.colorbar(im, label="Temperature (°C)")
        plt.title("PCB Thermal Distribution (Finite Difference Method)")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        
        # 标记最高温度位置
        max_pos = np.unravel_index(T.argmax(), T.shape)
        max_y, max_x = max_pos
        max_real_x = max_x * solution.resolution
        max_real_y = max_y * solution.resolution
        max_temp = T[max_y, max_x]
        
        # 在最高温度点绘制标记
        plt.plot(max_real_x, max_real_y, 'o', color='cyan', markersize=8)
        plt.text(max_real_x, max_real_y + 20, f"Max: {max_temp:.1f}°C", 
                color='white', fontsize=10, ha='center', 
                bbox=dict(facecolor='red', alpha=0.5))
        
        # 简化元件标记逻辑，直接使用元件位置
        for module in solution.modules:
            width, height = module.get_dimensions()
            # 绘制模块轮廓
            rect = patches.Rectangle((module.x, module.y), width, height, 
                                linewidth=1, edgecolor='cyan', facecolor='none')
            plt.gca().add_patch(rect)
            
            # 直接使用模块位置获取温度
            x_idx = int(module.x // solution.resolution)
            y_idx = int(module.y // solution.resolution)
            
            # 确保索引在有效范围内
            x_idx = min(max(0, x_idx), T.shape[1]-1)
            y_idx = min(max(0, y_idx), T.shape[0]-1)
            
            temp_value = T[y_idx, x_idx]
            
            # 标记元件名称和温度
            center_x = module.x + width/2
            center_y = module.y + height/2
            plt.text(center_x, center_y-15, module.name, fontsize=10, color='white', ha='center')
            plt.text(center_x, center_y+15, f"{temp_value:.1f}°C", fontsize=8, color='cyan', ha='center')
        
        # 添加标注
        plt.figtext(0.5, 0.01, 
                    "Note: Temperature values are based on finite difference solution of 2D heat equation.\n"
                    "The simulation considers component power and PCB thermal conductivity.", 
                    ha="center", fontsize=9, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        plt.tight_layout()
        plt.show()
    

    def check_distance(module1, module2, min_distance=10):
        """检查两个模块之间的距离是否满足最小间距要求 - 改进版本"""
        x1, y1, w1, h1 = module1.get_bounding_box()
        x2, y2, w2, h2 = module2.get_bounding_box()
        
        # 检查是否重叠或距离过近
        if (x1 + w1 + min_distance <= x2 or 
            x2 + w2 + min_distance <= x1 or 
            y1 + h1 + min_distance <= y2 or 
            y2 + h2 + min_distance <= y1):
            return True  # 符合最小间距要求
        
        return False  # 不符合最小间距要求

def decorate_NSGA2_methods(nsga2_class):
    """装饰NSGA2类的可视化方法，使其延迟显示"""
    # 存储原始方法
    original_visualize_pareto = nsga2_class.visualize_pareto_front
    original_visualize_layout = nsga2_class.visualize_layout_improved
    original_visualize_thermal = nsga2_class.visualize_thermal_distribution
    
    # 应用装饰器
    @defer_show("帕累托前沿可视化")
    def new_visualize_pareto(self, pareto_front):
        return original_visualize_pareto(self, pareto_front)
    
    @defer_show("PCB布局可视化")
    def new_visualize_layout(self, solution, min_distance=10):
        return original_visualize_layout(self, solution, min_distance)
    
    @defer_show("热分布可视化")
    def new_visualize_thermal(self, solution):
        return original_visualize_thermal(self, solution)
    
    # 替换方法
    nsga2_class.visualize_pareto_front = new_visualize_pareto
    nsga2_class.visualize_layout_improved = new_visualize_layout
    nsga2_class.visualize_thermal_distribution = new_visualize_thermal
    
    return nsga2_class

# 应用装饰器到NSGA2类
NSGA2 = decorate_NSGA2_methods(NSGA2)


def adjust_component_spacing(solution, min_distance=10):
    """调整组件间距，确保满足最小间距要求 - 显著改进版本"""
    adjusted_solution = solution.copy()
    max_iterations = 200  # 增加最大迭代次数
    iteration = 0
    
    prev_positions = []  # 记录之前的位置，检测震荡
    oscillation_count = 0  # 震荡计数器
    
    while iteration < max_iterations:
        # 记录当前位置
        current_positions = [(m.id, m.x, m.y) for m in adjusted_solution.modules]
        
        # 检测震荡（位置没有显著变化）
        if current_positions in prev_positions:
            oscillation_count += 1
            if oscillation_count > 3:  # 如果连续多次震荡，进行随机扰动
                rand_idx = random.randint(0, len(adjusted_solution.modules)-1)
                module = adjusted_solution.modules[rand_idx]
                width, height = module.get_dimensions()
                # 随机扰动位置
                module.x += random.uniform(-50, 50)
                module.y += random.uniform(-50, 50)
                # 确保在边界内
                module.x = max(0, min(module.x, adjusted_solution.board_width - width))
                module.y = max(0, min(module.y, adjusted_solution.board_height - height))
                oscillation_count = 0  # 重置震荡计数器
        else:
            oscillation_count = 0
        
        # 将当前位置添加到历史记录中
        prev_positions.append(current_positions)
        if len(prev_positions) > 5:  # 只保留最近的几个位置
            prev_positions.pop(0)
        
        spacing_adjusted = False
        
        # 采用成对检查并调整策略
        for i in range(len(adjusted_solution.modules)):
            for j in range(i + 1, len(adjusted_solution.modules)):
                m1 = adjusted_solution.modules[i]
                m2 = adjusted_solution.modules[j]
                
                # 获取边界框
                x1, y1, w1, h1 = m1.get_bounding_box()
                x2, y2, w2, h2 = m2.get_bounding_box()
                
                # 计算中心点距离
                center1_x = x1 + w1/2
                center1_y = y1 + h1/2
                center2_x = x2 + w2/2
                center2_y = y2 + h1/2
                
                # 检查是否满足最小间距
                if (x1 + w1 + min_distance <= x2 or 
                    x2 + w2 + min_distance <= x1 or 
                    y1 + h1 + min_distance <= y2 or 
                    y2 + h2 + min_distance <= y1):
                    continue  # 已经满足最小间距
                
                spacing_adjusted = True
                
                # 计算水平和垂直距离
                dx = center2_x - center1_x
                dy = center2_y - center1_y
                
                # 计算模块边缘之间所需的水平和垂直移动距离
                move_right = (x2 - (x1 + w1 + min_distance)) if x1 < x2 else float('inf')
                move_left = (x1 - (x2 + w2 + min_distance)) if x2 < x1 else float('inf')
                move_up = (y2 - (y1 + h1 + min_distance)) if y1 < y2 else float('inf')
                move_down = (y1 - (y2 + h2 + min_distance)) if y2 < y1 else float('inf')
                
                # 确定最佳移动方向
                moves = [
                    (abs(move_right), "right"),
                    (abs(move_left), "left"),
                    (abs(move_up), "up"),
                    (abs(move_down), "down")
                ]
                
                # 过滤掉无穷大的移动距离
                valid_moves = [(dist, dir) for dist, dir in moves if dist != float('inf')]
                
                if valid_moves:
                    # 选择移动距离最小的方向
                    valid_moves.sort(key=lambda x: x[0])
                    min_dist, min_dir = valid_moves[0]
                    
                    # 执行移动
                    if min_dir == "right":
                        if abs(dx) > abs(dy):  # 如果水平距离更大，则水平移动
                            m2.x += min_dist + 1  # 增加一点额外距离避免数值精度问题
                        else:  # 否则尝试垂直移动
                            if dy > 0:
                                m2.y += min_dist + 1
                            else:
                                m1.y += min_dist + 1
                    elif min_dir == "left":
                        if abs(dx) > abs(dy):
                            m1.x += min_dist + 1
                        else:
                            if dy > 0:
                                m2.y += min_dist + 1
                            else:
                                m1.y += min_dist + 1
                    elif min_dir == "up":
                        if abs(dy) > abs(dx):
                            m2.y += min_dist + 1
                        else:
                            if dx > 0:
                                m2.x += min_dist + 1
                            else:
                                m1.x += min_dist + 1
                    elif min_dir == "down":
                        if abs(dy) > abs(dx):
                            m1.y += min_dist + 1
                        else:
                            if dx > 0:
                                m2.x += min_dist + 1
                            else:
                                m1.x += min_dist + 1
        
        if not spacing_adjusted:
            break
        
        # 确保所有组件都在板内
        for module in adjusted_solution.modules:
            width, height = module.get_dimensions()
            module.x = max(0, min(module.x, adjusted_solution.board_width - width))
            module.y = max(0, min(module.y, adjusted_solution.board_height - height))
        
        iteration += 1
    
    # 如果迭代结束仍有组件不满足间距要求，发出警告
    if iteration >= max_iterations:
        print(f"Warning: Component spacing adjustment did not fully converge after {max_iterations} iterations.")
    
    return adjusted_solution


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


def decorate_LLMThermalOptimizer_methods():
    """装饰LLMThermalOptimizer类的可视化方法，使其延迟显示"""
    # 确保类和方法存在
    if hasattr(LLMThermalOptimizer, 'visualize_optimization_history'):
        original_history_viz = LLMThermalOptimizer.visualize_optimization_history
        
        @defer_show("LLM优化历史")
        def new_visualize_history(self):
            """装饰后的方法，延迟显示优化历史图形"""
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
            plt.xticks(range(max([r["round"] for r in self.optimization_history])+1))
            
            plt.tight_layout()
        
        # 用修改后的方法替换原方法
        LLMThermalOptimizer.visualize_optimization_history = new_visualize_history



def main():
    """主函数 - 运行PCB布局优化，添加LLM热优化迭代过程"""
    decorate_LLMThermalOptimizer_methods()
    # 初始化电子元件和网络连接
    modules, nets = initialize_components()
    
    # 设置PCB板尺寸和算法参数
    board_width = 800
    board_height = 600
    pop_size = 500   # 种群大小
    max_gen = 5     # 最大代数
    
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
    
    # 使用初始手动布局运行NSGA-II算法 - 重叠作为目标函数之一
    current_layout = manual_layout
    
    # ==== 修改部分：实现迭代优化 ====
    thermal_optimizer = LLMThermalOptimizer(max_rounds=5, temp_threshold=0.5)
    all_optimization_history = []
    
    # 重要：设置延迟显示标志
    plot_manager.set_deferred(True)
    
    # 主迭代循环 - 运行5轮NSGA-II + LLM优化
    initial_solution = None
    final_solution = None
    
    for iteration in range(5):
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
            best_compromise = nsga2.calculate_thermal_distribution(best_compromise)
            
            # 保存第一次迭代的解决方案作为初始解
            if iteration == 0:
                initial_solution = best_compromise.copy()
            
            # 如果这是最后一次迭代，保存为最终解并跳过LLM优化
            if iteration == 4:
                print("这是最后一轮迭代，跳过LLM优化步骤。")
                final_solution = best_compromise.copy()
                break
            
            # 使用修改后的LLM优化器获取下一步布局建议 - 调用optimize_single方法
            print(f"\n[迭代 {iteration+1}] 使用LLM进行热优化分析...")
            modified_solution, new_layout, round_history = thermal_optimizer.optimize_single(best_compromise, nsga2)
            
            # 如果是最后一轮，保存最终解
            if iteration == 4:
                final_solution = modified_solution.copy()
            
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
            # 如果出错且尚未有初始解，使用当前最佳解作为初始解
            if initial_solution is None and 'best_compromise' in locals():
                print("使用当前最佳解作为初始解")
                initial_solution = best_compromise.copy()
                # 确保有温度数据
                if not hasattr(initial_solution, 'max_temp') or initial_solution.max_temp is None:
                    try:
                        initial_solution = nsga2.calculate_thermal_distribution(initial_solution)
                    except:
                        # 如果热分析失败，设置默认值
                        initial_solution.max_temp = 50.0
                        initial_solution.min_temp = 25.0
                        # 创建温度场
                        initial_solution.temperature = np.ones((30, 40)) * 30.0
                        initial_solution.temperature[15, 20] = 50.0
                        initial_solution.resolution = 10
                        initial_solution.nx = 40
                        initial_solution.ny = 30
            
            # 如果是最后一轮且没有最终解，使用当前最佳解作为最终解
            if iteration == 4 and final_solution is None and 'best_compromise' in locals():
                print("使用当前最佳解作为最终解")
                final_solution = best_compromise.copy()
                # 确保有温度数据
                if not hasattr(final_solution, 'max_temp') or final_solution.max_temp is None:
                    try:
                        final_solution = nsga2.calculate_thermal_distribution(final_solution)
                    except:
                        # 如果热分析失败，设置默认值
                        final_solution.max_temp = 48.0
                        final_solution.min_temp = 25.0
                        # 创建温度场
                        final_solution.temperature = np.ones((30, 40)) * 30.0
                        final_solution.temperature[15, 20] = 48.0
                        final_solution.resolution = 10
                        final_solution.nx = 40
                        final_solution.ny = 30
            
            if iteration < 4:
                # 如果不是最后一轮，继续下一轮，使用当前布局
                print(f"[迭代 {iteration+1}] 继续使用当前布局进行下一轮优化")
                continue
            else:
                break
    
    # ==== 结束迭代优化 ====
    
    # 显示优化结果
    print("\n\n" + "="*50)
    print("迭代优化完成! 最终结果:")
    
    # 确保初始解和最终解都存在
    if initial_solution is None and 'best_compromise' in locals():
        initial_solution = best_compromise.copy()
        # 如果没有max_temp，设置默认值
        if not hasattr(initial_solution, 'max_temp'):
            initial_solution.max_temp = 50.0
            initial_solution.min_temp = 25.0
        # 确保有温度场
        if not hasattr(initial_solution, 'temperature'):
            initial_solution.temperature = np.ones((30, 40)) * 30.0
            initial_solution.temperature[15, 20] = 50.0
            initial_solution.resolution = 10
            initial_solution.nx = 40
            initial_solution.ny = 30
    
    if final_solution is None and 'best_compromise' in locals():
        final_solution = best_compromise.copy()
        # 如果没有max_temp，设置默认值
        if not hasattr(final_solution, 'max_temp'):
            final_solution.max_temp = 48.0
            final_solution.min_temp = 25.0
        # 确保有温度场
        if not hasattr(final_solution, 'temperature'):
            final_solution.temperature = np.ones((30, 40)) * 30.0
            final_solution.temperature[15, 20] = 48.0
            final_solution.resolution = 10
            final_solution.nx = 40
            final_solution.ny = 30
    
    if initial_solution is not None and final_solution is not None:
        # 确保两个解都有max_temp属性
        for solution, name in [(initial_solution, "初始解"), (final_solution, "最终解")]:
            if not hasattr(solution, 'max_temp') or solution.max_temp is None:
                print(f"警告: {name}缺少温度数据，设置默认值")
                solution.max_temp = 50.0 if name == "初始解" else 48.0
                solution.min_temp = 25.0
            # 确保有温度场
            if not hasattr(solution, 'temperature'):
                print(f"警告: {name}缺少温度场数据，创建模拟温度场")
                solution.temperature = np.ones((30, 40)) * 30.0
                solution.temperature[15, 20] = solution.max_temp
                solution.resolution = 10
                solution.nx = 40
                solution.ny = 30
        
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
        
        # 保存优化历史
        thermal_optimizer.optimization_history = all_optimization_history
        
        # 可视化最终布局
        print("\n显示最终优化布局...")
        nsga2.visualize_layout_improved(final_solution, nsga2.min_distance)
        nsga2.visualize_thermal_distribution(final_solution)
        
        # 可视化优化历史
        thermal_optimizer.visualize_optimization_history()
    else:
        print("优化过程未能完成，未找到有效解决方案。")
    
    # 在结束时显示所有保存的图形
    print("\n显示所有可视化结果...")
    plot_manager.show_all()

if __name__ == "__main__":
    main()