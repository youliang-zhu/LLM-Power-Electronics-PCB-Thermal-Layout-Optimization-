import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
from functools import wraps

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