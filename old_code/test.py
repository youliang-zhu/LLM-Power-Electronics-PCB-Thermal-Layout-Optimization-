import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import math
import numpy as np

class Module:
    def __init__(self, id, x=0, y=0, shape='rectangle', name="", width=0, height=0, rotation=0, power=0.0):
        self.id = id
        self.name = name if name else f"Module_{id}"
        self.width = width
        self.height = height
        self.shape = shape
        self.x = x
        self.y = y
        self.rotation = rotation  # 0, 90, 180, 270 degrees
        self.pins = []  # List of (x_offset, y_offset, pin_id) tuples
        self.power = power  # 组件功耗，单位为瓦特
        
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
        
        return True  # 重叠或距离不足
    
    def overlap_area(self, other):
        """重叠面积计算（考虑旋转）"""
        if not self.overlaps(other):
            return 0
        
        x1, y1, w1, h1 = self.get_bounding_box()
        x2, y2, w2, h2 = other.get_bounding_box()
        
        # 计算重叠区域
        x_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
        y_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)
        
        return x_overlap * y_overlap

class Net:
    """网络连接类，代表模块间的连线"""
    def __init__(self, id, connections, name="", weight=1.0):
        self.id = id
        self.name = name if name else f"Net_{id}"
        self.connections = connections  # List of (module_id, pin_id) tuples
        self.weight = weight  # 网络权重，影响线长计算
    
    def calc_length(self, modules):
        """计算该网络的总线长（欧几里得距离）- 改进版本"""
        total_length = 0
        connections = []
        
        # 获取所有引脚的位置
        for m_id, p_id in self.connections:
            if m_id < len(modules):
                pin_pos = modules[m_id].get_pin_position(p_id)
                if pin_pos:
                    connections.append((m_id, p_id, pin_pos))
        
        # 计算所有可能的引脚对之间的距离
        for i in range(len(connections) - 1):
            for j in range(i + 1, len(connections)):
                m1_id, p1_id, (x1, y1) = connections[i]
                m2_id, p2_id, (x2, y2) = connections[j]
                
                # 欧几里得距离代替曼哈顿距离，使得总线长估计更准确
                length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                total_length += length * self.weight
        
        return total_length

class ThermalAnalyzer:
    """PCB温度分析类"""
    def __init__(self, modules, board_width, board_height, resolution=20, ambient_temp=25.0):
        self.modules = modules
        self.board_width = board_width
        self.board_height = board_height
        self.resolution = resolution  # 网格精度 mm
        self.ambient_temp = ambient_temp  # 环境温度，单位°C
        
        # 计算网格数量
        self.nx = int(board_width // resolution + 1)
        self.ny = int(board_height // resolution + 1)
        
        # 初始化温度场和热源
        self.T = np.full((self.ny, self.nx), ambient_temp)
        self.power_density = np.zeros((self.ny, self.nx))
        
        # 组件位置记录
        self.component_positions = []
        
    def setup_thermal_model(self):
        """设置热模型，包括热源和初始条件"""
        # 重置温度场和热源
        self.T.fill(self.ambient_temp)
        self.power_density.fill(0.0)
        self.component_positions = []
        
        # 将组件设置为热源
        for module in self.modules:
            # 将mm坐标转为网格索引
            x_idx = min(int(module.x // self.resolution), self.nx-1)
            y_idx = min(int(module.y // self.resolution), self.ny-1)
            
            # 存储组件位置
            self.component_positions.append((x_idx, y_idx, module))
            
            # 设置功率密度
            area_mm2 = self.resolution * self.resolution
            self.power_density[y_idx, x_idx] = module.power / area_mm2 * 100  # 单位修改为W/(100mm²)
    
    def calculate_temperature(self, max_iter=500, tolerance=1e-2):
        """计算温度分布"""
        # 设置热模型
        self.setup_thermal_model()
        
        # 热传导系数
        k_pcb = 0.3  # W/(m·K) FR-4材料的热导率
        heat_transfer_coef = 0.1  # 热传导系数
        
        # 松弛因子，改善收敛性
        alpha = 0.7
        
        # 有限差分法迭代求解温度分布
        for it in range(max_iter):
            T_old = self.T.copy()
            
            # 对非边界点进行迭代计算
            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    # 跳过组件位置，保持为热源
                    if any(x == i and y == j for x, y, _ in self.component_positions):
                        continue
                    
                    # 使用有限差分法计算温度
                    self.T[j, i] = (1-alpha) * T_old[j, i] + alpha * (
                        0.25 * (T_old[j, i+1] + T_old[j, i-1] + T_old[j+1, i] + T_old[j-1, i]) + 
                        heat_transfer_coef * self.power_density[j, i]
                    )
            
            # 对组件位置应用热源方程
            for comp_x, comp_y, module in self.component_positions:
                # 计算热点温度
                surrounding_temp = 0.0
                count = 0
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx_pos = comp_x + dx
                    ny_pos = comp_y + dy
                    if 0 <= nx_pos < self.nx and 0 <= ny_pos < self.ny:
                        surrounding_temp += T_old[ny_pos, nx_pos]
                        count += 1
                
                if count > 0:
                    surrounding_temp /= count
                    
                # 使用简化的热源模型
                thermal_resistance = 5.0  # °C/W
                self.T[comp_y, comp_x] = surrounding_temp + module.power * thermal_resistance
            
            # 边界条件：边缘保持在环境温度
            self.T[0, :] = self.ambient_temp
            self.T[-1, :] = self.ambient_temp
            self.T[:, 0] = self.ambient_temp
            self.T[:, -1] = self.ambient_temp
            
            # 检查收敛
            diff = np.max(np.abs(self.T - T_old))
            if diff < tolerance:
                break
        
        return self.T
    
    def get_max_temperature(self):
        """获取最高温度"""
        self.calculate_temperature()
        return np.max(self.T)
    
    def get_temperature_variance(self):
        """获取温度方差，表示温度均匀性"""
        self.calculate_temperature()
        return np.var(self.T)
    
    def get_component_temperatures(self):
        """获取所有组件的温度"""
        self.calculate_temperature()
        temperatures = {}
        
        for comp_x, comp_y, module in self.component_positions:
            temperatures[module.id] = self.T[comp_y, comp_x]
            
        return temperatures
    
    def visualize_thermal_distribution(self):
        """可视化温度分布"""
        self.calculate_temperature()
        
        plt.figure(figsize=(10, 6))
        im = plt.imshow(self.T, cmap='hot', extent=[0, self.board_width, 0, self.board_height], origin='lower')
        plt.colorbar(label="Temperature (°C)")
        plt.title("PCB Thermal Distribution")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        
        # 标记组件位置和温度
        for comp_x, comp_y, module in self.component_positions:
            temp_value = self.T[comp_y, comp_x]
            plt.text(module.x, module.y-15, module.name, fontsize=10, color='white', ha='center')
            plt.text(module.x, module.y+15, f"{temp_value:.1f}°C", fontsize=8, color='cyan', ha='center')
        
        plt.show()


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
        self.thermal_analyzer = None  # 热分析器，延迟初始化
    
    def copy(self):
        """创建解决方案的副本"""
        new_modules = []
        for m in self.modules:
            new_module = Module(m.id, m.x, m.y, m.shape, m.name, m.width, m.height, m.rotation, m.power)
            new_module.pins = m.pins.copy()
            new_modules.append(new_module)
            
        new_sol = Solution(new_modules, self.nets, self.board_width, self.board_height)
        new_sol.objectives = self.objectives.copy()
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

    def calc_objectives(self):
        """计算目标函数值 - 修改后的版本，包含温度目标并将重叠作为约束"""
        # 1. 总线长
        total_wire_length = sum(net.calc_length(self.modules) for net in self.nets)
        
        # 2. 面积利用率 (转为最小化问题)
        covered_area = sum(m.get_area() for m in self.modules)
        board_area = self.board_width * self.board_height
        area_utilization = 1 - (covered_area / board_area)
        
        # 3. 温度目标 - 最高温度
        # 初始化热分析器（如果尚未初始化）
        if self.thermal_analyzer is None:
            self.thermal_analyzer = ThermalAnalyzer(self.modules, self.board_width, self.board_height)
        else:
            # 更新模块列表
            self.thermal_analyzer.modules = self.modules
        
        # 计算最高温度
        max_temperature = self.thermal_analyzer.get_max_temperature()
        
        # 设置目标函数值（不包含重叠）
        self.objectives = [total_wire_length, area_utilization, max_temperature]
        return self.objectives
        
    def is_feasible(self, min_distance=10):
        """检查布局是否合法（无重叠、在边界内，满足最小间距）- 作为约束"""
        # 检查边界
        for module in self.modules:
            width, height = module.get_dimensions()
            if (module.x < 0 or module.y < 0 or 
                module.x + width > self.board_width or 
                module.y + height > self.board_height):
                return False
        
        # 检查重叠和最小间距
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
        
    def initialize_population(self, modules, nets, manual_layout=None):
        """初始化种群，可选择包含手动布局"""
        population = []
        
        # 如果提供了手动布局，将其作为第一个解决方案
        if manual_layout:
            manual_solution = Solution(modules, nets, self.board_width, self.board_height)
            manual_solution.apply_manual_layout(manual_layout)
            if manual_solution.is_feasible():
                population.append(manual_solution)
                print("已添加手动布局作为初始解决方案")
            else:
                print("警告：手动布局不可行，可能超出边界或存在重叠")
                # 尝试调整使其可行
                adjusted_manual = self.legalization(manual_solution)
                population.append(adjusted_manual)
                print("已添加调整后的手动布局作为初始解决方案")
        
        # 生成剩余的随机解决方案
        while len(population) < self.pop_size:
            solution = Solution(modules, nets, self.board_width, self.board_height)
            solution.randomize_positions()
            
            # 合法化处理随机生成的解
            solution = self.legalization(solution)
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
        """判断解p是否支配解q"""
        better_in_any = False
        for i in range(len(p.objectives)):
            if p.objectives[i] > q.objectives[i]:  # 假设所有目标都是最小化问题
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
        """锦标赛选择"""
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
    
    def legalization(self, solution, min_distance=10):
        """布局合法化处理（考虑旋转和最小间距）- 改进版本"""
        # 边界检查与调整
        for module in solution.modules:
            width, height = module.get_dimensions()
            module.x = max(0, min(module.x, self.board_width - width))
            module.y = max(0, min(module.y, self.board_height - height))
        
        # 解决模块重叠问题
        max_iterations = 200  # 增加最大迭代次数以确保无重叠
        iteration = 0
        
        # 跟踪每次迭代的总重叠量
        prev_overlap = float('inf')
        
        while iteration < max_iterations:
            total_overlap = 0
            overlap_found = False
            
            # 首先检查所有模块对，计算总重叠量
            for i in range(len(solution.modules)):
                for j in range(i + 1, len(solution.modules)):
                    m1 = solution.modules[i]
                    m2 = solution.modules[j]
                    
                    overlap_area = m1.overlap_area(m2)
                    total_overlap += overlap_area
                    
                    if m1.overlaps(m2, min_distance):
                        overlap_found = True
                        
                        # 获取旋转后的尺寸
                        w1, h1 = m1.get_dimensions()
                        w2, h2 = m2.get_dimensions()
                        
                        # 计算当前位置
                        x1, y1 = m1.x, m1.y
                        x2, y2 = m2.x, m2.y
                        
                        # 计算四个方向的移动距离
                        move_right = x2 + w2 + min_distance - x1
                        move_left = x1 + w1 + min_distance - x2
                        move_up = y2 + h2 + min_distance - y1
                        move_down = y1 + h1 + min_distance - y2
                        
                        # 选择移动距离最小的方向
                        min_move = min(move_right, move_left, move_up, move_down)
                        
                        if min_move == move_right:
                            m1.x = x2 + w2 + min_distance
                        elif min_move == move_left:
                            m2.x = x1 + w1 + min_distance
                        elif min_move == move_up:
                            m1.y = y2 + h2 + min_distance
                        else:  # min_move == move_down
                            m2.y = y1 + h1 + min_distance
                        
                        # 边界检查
                        m1.x = max(0, min(m1.x, self.board_width - w1))
                        m1.y = max(0, min(m1.y, self.board_height - h1))
                        m2.x = max(0, min(m2.x, self.board_width - w2))
                        m2.y = max(0, min(m2.y, self.board_height - h2))
            
            # 检查收敛情况
            if not overlap_found:
                # 完全没有重叠，成功
                break
            
            if abs(prev_overlap - total_overlap) < 1e-6:
                # 总重叠量没有变化，尝试随机扰动以跳出局部最优
                rand_module = random.choice(solution.modules)
                w, h = rand_module.get_dimensions()
                rand_module.x = random.uniform(0, self.board_width - w)
                rand_module.y = random.uniform(0, self.board_height - h)
            
            prev_overlap = total_overlap
            iteration += 1
        
        # 如果迭代结束仍有重叠，记录警告
        if iteration >= max_iterations:
            print(f"Warning: Legalization did not fully converge, overlap may still exist.")
        
        # 重新计算目标函数
        solution.calc_objectives()
        return solution
    
    def run(self, modules, nets, manual_layout=None):
        """运行NSGA-II算法"""
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
                
                # 合法化处理
                child1 = self.legalization(child1)
                child2 = self.legalization(child2)
                
                child1.calc_objectives()
                child2.calc_objectives()
                
                offspring.append(child1)
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
            best_temperature = min(population, key=lambda x: x.objectives[2])
            
            print(f"Best wire length solution: Wire={best_wire_length.objectives[0]:.2f}, "
                  f"Area utilization={1 - best_wire_length.objectives[1]:.2f}, "
                  f"Temperature={best_wire_length.objectives[2]:.2f}°C")
            
            print(f"Best temperature solution: Wire={best_temperature.objectives[0]:.2f}, "
                  f"Area utilization={1 - best_temperature.objectives[1]:.2f}, "
                  f"Temperature={best_temperature.objectives[2]:.2f}°C")
        
        # 返回最终帕累托前沿
        return self.non_dominated_sort(population)[0]
    
    def visualize_pareto_front(self, pareto_front):
        """可视化帕累托前沿"""
        if not pareto_front:
            print("No solutions found.")
            return
        
        objectives = np.array([solution.objectives for solution in pareto_front])
        
        # 目标名称
        objective_names = ['Wire Length', 'Area Utilization', 'Max Temperature']
        
        for i in range(len(objectives[0])):
            for j in range(i + 1, len(objectives[0])):
                plt.figure(figsize=(10, 6))
                plt.scatter(objectives[:, i], objectives[:, j])
                plt.xlabel(f'{objective_names[i]}')
                plt.ylabel(f'{objective_names[j]}')
                plt.title(f'Pareto Front: {objective_names[i]} vs {objective_names[j]}')
                plt.grid(True)
                plt.show()
    
    def visualize_layout_improved(self, solution, min_distance=10, show_temperature=True):
        """改进的PCB布局可视化函数（增加温度信息显示）"""
        # 检查和调整组件间距
        solution = adjust_component_spacing(solution, min_distance)
        
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
        
        # 获取温度信息（如果需要）
        module_temperatures = {}
        if show_temperature:
            if solution.thermal_analyzer is None:
                solution.thermal_analyzer = ThermalAnalyzer(solution.modules, solution.board_width, solution.board_height)
            
            # 计算温度分布
            module_temperatures = solution.thermal_analyzer.get_component_temperatures()
        
        # 第一次遍历：绘制所有模块
        for i, module in enumerate(solution.modules):
            width, height = module.get_dimensions()
            
            # 模块颜色可以基于温度（如果可用）
            facecolor = module_colors[i]
            
            # 矩形模块
            rect = patches.Rectangle((module.x, module.y), width, height, 
                                    linewidth=1, edgecolor='black', facecolor=facecolor, 
                                    alpha=0.7)
            ax.add_patch(rect)
            
            # 添加模块标签和温度信息
            center_x = module.x + width/2
            center_y = module.y + height/2
            
            label_text = f"{module.name}\n({module.id})"
            if show_temperature and module.id in module_temperatures:
                label_text += f"\n{module_temperatures[module.id]:.1f}°C"
                
            ax.text(center_x, center_y, label_text, 
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
        
        # 第三次遍历：绘制网络连接
        for net_idx, net in enumerate(solution.nets):
            # 为每个网络使用固定的颜色和线型
            line_color = net_colors[net_idx % len(net_colors)]
            line_style = ['-', '--', '-.', ':'][net_idx % 4]
            
            # 获取该网络所有引脚的位置
            net_pins = []
            for m_id, p_id in net.connections:
                if (m_id, p_id) in pin_positions:
                    net_pins.append((m_id, p_id, pin_positions[(m_id, p_id)]))
            
            # 直接连接网络内的所有引脚对
            for i in range(len(net_pins) - 1):
                for j in range(i + 1, len(net_pins)):
                    m1_id, p1_id, (x1, y1, m1_name) = net_pins[i]
                    m2_id, p2_id, (x2, y2, m2_name) = net_pins[j]
                    
                    # 直线连接，增加线宽以更容易看清
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
        
        # 设置图表参数
        plt.xlim(0, self.board_width)
        plt.ylim(0, self.board_height)
        
        title = 'PCB Layout Visualization'
        if show_temperature:
            # 获取解决方案的温度目标值
            max_temp = solution.objectives[2] if len(solution.objectives) > 2 else None
            if max_temp:
                title += f" (Max Temp: {max_temp:.1f}°C)"
        
        plt.title(title, fontsize=14)
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def visualize_thermal_distribution(self, solution):
        """可视化温度分布"""
        if solution.thermal_analyzer is None:
            solution.thermal_analyzer = ThermalAnalyzer(solution.modules, solution.board_width, solution.board_height)
        
        solution.thermal_analyzer.visualize_thermal_distribution()


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
    # 使用提供的尺寸初始化组件，增加功耗参数
    modules = [
        Module(0, 400, 300, name="C1", width=230, height=130, rotation=0, power=0.1),  # 电容1
        Module(1, 450, 300, name="C2", width=230, height=130, rotation=0, power=0.1),  # 电容2
        Module(2, 300, 100, name="Y1", width=154, height=54, rotation=0, power=0.05),  # 晶振
        Module(3, 350, 200, name="R1", width=232, height=60, rotation=0, power=0.0014),  # 电阻1 (计算功耗)
        Module(4, 400, 200, name="R2", width=232, height=60, rotation=0, power=0.0014),  # 电阻2
        Module(5, 350, 250, name="R3", width=232, height=60, rotation=0, power=0.144),   # 电阻3
        Module(6, 400, 250, name="R4", width=232, height=60, rotation=0, power=0.144),   # 电阻4
        Module(7, 200, 200, 'semicircle_arch', name="Q1", width=79, height=98, rotation=0, power=0.5),  # 晶体管1
        Module(8, 200, 300, 'semicircle_arch', name="Q2", width=79, height=98, rotation=0, power=0.5)   # 晶体管2
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


def main():
    """主函数 - 运行PCB布局优化"""
    # 初始化电子元件和网络连接
    modules, nets = initialize_components()
    
    # 设置PCB板尺寸和算法参数
    board_width = 800
    board_height = 600
    pop_size = 50    # 减小种群大小以加快计算速度
    max_gen = 1     # 减小代数以加快演示
    
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
    
    # === 查看手动布局的效果 ===
    print("\n手动布局的效果:")
    # 创建一个Solution对象并应用手动布局
    manual_solution = Solution(modules, nets, board_width, board_height)
    # 应用手动布局
    for item in manual_layout:
        module_id = item["id"]
        if module_id < len(manual_solution.modules):
            manual_solution.modules[module_id].x = item["x"]
            manual_solution.modules[module_id].y = item["y"]
            manual_solution.modules[module_id].rotation = item["rotation"]
    
    # 计算目标函数值
    objectives = manual_solution.calc_objectives()
    print(f"总线长: {objectives[0]:.2f}")
    print(f"面积利用率: {1 - objectives[1]:.2f}")
    print(f"最高温度: {objectives[2]:.2f}°C")
    
    # 检查布局是否可行
    is_feasible = manual_solution.is_feasible(min_distance=10)
    print(f"布局是否可行: {'是' if is_feasible else '否'}")
    
    # 可视化手动布局
    nsga2.visualize_layout_improved(manual_solution, min_distance=10)
    
    # 可视化手动布局的温度分布
    nsga2.visualize_thermal_distribution(manual_solution)
    
    # 运行NSGA-II算法
    print("\n开始运行NSGA-II算法优化...")
    pareto_front = nsga2.run(modules, nets, manual_layout)
    
    # 可视化结果
    nsga2.visualize_pareto_front(pareto_front)
    
    # 选择和显示多个解决方案
    if pareto_front:
        # 选择线长最短的解
        best_wire_length = min(pareto_front, key=lambda x: x.objectives[0])
        print("\nBest wire length solution:")
        print(f"Total wire length: {best_wire_length.objectives[0]:.2f}")
        print(f"Area utilization: {1 - best_wire_length.objectives[1]:.2f}")
        print(f"Max temperature: {best_wire_length.objectives[2]:.2f}°C")
        nsga2.visualize_layout_improved(best_wire_length, min_distance=10)
        nsga2.visualize_thermal_distribution(best_wire_length)
        
        # 选择最低温度的解
        best_temperature = min(pareto_front, key=lambda x: x.objectives[2])
        if best_temperature != best_wire_length:  # 避免重复显示相同解
            print("\nBest temperature solution:")
            print(f"Total wire length: {best_temperature.objectives[0]:.2f}")
            print(f"Area utilization: {1 - best_temperature.objectives[1]:.2f}")
            print(f"Max temperature: {best_temperature.objectives[2]:.2f}°C")
            nsga2.visualize_layout_improved(best_temperature, min_distance=10)
            nsga2.visualize_thermal_distribution(best_temperature)
        
        # 选择一个综合考虑的解（所有目标的加权和最小）
        def weighted_sum(solution, weights=[0.4, 0.1, 0.5]):  # 温度权重高
            # 归一化目标值
            obj_max = [max(sol.objectives[i] for sol in pareto_front) for i in range(3)]
            obj_min = [min(sol.objectives[i] for sol in pareto_front) for i in range(3)]
            
            norm_obj = [(solution.objectives[i] - obj_min[i]) / (obj_max[i] - obj_min[i] + 1e-10)
                        for i in range(3)]
            
            return sum(w * obj for w, obj in zip(weights, norm_obj))
        
        best_compromise = min(pareto_front, key=weighted_sum)
        if best_compromise not in [best_wire_length, best_temperature]:  # 避免重复
            print("\nBest compromise solution:")
            print(f"Total wire length: {best_compromise.objectives[0]:.2f}")
            print(f"Area utilization: {1 - best_compromise.objectives[1]:.2f}")
            print(f"Max temperature: {best_compromise.objectives[2]:.2f}°C")
            nsga2.visualize_layout_improved(best_compromise, min_distance=10)
            nsga2.visualize_thermal_distribution(best_compromise)
    else:
        print("No solutions found.")


if __name__ == "__main__":
    main()