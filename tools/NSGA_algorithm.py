import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
from functools import wraps

from library.pcb_layout_model import *

class Solution:
    """表示一个PCB布局解决方案"""
    def __init__(self, modules, nets, board_width, board_height):
        self.modules = modules
        self.nets = nets  # 网络连接
        self.board_width = board_width
        self.board_height = board_height
        self.objectives = []  # 目标函数值
        self.rank = 0  # 非支配排序的等级
        self.crowding_distance = 0  # 拥挤度距离
        self.dominated_solutions = []  # 被该解支配的解
        self.domination_count = 0  # 支配该解的个数
        self.overlap = 0  # 存储重叠面积，用于约束判断

        self.max_temp = None
        self.min_temp = None
        self.temperature = None
        self.resolution = None
        self.nx = None
        self.ny = None
    
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