import numpy as np # 导入 NumPy 库，并将其别名为 np。NumPy 是 Python 中用于科学计算的核心库，提供了高性能的多维数组对象和用于处理这些数组的工具。

from scipy.sparse import csr_matrix # 从 SciPy 库的 sparse 模块中导入 csr_matrix 类。csr_matrix 用于表示稀疏矩阵，它以压缩稀疏行（Compressed Sparse Row）格式存储数据，能有效处理大量零元素的矩阵。
from scipy.sparse.linalg import lsqr # 从 SciPy 库的 sparse.linalg 模块中导入 lsqr 函数。lsqr 是一个迭代求解稀疏线性最小二乘问题的函数，常用于求解 A*x = b 形式的方程，其中 A 是稀疏矩阵。
from sklearn.linear_model import OrthogonalMatchingPursuit # 从 scikit-learn 库的 linear_model 模块中导入 OrthogonalMatchingPursuit (OMP) 类。OMP 是一种贪婪算法，用于稀疏信号恢复和特征选择，通过迭代选择与残差最相关的原子来构建稀疏解。

# 导入自定义的 Sketching 模块中的类
from Sketching.cm_sketch import CMsketch # 导入 CMsketch 类，这可能是一个计数最小素描（Count-Min Sketch）的实现，用于估计数据流中元素的频率。
from Sketching.bloom_filter import BloomFilter # 导入 BloomFilter 类，这可能是一个布隆过滤器（Bloom Filter）的实现，用于快速判断一个元素是否在一个集合中，存在一定的假阳性。
from UCL_sketch.heavy_filter import heavyFilter # 导入 heavyFilter 类，这可能是一个重元素过滤器（Heavy Hitter Filter）的实现，用于识别数据流中出现频率非常高的元素。


class UCLSketch: # 定义一个名为 UCLSketch 的类，它可能是一个基于Sketching算法的数据结构，用于处理网络流量或其他数据流的计数和分析。
    def __init__( # 类的构造函数，在创建 UCLSketch 实例时被调用。
        self, # self 指代类实例本身。
        slot_num: int, # slot_num（槽数量）：用于 heavyFilter 的参数，指定重元素过滤器的槽位数量。
        width: int, # width（宽度）：用于 CMsketch 的参数，指定计数最小素描的宽度，通常对应哈希桶的数量。
        depth: int, # depth（深度）：用于 CMsketch 的参数，指定计数最小素描的深度，通常对应哈希函数的数量。
        bf_width: int, # bf_width（布隆过滤器宽度）：用于 BloomFilter 的参数，指定布隆过滤器的比特数组大小。
        bf_hash: int, # bf_hash（布隆过滤器哈希函数数量）：用于 BloomFilter 的参数，指定布隆过滤器使用的哈希函数数量。
        KEY_T_SIZE=8, # KEY_T_SIZE（键类型大小）：一个可选参数，默认为 8，可能表示键的预期字节大小，用于内部存储或哈希计算。
        decode_mode='ML' # decode_mode（解码模式）：一个可选参数，默认为 'ML'。用于指定如何从 CM Sketch 的结果中解码出流量值。此处'ML'可能为Machine Learning，但代码中只有OMP, LSQR, CM三种模式。
    ):
        self.mode = decode_mode # 将传入的 decode_mode 赋值给实例属性 self.mode。
        self.hTable = heavyFilter(slot_num, KEY_T_SIZE) # 初始化一个 heavyFilter 实例，用于存储和管理“重元素”（Heavy Hitters）。
        self.cm = CMsketch(width, depth, KEY_T_SIZE) # 初始化一个 CMsketch 实例，用于处理和计数次重元素（或所有元素，取决于逻辑）。
        self.bf = BloomFilter(bf_width, bf_hash, KEY_T_SIZE) # 初始化一个 BloomFilter 实例，用于标记哪些元素已经存在于 Sketch 中，以减少重复处理。

        self.milestones = [] # 初始化一个空列表 self.milestones，用于记录不同阶段 flowKeys 和 evictKeys 的长度。
        self.evictKeys = [] # 初始化一个空列表 self.evictKeys，用于存储从 heavyFilter 中被“驱逐”出来，但仍然活跃的键（可能表示“热点”流量）。
        self.flowKeys = [] # 初始化一个空列表 self.flowKeys，用于存储其他流量键（可能表示“正常”流量或“温和”流量）。
        self.cmResult = {} # 初始化一个空字典 self.cmResult，用于存储通过解码算法（如 OMP 或 LSQR）从 CM Sketch 中恢复的键值对。

    def get_keys(self): # 定义一个方法，用于获取当前所有已知的（在 flowKeys 和 evictKeys 中的）键。
        self.milestones.append((len(self.flowKeys), len(self.evictKeys))) # 记录当前 flowKeys 和 evictKeys 的长度，作为一个里程碑。
        keys, index = [], [] # 初始化两个空列表，keys 用于存储所有键，index 用于存储对应的索引。
        s1 = s2 = 0 # 初始化起始索引 s1 和 s2 为 0。
        for milestone in self.milestones: # 遍历之前记录的所有里程碑。
            coldline, hotline = milestone[0], milestone[1] # 从里程碑中获取 flowKeys 和 evictKeys 的长度。
            keys += (self.flowKeys[s1:coldline] + self.evictKeys[s2:hotline]) # 将当前里程碑内的 flowKeys 和 evictKeys 切片并添加到 keys 列表中。
            index += [i for i in range(coldline+s2, coldline+hotline)] # 为这些键生成一个模拟的（或用于后续处理的）索引范围。注意这里的索引计算可能需要根据实际需求调整，这里的索引计算方式看起来有点特别，似乎是为了配合后续的矩阵索引。
            s1, s2 = coldline, hotline # 更新 s1 和 s2 到当前里程碑的长度，为下一次迭代做准备。
        return keys, index # 返回所有收集到的键和对应的索引。这个逻辑看起来是将不同时间点添加到 flowKeys 和 evictKeys 中的键合并起来，并保留它们在合并后的序列中的“相对”索引。

    def insert(self, key, val=1): # 定义一个插入方法，用于向 Sketch 中插入一个键值对。
        evict_or_not, temp_key = self.hTable.insert(key, val) # 调用 heavyFilter 的 insert 方法。
        # evict_or_not 可能表示是否有元素被驱逐 (0: 无驱逐, 1: 有驱逐但新元素是原有元素的更新, -1: 有驱逐且新元素是新加入的)
        # temp_key 是被驱逐的元素（如果发生驱逐），或者传入的 key 的临时表示。
        
        if evict_or_not != 0: # 如果 heavyFilter 发生了驱逐行为（值为 -1 或 1）。
            exist_or_not = self.bf.getbit(temp_key.key) # 检查被驱逐的 temp_key（或其内部的 key 属性）是否在 BloomFilter 中已经存在。
            self.cm.insert(temp_key.key, temp_key.val) # 将被驱逐的键及其值插入到 CM Sketch 中。
            if not exist_or_not: # 如果 BloomFilter 中不存在这个键（说明是首次从 heavyFilter 出来并进入次级存储）。
                self.bf.setbit(temp_key.key) # 在 BloomFilter 中设置这个键的比特位（标记为已存在）。
                if evict_or_not == 1 and temp_key.val > 1: # 如果是因达到容量上限而发生驱逐 (evict_or_not == 1)，并且被驱逐的键的值大于 1。
                    self.evictKeys.append(temp_key.key) # 将这个键添加到 evictKeys 列表中（可能表示热点流量被驱逐）。
                else: # 否则（如果是新的元素被驱逐，或者值不大于1）。
                    self.flowKeys.append(temp_key.key) # 将这个键添加到 flowKeys 列表中（可能表示一般流量）。
            elif evict_or_not == 1 and temp_key.val > 1: # 如果 BloomFilter 中已经存在这个键，并且是因达到容量上限而发生驱逐，且被驱逐的键的值大于 1。
                if temp_key.key not in self.evictKeys: # 并且这个键目前不在 evictKeys 中。
                    try: # 尝试将这个键从 flowKeys 转移到 evictKeys。
                        self.flowKeys.remove(temp_key.key) # 从 flowKeys 中移除。
                        self.evictKeys.append(temp_key.key) # 添加到 evictKeys 中。
                    except: # 如果尝试移除失败（例如，键不在 flowKeys 中），则忽略错误。
                        pass # 这里的 `pass` 意味着不处理任何异常。

    def return_cs_components(self, M: int, N: int): # 定义一个方法，用于返回构造压缩感知（Compressed Sensing）所需的组件。
        # M 和 N 分别表示矩阵 A 的行数和列数（M = CM Sketch 的总单元数，N = 键的数量）。
        b = np.zeros(M,) # 初始化一个 M 维的零向量 b，用于存储 CM Sketch 的计数结果。
        keys, index = self.get_keys() # 获取当前所有活跃的键和它们对应的索引。
        A_data, A_rows, A_cols = [], [], [] # 初始化三个空列表，用于存储稀疏矩阵 A 的数据、行索引和列索引。

        for i in range(self.cm.depth): # 遍历 CM Sketch 的每一行（深度）。
            for j, key in enumerate(keys): # 遍历所有键，j 是键在 keys 列表中的索引。
                idx = i * self.cm.width + self.cm.hash(key, i) # 计算在 CM Sketch 扁平化表示中的索引。
                # self.cm.hash(key, i) 根据键和哈希函数索引 i 计算哈希桶位置。
                A_data.append(1) # 将数据值 1 添加到 A_data 列表，表示在对应位置有贡献（每个键对其哈希到的 CM Sketch 单元贡献 1）。
                A_rows.append(idx) # 将计算出的扁平化索引作为行索引添加到 A_rows。
                A_cols.append(j) # 将键在 keys 列表中的索引 j 作为列索引添加到 A_cols。

            for j in range(self.cm.width): # 遍历 CM Sketch 每一行的所有宽度（列），将 CM Sketch 中的实际计数填充到向量 b。
                b[i * self.cm.width + j] = self.cm.matrix[i][j] # 将 CM Sketch 中 (i, j) 位置的值赋给 b 向量的对应位置。
        
        A = csr_matrix((A_data, (A_rows, A_cols)), shape=(M, N)) # 使用 A_data、A_rows 和 A_cols 构建一个 CSR 格式的稀疏矩阵 A，形状为 (M, N)。
        return A, b, index # 返回构建好的稀疏矩阵 A、计数向量 b 和键的原始索引。

    def solve_equations(self, x=None): # 定义一个方法，用于通过求解方程来估算键的频率。
        # assert x and self.mode=='ML', 'results should not be None in Learning version.' # 被注释掉的断言，可能原意是强制在'ML'模式下 x 不能为空，但实际代码中的解码模式是 OMP 或 LSQR。
        if self.cmResult != {}: # 如果 self.cmResult 字典不为空，说明已经计算过了，直接返回，避免重复计算。
            return
        M = self.cm.depth * self.cm.width # M 是 CM Sketch 的总单元数（行数 * 宽度）。
        keys = self.flowKeys + self.evictKeys # 获取所有用于解码的键。
        N = len(keys) # N 是键的数量。

        if self.mode=='OMP': # 如果解码模式是 'OMP' (正交匹配追踪)。
            A, b, _ = self.return_cs_components(M, N) # 获取压缩感知组件 A 和 b。
            omp = OrthogonalMatchingPursuit() # 实例化 OrthogonalMatchingPursuit 对象。
            x = omp.fit(A.toarray(), b).coef_ # 将稀疏矩阵 A 转换为密集矩阵（toarray()），然后用 OMP 拟合 A 和 b，得到系数 x (即键的估计频率)。
            x[x<1] = 1 # 将所有小于 1 的估计频率值设置为 1（频率至少为 1）。
        elif self.mode=='LSQR': # 如果解码模式是 'LSQR' (最小二乘)。
            A, b, _ = self.return_cs_components(M, N) # 获取压缩感知组件 A 和 b。
            x, *_ = lsqr(A, b) # 使用 lsqr 算法求解 A*x = b 的最小二乘解，得到 x (键的估计频率)。`*`用于忽略lsqr返回的其他值。
            x[x<1] = 1 # 将所有小于 1 的估计频率值设置为 1。

        for i, key in enumerate(keys): # 遍历所有键及其估计频率。
            self.cmResult[key] = x[i] # 将键和其对应的估计频率存储到 self.cmResult 字典中。

    def query(self, key, results=None): # 定义一个查询方法，用于查询某个键的频率。
        table_ans = self.hTable.query(key) # 首先从 heavyFilter（重元素过滤器）中查询键的频率。
        
        if self.mode != 'CM': # 如果解码模式不是 'CM' (表示需要进行复杂的解码，如 OMP 或 LSQR)。
           self.solve_equations(results) # 调用 solve_equations 方法来计算或获取所有键的估计频率。这里的 `results` 参数在函数内部并未被实际使用。
           exist_or_not = self.bf.getbit(key) # 检查布隆过滤器，看这个键是否曾经被记录过（即是否可能存在于 CM Sketch 的解码结果中）。

           if exist_or_not: # 如果布隆过滤器显示键存在。
               try: # 尝试从 self.cmResult 中获取该键的解码频率。
                   cm_ans = self.cmResult[key] # 获取解码结果。
               except: # 如果键不在 self.cmResult 中（例如，尽管布隆过滤器显示存在，但它可能是一个假阳性，或者未参与解码）。
                   cm_ans = 1 # 默认给一个最小值 1。
           else: # 如果布隆过滤器显示键不存在。
               cm_ans = 0 # 则 CM Sketch 的贡献为 0。

        else: # 如果解码模式是 'CM' (表示直接从 CM Sketch 中查询，不进行复杂的解码)。
            cm_ans = self.cm.query(key) # 直接调用 CM Sketch 的 query 方法获取键的频率估计。

        return table_ans + cm_ans # 返回 heavyFilter 和 CM Sketch（或其解码结果）的频率之和，作为最终的查询结果。
    
    def get_current_state(self, return_A=True): # 获取当前 Sketch 的状态（用于机器学习模型的输入）。
        M = self.cm.depth * self.cm.width # M 是 CM Sketch 的总单元数。
        keys = self.flowKeys + self.evictKeys # 获取所有活跃的键。
        N = len(keys) # N 是键的数量。
        if return_A: # 如果 return_A 为 True，则返回矩阵 A 和键索引。
            A, _, index = self.return_cs_components(M, N) # 调用 return_cs_components 获取 A 矩阵。
            return A.A, index # 返回 A 矩阵的密集表示 (A.A) 和索引。
        
        b = np.zeros(M,) # 否则，初始化一个 M 维的零向量 b。
        for i in range(self.cm.depth): # 遍历 CM Sketch 的每一行。
            for j in range(self.cm.width): # 遍历每一列。
                b[i * self.cm.width + j] = self.cm.matrix[i][j] # 将 CM Sketch 的内容填充到 b 向量中。
    
        return b.reshape(1, self.cm.depth, self.cm.width) # 返回 b 向量，并将其重塑为 (1, depth, width) 的形状（这可能是为了某些特定ML模型输入格式）。
    
    def refresh(self): # 定义一个刷新方法，用于重置 UCLSketch 的状态，清空内部计数和缓存。
        self.cmResult = {} # 清空 CM Sketch 的解码结果。
        self.milestones = [] # 清空里程碑记录。
        self.evictKeys = [] # 清空被驱逐的键列表。
        self.flowKeys = [] # 清空流量键列表。
        self.cmResult = {} # 再次清空 CM Sketch 的解码结果（冗余，但无害）。
    
    def get_memory_usage(self): # 定义一个方法，用于计算并打印 UCLSketch 实例的内存使用情况。
        ht_size = self.hTable.get_memory_usage() # 获取 heavyFilter 的内存使用量。
        bf_size = self.bf.get_memory_usage() # 获取 BloomFilter 的内存使用量。
        cm_size = self.cm.get_memory_usage() # 获取 CM Sketch 的内存使用量。
        
        print("----- Memory Usage -----") # 打印内存使用标题。
        print(f"Hash Table Size(Byte): {ht_size} ({ht_size / 1024:.2f} KB)") # 打印 Hash Table 的内存使用量，同时显示 KB 值。
        print(f"Bloom Filter Size(Byte): {bf_size} ({bf_size / 1024:.2f} KB)") # 打印 Bloom Filter 的内存使用量，同时显示 KB 值。
        print(f"CM Sketch Size(Byte): {cm_size} ({cm_size / 1024:.2f} KB)") # 打印 CM Sketch 的内存使用量，同时显示 KB 值。
        print(f"Total Memory(MB): {(ht_size + cm_size + bf_size) / 1024:.2f} KB") # 打印总内存使用量，同时显示 KB 值 (注意这里写的是MB，但计算结果是KB)。
        print("------------------------") # 打印分隔线。
        
        return ht_size + cm_size + bf_size # 返回总内存使用量（字节）。
