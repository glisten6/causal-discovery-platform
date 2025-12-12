## 单调性约束的因果发现

### 📖 概述

这个模块实现了带单调性约束的因果发现数据生成和实验。

**核心思想**：
1. 生成 DAG 结构
2. 为部分边指定单调性（递增/递减）
3. 使用单调非线性函数生成数据
4. 提取单调边作为先验知识
5. 用先验约束改进因果发现

---

### 📂 文件说明

#### `monotonic_data_generation.py`
**单调性 DAG 生成器**

主类：`MonotonicDAGGenerator`

**功能**：
- 生成随机 DAG
- 标记单调边（递增/递减/非单调）
- 使用多种单调/非单调函数生成数据
- 提取单调边矩阵作为先验

**单调函数类型**：
- **递增**：线性、sigmoid、log、sqrt、power
- **递减**：负线性、负指数
- **非单调**：二次、sin、tanh

#### `baseline_monotonic.py`
**单调性约束实验脚本**

完整的实验流程，对比有无单调性先验的效果。

---

### 🚀 快速开始

#### 1. 生成单调性数据

```python
from monotonic_data_generation import MonotonicDAGGenerator

# 创建生成器
gen = MonotonicDAGGenerator(
    n_nodes=20,          # 节点数
    n_edges=40,          # 边数
    monotonic_ratio=0.6, # 60% 边是单调的
    seed=42
)

# 打印摘要
gen.print_summary()

# 生成数据
X = gen.generate_data(n_samples=1000, noise_scale=1.0)

# 获取单调边约束
constraint = gen.get_monotonic_edges_matrix(only_monotonic=True)

# 或者采样部分单调边
constraint_sampled = gen.sample_monotonic_edges(sample_ratio=0.5)

# 可视化
gen.visualize(save_path='monotonic_dag.png')
```

#### 2. 运行完整实验

```bash
cd algorithm/CIR
python baseline_monotonic.py
```

#### 3. 查看结果

结果保存在 `algorithm/CIR/exp/monotonic/` 目录下：
- `comparison.png`: 对比图表
- `{nodes}_{multiplier}/`: 每个配置的详细结果

---

### 📊 单调性矩阵说明

生成器的 `monotonic_matrix` 编码了每条边的单调性：

| 值 | 含义 |
|----|------|
| 0 | 无边 |
| 1 | 单调递增边 |
| -1 | 单调递减边 |
| 2 | 非单调边（复杂非线性）|

**示例**：
```python
monotonic_matrix = [
    [0,  1,  0, -1],  # 节点0: →1递增, →3递减
    [0,  0,  2,  0],  # 节点1: →2非单调
    [0,  0,  0,  1],  # 节点2: →3递增
    [0,  0,  0,  0]   # 节点3: 无出边
]
```

---

### 🔧 单调函数详解

#### 递增函数

| 函数类型 | 公式 | 特点 |
|---------|------|------|
| `linear_pos` | $w \cdot x$ | 线性递增 |
| `sigmoid` | $w \cdot \frac{2}{1+e^{-s \cdot x}} - w$ | S 型递增，有界 |
| `log` | $w \cdot \text{sign}(x) \cdot \log(1+\|x\|)$ | 对数递增，增速递减 |
| `sqrt` | $w \cdot \text{sign}(x) \cdot \sqrt{\|x\|}$ | 平方根递增 |
| `power` | $w \cdot \text{sign}(x) \cdot \|x\|^p$ (p<1) | 幂函数递增 |

#### 递减函数

| 函数类型 | 公式 | 特点 |
|---------|------|------|
| `linear_neg` | $-w \cdot x$ | 线性递减 |
| `exp_neg` | $-w \cdot \text{sign}(x) \cdot (1 - e^{-\|x\|})$ | 指数递减，有界 |

#### 非单调函数

| 函数类型 | 公式 | 特点 |
|---------|------|------|
| `quadratic` | $w \cdot x^2$ | 二次函数，V 型 |
| `sin` | $w \cdot \sin(f \cdot x)$ | 周期振荡 |
| `tanh_flip` | $w \cdot (x - 0.5 \tanh(x))$ | 复杂非单调 |

---

### 🎯 使用场景

#### 场景 1：测试单调性先验的有效性

```python
# 生成数据
gen = MonotonicDAGGenerator(n_nodes=30, n_edges=60, monotonic_ratio=0.7)
X = gen.generate_data(n_samples=500)

# 获取单调边先验
constraint = gen.sample_monotonic_edges(sample_ratio=0.5)

# 对比实验
model_no_prior = NotearsNonlinear()
model_no_prior.learn(X)

config = {"orient": {"use": True, "alpha": 3.0, ...}}
model_with_prior = NotearsNonlinear(
    config=config,
    candidate_dict={"orient": constraint}
)
model_with_prior.learn(X)

# 比较效果
```

#### 场景 2：不同单调性比例的影响

```python
monotonic_ratios = [0.2, 0.4, 0.6, 0.8]

for ratio in monotonic_ratios:
    gen = MonotonicDAGGenerator(
        n_nodes=20,
        n_edges=40,
        monotonic_ratio=ratio
    )
    # ... 运行实验 ...
```

#### 场景 3：采样率影响

```python
sample_ratios = [0.3, 0.5, 0.7, 1.0]

for ratio in sample_ratios:
    constraint = gen.sample_monotonic_edges(sample_ratio=ratio)
    # ... 运行实验 ...
```

---

### 📈 实验结果解读

#### 预期结果

**当单调性先验准确时**：
- ✅ **SHD 降低**：结构距离减小
- ✅ **召回率提升**：找到更多真实边
- ✅ **精确率提升**：误报边减少
- ✅ **F1 分数提升**：综合性能更好

**单调性先验的优势**：
1. **方向信息**：单调性隐含因果方向
2. **稳定性**：单调函数梯度行为更稳定
3. **可解释性**：符合领域知识（如经济学、生物学）

---

### 🔬 进阶用法

#### 自定义函数类型

修改 `_assign_functions` 方法添加新函数：

```python
def _assign_functions(self):
    # ...
    if mono_type == 1:  # 递增
        func_type = np.random.choice([
            'linear_pos',
            'sigmoid',
            'custom_function',  # 添加新函数
            # ...
        ])
        
        if func_type == 'custom_function':
            functions[(i, j)] = {
                'type': 'custom_function',
                'weight': weight,
                'params': {'a': 1.0, 'b': 2.0}
            }
```

然后在 `_apply_function` 中实现：

```python
def _apply_function(self, x, func_info):
    # ...
    elif func_type == 'custom_function':
        a = params['a']
        b = params['b']
        return weight * (a * x + b * x**0.5)
```

#### 添加噪声类型

```python
def generate_data(self, n_samples, noise_scale=1.0, noise_type='gaussian'):
    # ...
    if noise_type == 'gaussian':
        noise = np.random.randn(n_samples) * noise_scale
    elif noise_type == 'uniform':
        noise = np.random.uniform(-noise_scale, noise_scale, n_samples)
    elif noise_type == 'exponential':
        noise = (np.random.exponential(noise_scale, n_samples) - noise_scale)
    
    X[:, node] = contribution + noise
```

#### 混合单调性

部分边单调，部分边非单调：

```python
# 生成器1：高单调性（代表简单系统）
gen1 = MonotonicDAGGenerator(monotonic_ratio=0.8)

# 生成器2：低单调性（代表复杂系统）
gen2 = MonotonicDAGGenerator(monotonic_ratio=0.3)

# 对比实验...
```

---

### ⚙️ 参数调优

#### `monotonic_ratio`（单调边比例）

| 值 | 含义 | 适用场景 |
|----|------|---------|
| 0.2-0.4 | 低单调性 | 复杂非线性系统 |
| 0.5-0.7 | 中等单调性 | 一般系统（推荐） |
| 0.8-1.0 | 高单调性 | 简单系统、经济模型 |

#### `sample_ratio`（先验采样比例）

| 值 | 含义 | 效果 |
|----|------|------|
| 0.3 | 少量先验 | 轻度改进 |
| 0.5 | 中等先验 | 明显改进（推荐） |
| 0.7-1.0 | 大量先验 | 显著改进，但可能过拟合 |

#### `noise_scale`（噪声强度）

| 值 | SNR | 难度 |
|----|-----|------|
| 0.5 | 高 | 简单 |
| 1.0 | 中 | 中等（推荐） |
| 2.0 | 低 | 困难 |

---

### 🐛 故障排除

#### 问题 1：生成的 DAG 边数不足

**原因**：随机生成可能无法达到目标边数

**解决**：
```python
# 增加尝试次数
gen = MonotonicDAGGenerator(n_nodes=20, n_edges=50)
actual_edges = (gen.W != 0).sum()
print(f"实际生成 {actual_edges} 条边")
```

#### 问题 2：单调性检验失败

**原因**：函数实现可能有误

**解决**：可视化函数检查
```python
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
func_info = {'type': 'sigmoid', 'weight': 1.0, 'params': {'scale': 1.0}}
y = [gen._apply_function(xi, func_info) for xi in x]

plt.plot(x, y)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Function Visualization')
plt.grid(True)
plt.show()
```

#### 问题 3：约束矩阵全0

**原因**：单调边采样比例过低或 `monotonic_ratio` 太小

**解决**：
```python
# 检查单调边数量
n_monotonic = (np.abs(gen.monotonic_matrix) == 1).sum()
print(f"单调边数量: {n_monotonic}")

# 增加单调性比例
gen = MonotonicDAGGenerator(monotonic_ratio=0.7)

# 或增加采样比例
constraint = gen.sample_monotonic_edges(sample_ratio=0.8)
```

---

### 📚 参考文献

1. **单调性与因果推断**
   - Peters, J., et al. (2014). "Causal inference by using invariant prediction"
   
2. **约束因果发现**
   - Zheng, X., et al. (2018). "DAGs with NO TEARS"
   
3. **单调函数在因果中的应用**
   - Hoyer, P. O., et al. (2009). "Nonlinear causal discovery with additive noise models"

---

### 💡 总结

**优势**：
- ✅ 真实反映单调性关系（常见于自然/社会系统）
- ✅ 提供有效的先验知识
- ✅ 可控的数据生成过程
- ✅ 易于扩展和定制

**注意事项**：
- ⚠️ 真实世界不一定有这么高的单调性
- ⚠️ 采样的先验可能包含错误
- ⚠️ 需要根据具体问题调整参数

**下一步**：
- 🔬 在真实数据集上验证
- 🎯 结合领域知识提取单调性先验
- 📊 对比不同约束类型的效果






