三种数据结构：HeavyFilter   Sketch  BloomFilter

数据结构：

1. HF 追踪频繁的键值对 跟踪热点

![](https://cdn.nlark.com/yuque/0/2025/png/40454586/1754816137715-dd301ca8-b861-48b3-981c-b5e3193c8abd.png)

2. Sketch 记录剩余项 记录被过滤掉、非热点的键

![](https://cdn.nlark.com/yuque/0/2025/png/40454586/1754816154021-b2550805-84a8-4a39-a3c9-3facfc403a5a.png)

3. BF用于键识别 判断是否出现过键

![](https://cdn.nlark.com/yuque/0/2025/png/40454586/1754816165402-01ec52f4-e0f0-445b-9057-0642ebd84a13.png)

键值对插入过程：

算法1：

对键值对KV

s1：计算HF索引

 i=hash(k)

s2: 检查HF[i]是否空余

`if (HF[i].key == NULL)： `

`HF[i].key = k , HF[i].new = v , HF[i].old = 0`

`if (HF[i].key == k)  key相同：`

`HF[i].new += v 值累加到槽`

`if（HF[i].key != k） key不同：`

HF中的哈希位置：

case1：槽位空、现有相同的键值——插入该位置

case2：当前键值不同、新计数 < 旧计数   新项目顶替，旧项目进入草图，更新BF

case3：当前键值不同，新计数 > 旧计数 新项目插入草图
