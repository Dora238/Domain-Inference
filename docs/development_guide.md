# Domain-Inference 开发文档

## 项目概述

Domain-Inference的目标是针对语言分类模型，我们希望得到每个label对应的知识。
- 得到一个包含label对应知识的embedding
- 通过embedding引导语言生成模型生成对应label的文本
- 用ChatGPT对生成的文本进行concept extraction


### 核心功能

1. **初始化Embedding**：基于WordNet词汇网络的条件文本生成
2. **设计合理的generator/decoder**：发现和利用语言模型中隐含的领域知识
3. **优化Embedding**：通过优化Embedding，引导generator生成符合label的文本
4. **多样性与成功率平衡**：在生成文本时平衡多样性和目标领域成功率

## 系统架构

Domain-Inference 系统由以下核心组件构成：

### 1. 黑盒分类器（BlackBox）

用于评估生成文本是否符合目标领域。

- 位置：`blackbox.py`
- 主要功能：
  - 预测文本的领域类别
  - 为优化过程提供反馈

### 2. WordNet条件生成器（WordNetConditioner）

基于WordNet词汇网络创建可学习的提示向量。

- 位置：`wordnet_conditioner.py`
- 主要功能：
  - 从WordNet初始化词汇嵌入
  - 聚类相关词汇
  - 生成领域特定的条件向量

### 3. 软提示优化器（Optimizer）

优化soft prompt以满足目标领域要求。

- 位置：`optimizer.py`和`generater.py`
- 主要功能：
  - 优化x_start嵌入向量
  - 计算文本多样性
  - 估计梯度和优化参数

### 4. main函数

- 位置：`cli.py`
- 主要功能：
  - 组织和调用各组件
  - 运行完整的领域发现流程

## 核心流程

1. **初始化组件**：加载DiffusionTextGenerator、WordNetConditioner和BlackBox
2. **WordNet条件生成**：从WordNet获取领域相关词汇并生成初始嵌入
3. **维度优化**：通过`optimize_dimension`优化嵌入维度
4. **软提示动态优化**：通过`optimize_soft_prompt_dynamic`优化软提示向量
5. **文本生成与评估**：使用优化后的向量生成文本并评估成功率和多样性
6. **结果保存**：将优化结果保存到输出目录
