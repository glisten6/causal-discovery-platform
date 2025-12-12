---
inclusion: always
---

## 交流偏好
- 默认使用中文进行所有交流和代码注释
- 代码变量和函数名使用英文，但注释必须使用中文
- 错误信息和调试信息优先使用中文解释

# Product Overview

因果发现与可视化平台，用于构建、编辑和分析因果图（贝叶斯网络）。

## Core Purpose
- 可以上传候选因果结构，根据这个候选因果图进行带有先验知识的因果算法，然后专家可以修改，迭代进行因果发现算法。
- 从观测数据中发现因果关系（使用结构学习算法）
- 提供交互式Web界面进行因果图可视化和编辑
- 支持因果图的版本控制
- 允许领域专家对算法发现的关系进行增强和修正

## Key Features
- **因果发现 (Causal Discovery)**: 从CSV数据自动学习结构，支持PC、NOTEARS、GES、LiNGAM等算法
- **图编辑 (Graph Editing)**: 添加/删除/更新节点和边，支持关系类型和来源属性
- **可视化 (Visualization)**: 使用PyVis进行交互式图渲染，支持布局持久化
- **版本控制 (Version Control)**: 跟踪图修改历史，支持回滚
- **品种管理 (Variety Management)**: 独立管理多个因果图（称为"品种"）

## Key Concepts
- **Variety（品种）**: 一个独立的因果图实例，存储在`variety_jsons/`目录
- **Pending Changes**: 图修改在提交前暂存，支持接受/拒绝工作流
- **StructureModel**: CausalNex中的核心图结构类

## Application Domain
主要面向农业因果分析（作物产量、土壤因素、气候条件），但可泛化到任何因果发现场景。

## Development Guidelines
- 后端API返回JSON格式的图结构
- 前端使用vis.js渲染交互式网络图
- 图数据流: CSV → discover_file() → StructureModel → JSON → CausalGraph → PyVis → HTML
