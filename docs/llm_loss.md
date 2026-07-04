大语言模型最核心、最常用的损失函数是**交叉熵损失（Cross-Entropy Loss）**，等价地也可称为**负对数似然损失（Negative Log-Likelihood, NLL）**。但在预训练、指令微调和偏好对齐等不同阶段，具体目标有所不同。

## 1. 自回归预训练：下一词预测损失

GPT、LLaMA、Qwen 等生成式大语言模型通常采用自回归语言建模。给定词元序列

[
x_1,x_2,\ldots,x_T,
]

模型将联合概率分解为

[
p_\theta(x_)
============

\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
]

训练目标是最大化训练文本的对数似然：

[
\max_\theta
\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t}).
]

等价地，最小化负对数似然：

[
\mathcal_}
==========

-\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t}).
]

通常取 token 平均：

[
\mathcal_}
==========

-\frac{1}{T}
\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t}).
]

若第 (t) 个位置模型输出词表上的概率分布

[
p_
==

\frac{\exp(z_{t,j})}
{\sum_{k=1}^{V}\exp(z_{t,k})},
]

其中 (V) 是词表大小，那么该位置的交叉熵损失为

[
\mathcal_t
==========

-\sum_{j=1}^{V}y_{t,j}\log p_{t,j}.
]

真实标签 (y_t) 通常是 one-hot，因此可化简为

[
\mathcal{L}*t=-\log p*{t,x_t}.
]

也就是说，模型对正确 token 分配的概率越高，损失越小。

---

## 2. 为什么使用交叉熵

交叉熵适合大语言模型，主要因为下一词预测本质上是一个**大规模多分类问题**：

[
\text{context }x_{<t}
\longrightarrow
\text{next token }x_t.
]

交叉熵具有以下性质：

1. **直接对应最大似然估计**
   最小化交叉熵等价于最大化真实文本的生成概率。
2. **可以训练完整概率分布**
   模型不仅给出一个 token，还学习整个词表上的条件分布。
3. **对高置信度错误惩罚较强**
   如果正确 token 的概率接近 0，

   [
   -\log p_{t,x_t}
   ]

   会非常大。
4. **可微且容易通过反向传播优化**。

---

## 3. 掩码语言模型损失

BERT 一类模型采用 Masked Language Modeling，随机遮挡一部分 token，只在遮挡位置计算损失。

设遮挡位置集合为 (\mathcal M)，则

[
\mathcal_}
==========

-\frac{1}{|\mathcal M|}
\sum_{t\in\mathcal M}
\log p_\theta(x_t\mid x_{\setminus \mathcal M}).
]

与 GPT 的区别是：

* GPT：根据左侧上下文预测下一个 token；
* BERT：根据双向上下文恢复被遮挡 token。

现代生成式 LLM 主要采用前者。

---

## 4. 指令微调中的损失

在 Supervised Fine-Tuning，SFT 阶段，通常仍然使用 token-level 交叉熵。

训练样本形式为

[
x=(\text{instruction},\text{input}),
\qquad
y=(y_1,\ldots,y_T),
]

目标为

[
\mathcal_}
==========

-\sum_{t=1}^{T}
\log p_\theta(y_t\mid x,y_{<t}).
]

不过通常只对 assistant 的回答部分计算损失，而不对 system prompt 和用户输入计算损失。通过 mask 可写为

[
\mathcal_}
==========

-\frac{
\sum_{t=1}^{T}m_t
\log p_\theta(x_t\mid x_{<t})
}{
\sum_{t=1}^{T}m_t
},
]

其中

[
m_t=
\begin{cases}
1, & \text{assistant 输出 token},
0, & \text{prompt 或 padding token}.
\end{cases}
]

---

## 5. RLHF 中的损失

RLHF 通常包含两个不同的训练目标。

### 5.1 奖励模型损失

对于同一 prompt (x)，人工标注一个更优回答 (y_w) 和较差回答 (y_l)。奖励模型输出标量：

[
r_\phi(x,y).
]

使用 Bradley–Terry 排序损失：

[
\mathcal_}
==========

-\log
\sigma
\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right),
]

其中 (\sigma(\cdot)) 是 sigmoid 函数。

该损失要求

[
r_\phi(x,y_w)>r_\phi(x,y_l).
]

### 5.2 PPO 策略优化目标

使用奖励模型训练语言模型时，目标不仅要获得高奖励，还要避免模型偏离原始模型过远。典型目标可概括为

[
\max_\theta
\mathbb E_
\left[
r_\phi(x,y)
-----------

\beta
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid x)
|
\pi_{\mathrm{ref}}(\cdot\mid x)
\right)
\right].
]

其中：

* (r_\phi(x,y))：回答的奖励；
* (\pi_{\mathrm{ref}})：参考模型；
* KL 项：限制策略模型过度偏离参考模型；
* (\beta)：控制约束强度。

---

## 6. DPO 中的损失

Direct Preference Optimization 不显式训练奖励模型，也不需要在线强化学习。对于偏好回答 (y_w) 和非偏好回答 (y_l)，DPO 损失为

[
\mathcal_}
==========

\log\sigma
\left[
\beta
\left(
\log
\frac
}(y_w\mid x)}
-------------

\log
\frac{\pi_\theta(y_l\mid x)}
{\pi_{\mathrm{ref}}(y_l\mid x)}
\right)
\right].
]

它鼓励模型相对于参考模型：

* 提高偏好回答的概率；
* 降低非偏好回答的概率。

---

## 7. 常见辅助损失

大型模型还可能加入其他损失项。

### 负载均衡损失

MoE 模型为了避免所有 token 都被分配给少数专家，会加入负载均衡损失：

[
\mathcal
========

\mathcal{L}*{\mathrm{LM}}
+
\lambda*{\mathrm{balance}}
\mathcal{L}_{\mathrm{balance}}.
]

### KL 蒸馏损失

知识蒸馏中，学生模型拟合教师模型的输出分布：

[
\mathcal_}
==========

T^2
D_{\mathrm{KL}}
\left(
p_{\mathrm{teacher}}^{(T)}
|
p_{\mathrm{student}}^{(T)}
\right).
]

### 多模态对比损失

视觉语言模型还可能使用图像—文本对比学习损失，例如 InfoNCE：

[
\mathcal_}
==========

-\log
\frac{
\exp(\operatorname{sim}(v_i,t_i)/\tau)
}{
\sum_j
\exp(\operatorname{sim}(v_i,t_j)/\tau)
}.
]

---

## 总结

不同训练阶段的典型损失可以概括为：

| 训练阶段      | 主要损失                        |
| ------------- | ------------------------------- |
| 自回归预训练  | token-level 交叉熵 / 负对数似然 |
| BERT 式预训练 | masked-token 交叉熵             |
| 指令微调 SFT  | 回答部分的 token-level 交叉熵   |
| 奖励模型训练  | pairwise ranking loss           |
| RLHF / PPO    | 奖励目标 + KL 正则化            |
| DPO           | 直接偏好优化损失                |
| MoE           | 语言建模损失 + 负载均衡损失     |
| 模型蒸馏      | 交叉熵 + KL 散度                |

其中最基础、占训练计算量最大的仍然是

[
\boxed
======

-\frac{1}{T}
\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t})
}
]

即**下一 token 预测的交叉熵损失**。
