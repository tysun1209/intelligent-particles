# 用强化学习控制自推进粒子的转向

一个最小的连续控制示例：让二维周期方盒中的定速粒子学习向东运动。每个粒子读取自身朝向与邻居的平均朝向，选择一个转角。环境按新方向推进粒子并加入旋转扩散。每一步的奖励为 $r_i=\cos\theta_i$，其中 $\theta_i$ 是推进后相对于正东方向的夹角。

## 安装并运行

需要 Python 3.10 或更高版本。在仓库根目录执行：

```bash
python -m pip install -e ".[learning]"
python examples/train_steering.py
```

示例用 16 个粒子训练一个共享策略，单步转向幅度上限为 0.3 弧度。训练后冻结策略，在新的种子上输出每个粒子的平均累计奖励。默认三个训练回合用于演示接口，评估学习效果时应增加训练量，并比较多次独立训练的结果。[训练与评估入口](examples/train_steering.py)

## 学习方法

策略网络输出一个高斯分布的均值与标准差，训练时从中采样，经 `tanh` 得到一个归一化转角。环境执行的转角等于该值乘以 `max_turn`。评估时直接使用均值经 `tanh` 变换后的转角。粒子共享网络参数，每个粒子根据自身观测独立决策。

训练采用单动作的软演员–评论家算法（SAC）。不同粒子的观测、动作、个体奖励和下一步观测存入同一回放池。更新时对均匀抽取的小批量计算两个目标 Q 值，取较小值并加入熵项，更新价值网络和策略网络。示例使用固定熵温度和软目标更新。[SAC](src/intelligent_particles/models/sac.py) · [经验回放](src/intelligent_particles/replay.py) · [训练循环](src/intelligent_particles/learning.py)

## 修改实验

| 改动 | 模块 |
|---|---|
| 粒子可以感知什么 | [observation.py](src/intelligent_particles/observation.py) |
| 怎样解释策略输出的动作 | [action.py](src/intelligent_particles/action.py) |
| 粒子要学会什么 | [reward.py](src/intelligent_particles/reward.py) |
| 粒子怎样推进、受到多大噪声 | [dynamics.py](src/intelligent_particles/dynamics.py) |
| 粒子数量、初态、周期边界 | [environment.py](src/intelligent_particles/environment.py) |

例如，在训练入口将 `eastward_reward` 换为同一模块中的 `alignment_reward`，粒子的奖励就变为自身朝向与群体平均朝向的点积。重新训练后，可以用固定的局部对齐规则作对照。[对照示例](examples/periodic_alignment.py)
