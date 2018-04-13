# DQN军火库
起源：DQN
降低高估bias：Double DQN
引入Dueling结构：Dueling DQN
简单并行但有效：A3C类的DQN
使用LSTM unit：DRQN
引入层次结构（任务上）：h-DQN
在网络参数空间引入noise：NoisyNet
引入attention机制：DARQN
减少近似误差方差：Averaged-DQN
count每个state：\#Exploration
对memory中采样做权重：Prioritized Experience Replay
有钱就是任性啊：并行的Prioritized Experience Replay

todo：
c51，PGQ等...一堆扩展

## 引子
最开始的DQN如同一声惊雷，吸引了无数研究者的注意，alpha-go的出现更让外界对于DRL这个领域开始注目。从之前的几乎每个老师都说自己做Deep learning，感觉慢慢也有一些老师也开始说自己做DRL了，潮流真是有趣，在目光集中之后，也不乏有泼冷水着，not work yet，没有随机的好之类的博客，论文也慢慢浮现。感觉自己不免也跟着外界的喧嚣而浮躁，目光飘忽，没有太多集中，所以特别梳理一下DQN及其相应组件的发展，尝试了解一下研究的进展，为之后的研究打打更好的基础。另外就是再挖一个坑，争取对每个扩展都写一个更详细的介绍文章，坑多不压身。

----------------------------------------
## 起源：DQN
### 目的
从row pixel直接输入，end-to-end的方式训练agent的控制policy
### 效果
同一套参数在多数atari game有非常棒的参数
### 实际做法
采用神经网络作为Q-tabel的函数近似
#### experience replay
将agent与环境交互的(s, a, r, s')在一个memory中。在学习时，直接对memory中的数据做均匀的采样。可以有效的缓解数据直接的相关性，而且能够利用Q-learning的off policy的性质，更有效的利用数据
#### target network
使用两个DQN，其中一个DQN的参数固定在一定时间之前，然后一段时间更新一次（更新为target network的参数）。另外DQN(称为target network)不断与环境交互，并一定时间learning一下。DQN的loss为：
$$
L = (r + \gamma * max_{a'}Q_{target}(s', a') - Q(s, a))^{2}
$$
通过target network, 在一段时间内target network的参数是不变的，所以计算出来的$max_{a'}Q_{target}(s', a')$是不变的，能够缓解学习目标不断改变导致的震荡
### 论文
Playing Atari with Deep Reinforcement Learning
Human-level control through deep reinforcement learning

----------------------------------------
## 降低高估bias：Double DQN
### 目的
降低max op引入的高估bias
### 效果
学习过程更稳定，在一部分游戏上，最终的学校效果更好
### 实际做法
借鉴Double Q-learning的方式，但是不是对两个Q值一起做learning，而是采用当前与环境交互的DQN作为选取max action的函数，target network作为max Q值的估计函数。只修改了loss函数：
$$
L = (r + \gamma * Q_{target}(s', argmax_{a'}Q(s',a')) - Q(s, a))^{2}
$$
降低高估偏差使得learning过程中更稳定
### 论文
Deep Reinforcement Learning with Double Q-learning

----------------------------------------
## 引入Dueling结构：Dueling DQN
### 目的
借鉴Advantage的思想，采用Dueling的结构，将advantage 与 state value分开学（隐式地学），能够更好地面对state value接近的情况
### 效果
通过实验验证dueling的结构能够比较好分离action与state之间的影响（见Enduro的图）
学习过程中squared error更低，并最终在一些game效果有提升
### 实际做法
![](https://raw.githubusercontent.com/wwxFromTju/DQN-Arsenal/master/media/dueling.png)
上面的结构为普通的DQN结构，下面的结构为Dueling的结构，分叉两支分别为state（记为V）与action（记为A），将最终的Q(s, a)的计算写成:
$$
Q(s, a) = V(s) + (A(s, a) - \frac{1}{|A|}\sum_{a'}A(s, a'))
$$
loss可以采用传统DQN，也可以采用Double DQN
### 论文
Dueling Network Architectures for Deep Reinforcement Learning

----------------------------------------
## 简单并行但有效：A3C类的DQN
### 目的
利用并行探视探索不同的环境（或者参数略微不同）带来的不相关性，去除replay buffer
### 效果
简单的想法，又快又好
### 实际做法
不使用replay buffer，直接开多个环境（不同进程之类的），然后可以控制rand seed之类的，然后在每个进程中累积梯度，然后在一个时间后更新全局的参数
### 论文
Asynchronous Methods for Deep Reinforcement Learning

----------------------------------------
## 使用LSTM unit：DRQN
### 目的
改变以前forward，reactor的结构，使得policy能够利用时序上的信息
### 效果
在一些部分可观察，或者需要前后frame推断信息的设置下，DRQN（Deep Recurrent Q-Network）能够有解决相应环境下的learning
### 实际做法
![](https://raw.githubusercontent.com/wwxFromTju/DQN-Arsenal/master/media/drqn.png)
如图所示，将最后的fc层修改为LSTM层（相应的结构即可），但是memory存的(s, a, r, s')需要修改为存一整个episode，然后更新的时候是抽episode来做更新，可以从s0开始，也可以从episode中随机的一个s开始。
### 论文
Deep Recurrent Q-Learning for Partially Observable MDPs

----------------------------------------
## 引入层次结构（任务上）：h-DQN
### 目的
缓解稀疏reward环境中探索，学习的困难。在不同尺度下进行学习
### 效果
在普通DQN类（DDQN之类的）几乎无效的Montezuma’s Revenge中能够学习到有效的策略
### 实际做法
![](https://raw.githubusercontent.com/wwxFromTju/DQN-Arsenal/master/media/h-dqn.png)
将原来一个学习目标分解为不同层次，一个层次是选择goal（具有多个goal），一个层次是最大化采用这个goal时的累积收益。即采用一个goal，然后做多次底层的action，然后再采取一个goal（高层的action）。所以高层就是一个普通的DQN（或者之类的），学习选择goal，最大化高层的reward（每个goal时底层action产生的累积收益），然后底层就是传统DQN，修改输入，多了一个goal作为输入，其他与DQN类似

### 论文
Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation

----------------------------------------
## 在网络参数空间引入noise：NoisyNet
### 目的
更好地进行探索
### 效果
相比$\epsilon greedy$和entropy bonus有更好的效果，并在一些atari game上的最终学习效果比较好
### 实际做法
将传统的层中加入noise的部分：
$$
y_{noise} = y_{normal} + (W_{noise} \odot \epsilon_{w})x + b_{noise} \odot \epsilon_{b}
$$
简单理解为：在普通的层的基础上，引入了noise的参数$W_{noise}与b_{noise}$，这两个参数与普通的参数一起学习，然后随机sample 对应的噪声$\epsilon_{w} 与 \epsilon_{b}$, 所以可以理解为学习noise的scale的尺度
### 论文
Noisy Networks for Exploration


----------------------------------------
## 引入attention机制：DARQN
### 目的
更好的效果 + 通过attention来分析policy，并进行监控
### 效果
在有些game下好，但是也有一部分没有DQN好，但是可视化的时候的确可以解释当前需要关注什么
![](https://raw.githubusercontent.com/wwxFromTju/DQN-Arsenal/master/media/darqn.png)

### 实际做法
分为soft和hard两种做法。如图所示：
![](https://raw.githubusercontent.com/wwxFromTju/DQN-Arsenal/master/media/darqn2.png)
对于g的部分为：$h_t 与 v_{t+1}$的函数，soft采用是计算出g后，采用g与v的加权和作为z。hard为只指定一个（所以可以用分布，然后抽样）
### 论文
Deep Attention Recurrent Q-Network

----------------------------------------
## 减少近似误差方差：Averaged-DQN
### 目的
降低估计error的方差，那么估计越准，采样就可以越少，学习也可以更快更稳定
### 效果
在一些atari上（比如breakout，seaquest，asterix），还有自己的提出的grid world上效果比较好
### 实际做法
保存之前k个不同参数的DQN（FIFO的方式替换），然后使用这k个DQN来计算平均的target：
$$
Q_{target}(s, a) = \frac{1}{K}\sum_{i}^{k}Q(s, a| \theta_{i})
$$
然后可以直接结合其他的DQN来计算·
### 论文
Averaged-DQN: Variance Reduction and Stabilization for Deep Reinforcement Learning

----------------------------------------
## count每个state：\#Exploration
### 目的
更细粒度的控制探索率，从全局采用同一个探索率，改变成针对每个state的探索控制
### 效果
在一些连续控制问题与atari上面有更好的效果
### 实际做法
对于state采用hash的方法，小的空间（比如一些连续控制）可以直接用SimHash，大的空间（比如atari的pixel输入）可以采用类似自编码器的结构学习，如图：
![](https://raw.githubusercontent.com/wwxFromTju/DQN-Arsenal/master/media/count-exploration.png)

然后设置额外的探索bound为：$\frac{\beta}{\sqrt{n（\phi(s_m)}）}$,其中$n（\phi(s_m)）$为访问次数，每次递增1即可，$\phi(s_m)$为对一个的hash code
### 论文
\#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning


----------------------------------------
## 对memory中采样做权重：Prioritized Experience Replay
### 目的
缓解一下稀疏reward环境下导致的memory中无用数据太多，sample出几乎对learning无用的数据（比如r为0）而似的最终学的比较慢，甚至学不到的情况（这个目的和h-DQN还是很像的，但是这里的Montezuma’s Revenge几乎没有效果）
### 效果
在设置的toy的稀疏mdp下，还有一部分atari中能够学的比较快，而且最终效果也不错
### 实际做法
采用TD-error作为每个(s, a, r, s')权重，然后可以采用Rank-based prioritization抽取（可以采用array-based binary heap作为数据结构），也可以采用Proportional prioritization（可以采用array-based binary heap作为数据结构）
### 论文
PRIORITIZED EXPERIENCE REPLAY

----------------------------------------
## 对memory中采样做权重：Prioritized Experience Replay
### 目的
缓解一下稀疏reward环境下导致的memory中无用数据太多，sample出几乎对learning无用的数据（比如r为0）而似的最终学的比较慢，甚至学不到的情况（这个目的和h-DQN还是很像的，但是这里的Montezuma’s Revenge几乎没有效果）
### 效果
在设置的toy的稀疏mdp下，还有一部分atari中能够学的比较快，而且最终效果也不错
### 实际做法
采用TD-error作为每个(s, a, r, s')权重，然后可以采用Rank-based prioritization抽取（可以采用array-based binary heap作为数据结构），也可以采用Proportional prioritization（可以采用array-based binary heap作为数据结构）
### 论文
PRIORITIZED EXPERIENCE REPLAY

----------------------------------------
## 有钱就是任性啊：并行的Prioritized Experience Replay
### 目的
有钱，想要更快的学习（利用更多的机器，更有效的利用Prioritized Experience Replay，而且利用了类似A3C的探索效果）
### 效果
论文里面的原话：substantially improves the state of the art on the Arcade Learning Environment, achieving better final performance in a fraction of the wall-clock training time.
### 实际做法
![](https://raw.githubusercontent.com/wwxFromTju/DQN-Arsenal/master/media/distributed-prioritized-experience-replay.png)
采用类似的结构，一个learn一直在学习，然后调整权重，然后不同的actor探索不同的数据，然后初始化优先级
### 论文
DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY




