### GPT問答紀錄：https://chatgpt.com/share/67f68e6a-0934-8010-adfe-ae75f35fd5e5

---

## 🎰 Epsilon-Greedy Algorithm

### (1) 演算法公式（LaTeX）

$$
a_t =
\begin{cases}
\text{random action} & \text{with probability } \varepsilon \\
\arg\max_a Q_t(a) & \text{with probability } 1 - \varepsilon
\end{cases}
$$

$$
Q_{t+1}(a) = Q_t(a) + \alpha (R_t - Q_t(a))
$$

---

### (2) 解釋該算法的關鍵邏輯或分析

Epsilon-Greedy 是一種平衡「探索」（exploration）與「利用」（exploitation）的方法。在每一步中，它以機率 $\varepsilon$ 隨機選擇一個動作（探索），以機率 $1 - \varepsilon$ 選擇目前估計獎勵最高的動作（利用）。這種策略能避免總是選擇已知最佳動作，從而可能發現更好的選擇。  
選擇的 epsilon 值會顯著影響學習的結果，較大的 epsilon 會加強探索、避免早期過度利用；較小的 epsilon 會加快收斂但風險是錯失更優選項。

---

### (3) 程式碼與圖表（Python + Matplotlib）

```python
import numpy as np
import matplotlib.pyplot as plt

class EpsilonGreedy:
    def __init__(self, k_arm=10, epsilon=0.1, steps=1000):
        self.k = k_arm
        self.epsilon = epsilon
        self.steps = steps
        self.q_true = np.random.normal(0, 1, k_arm)
        self.q_est = np.zeros(k_arm)
        self.action_count = np.zeros(k_arm)
        self.rewards = []

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.q_est)

    def run(self):
        for _ in range(self.steps):
            action = self.select_action()
            reward = np.random.normal(self.q_true[action], 1)
            self.action_count[action] += 1
            self.q_est[action] += (reward - self.q_est[action]) / self.action_count[action]
            self.rewards.append(reward)
        return self.rewards

# 執行與繪圖
np.random.seed(0)
agent = EpsilonGreedy(epsilon=0.1)
rewards = agent.run()
cumulative_rewards = np.cumsum(rewards)

plt.plot(cumulative_rewards)
plt.title("Epsilon-Greedy: cumulative rewards")
plt.xlabel("steps")
plt.ylabel("accumulated reward")
plt.grid()
plt.show()
```
![image](https://github.com/user-attachments/assets/673f61c2-888e-4a02-8832-c9913088b157)

---

### (4) 結果解釋
- 時間面向分析：Epsilon-Greedy 的收斂速度取決於 epsilon 值。當 epsilon 趨近於 0 時，算法傾向於 exploitation，但容易陷入局部最優；epsilon 趨近於 1 時則幾乎完全探索，導致收斂變慢。
- 空間面向分析：儲存估計值與計數器所需空間為 $O(k)$，非常輕量，適合大規模問題。
- 效果比較：
  - 優點：簡單、效率高、實現容易。
  - 缺點：固定的 epsilon 無法根據學習進度調整探索率。

---

## 📈 UCB (Upper Confidence Bound) Algorithm

### (1) 演算法公式（LaTeX）

$$
a_t = \arg\max_a \left[ Q_t(a) + c \cdot \sqrt{\frac{\ln t}{N_t(a)}} \right]
$$

其中：
- $Q_t(a)$：第 $t$ 步對動作 $a$ 的平均估計獎勵  
- $N_t(a)$：第 $t$ 步之前動作 $a$ 被選擇的次數  
- $c$：調節探索程度的常數  
- $t$：目前的時間步

---

### (2) 解釋該算法的關鍵邏輯或分析

UCB 演算法基於樂觀初始原則（Optimism in the Face of Uncertainty）。它在選擇動作時，除了考慮當前的平均估計獎勵外，還會加上一個「不確定性懲罰項」，該項根據動作被選擇的頻率調整。  
若某個動作很少被選擇，其不確定性較高，UCB 會傾向給予較高的「信賴上限」，鼓勵探索這些尚不確定的選項。這樣的設計能自然平衡探索與利用，不需額外設定 epsilon。

---

### (3) 程式碼與圖表（Python + Matplotlib）

```python
import numpy as np
import matplotlib.pyplot as plt

class UCB:
    def __init__(self, k_arm=10, c=2, steps=1000):
        self.k = k_arm
        self.c = c
        self.steps = steps
        self.q_true = np.random.normal(0, 1, k_arm)
        self.q_est = np.zeros(k_arm)
        self.action_count = np.zeros(k_arm)
        self.rewards = []

    def select_action(self, t):
        ucb_values = self.q_est + self.c * np.sqrt(np.log(t + 1) / (self.action_count + 1e-5))
        return np.argmax(ucb_values)

    def run(self):
        for t in range(self.steps):
            action = self.select_action(t)
            reward = np.random.normal(self.q_true[action], 1)
            self.action_count[action] += 1
            self.q_est[action] += (reward - self.q_est[action]) / self.action_count[action]
            self.rewards.append(reward)
        return self.rewards

# 執行與繪圖
np.random.seed(0)
agent = UCB(c=2)
rewards = agent.run()
cumulative_rewards = np.cumsum(rewards)

plt.plot(cumulative_rewards)
plt.title("UCB: cumulative rewards")
plt.xlabel("steps")
plt.ylabel("accumulated reward")
plt.grid()
plt.show()

```
![image](https://github.com/user-attachments/assets/3d8831d3-2e90-4c83-a29d-14f82cc25580)

---

### (4) 結果解釋
- 時間面向分析：UCB 演算法會優先探索尚未被選擇或選擇次數少的動作。這種策略在早期探索充分，在後期集中利用已知最佳選項，收斂速度通常比 epsilon-greedy 更快。
- 空間面向分析：與 Epsilon-Greedy 相同，只需儲存每個動作的估計值與次數，空間複雜度為 $O(k)$。
- 效果比較：
  - 優點：不需手動設定探索機率，能自動調整探索強度。
  - 缺點：對 $c$ 的敏感度較高，若設定不當可能導致過度探索或收斂不佳。

---

## 🔥 Softmax Algorithm

### (1) 演算法公式（LaTeX）

$$
P(a_t = a) = \frac{e^{Q_t(a)/\tau}}{\sum_{b=1}^{k} e^{Q_t(b)/\tau}}
$$

其中：
- $Q_t(a)$：動作 $a$ 在時間 $t$ 的估計值
- $\tau$：溫度參數（temperature），用來控制動作機率分佈的平滑程度  
  - $\tau \rightarrow 0$：更偏向 exploitation  
  - $\tau \rightarrow \infty$：更偏向 uniform 探索

---

### (2) 解釋該算法的關鍵邏輯或分析

Softmax 演算法使用機率性選擇動作，每個動作被選擇的機率與其估計價值呈指數比例關係。透過「溫度參數」$\tau$ 來控制探索程度，當 $\tau$ 較低時更偏好選擇估計值高的動作；$\tau$ 較高時動作選擇更隨機。

與 Epsilon-Greedy 不同的是，Softmax 不會硬性將探索與利用分離，而是根據各選項的價值分佈做加權選擇，這使得它在處理相近價值的動作時更加穩定與連貫。

---

### (3) 程式碼與圖表（Python + Matplotlib）

```python
import numpy as np
import matplotlib.pyplot as plt

class Softmax:
    def __init__(self, k_arm=10, tau=0.1, steps=1000):
        self.k = k_arm
        self.tau = tau
        self.steps = steps
        self.q_true = np.random.normal(0, 1, k_arm)
        self.q_est = np.zeros(k_arm)
        self.action_count = np.zeros(k_arm)
        self.rewards = []

    def softmax_prob(self):
        exp_est = np.exp(self.q_est / self.tau)
        return exp_est / np.sum(exp_est)

    def select_action(self):
        probs = self.softmax_prob()
        return np.random.choice(self.k, p=probs)

    def run(self):
        for _ in range(self.steps):
            action = self.select_action()
            reward = np.random.normal(self.q_true[action], 1)
            self.action_count[action] += 1
            self.q_est[action] += (reward - self.q_est[action]) / self.action_count[action]
            self.rewards.append(reward)
        return self.rewards

# 執行與繪圖
np.random.seed(0)
agent = Softmax(tau=0.1)
rewards = agent.run()
cumulative_rewards = np.cumsum(rewards)

plt.plot(cumulative_rewards)
plt.title("Softmax: 累積獎勵")
plt.xlabel("步驟")
plt.ylabel("累積獎勵")
plt.grid()
plt.show()
```
![image](https://github.com/user-attachments/assets/490b699b-edf3-48b2-9aa2-29aa962edf33)

---

### (4) 結果解釋
- 時間面向分析：Softmax 可平滑地在探索與利用之間做切換，收斂速度受到 $\tau$ 的強烈影響。當 $\tau$ 適中時，它能快速辨識並集中在高回報動作；若 $\tau$ 太高，則會過度探索而拉長收斂時間。
- 空間面向分析：與其他演算法相同，所需儲存的估計值與計數器為 $O(k)$。
- 效果比較：
  - 優點：具備連續性與平滑性的機率控制方式，避免硬切換行為，對相近價值的動作特別有效。
  - 缺點：需要選擇合適的 $\tau$ 值，且對此超參數較為敏感；若 $\tau$ 太小，可能變得近似貪婪策略。

---

## 🎯 Thompson Sampling Algorithm

### (1) 演算法公式（LaTeX）

對於每個動作 $a$，維護其 Beta 分布參數 $(\alpha_a, \beta_a)$，每一步：
1. 為每個動作 $a$ 取樣：
   $$
   \theta_a \sim \text{Beta}(\alpha_a, \beta_a)
   $$
2. 選擇最大值對應的動作：
   $$
   a_t = \arg\max_a \theta_a
   $$
3. 根據觀察到的獎勵 $r_t \in \{0,1\}$ 更新參數：
   $$
   \alpha_a = \alpha_a + r_t,\quad \beta_a = \beta_a + (1 - r_t)
   $$

> 備註：此處我們以 Bernoulli Bandit 問題（獎勵為 0 或 1）作為例子。

---

### (2) 解釋該算法的關鍵邏輯或分析

Thompson Sampling 是一種基於貝葉斯推論的策略，它為每個動作維護一個機率分布（通常為 Beta 分布），用以表示該動作為最優的可能性。  
每一步都從這些分布中取樣，並選擇取樣結果最大的動作。這樣的機制在自然中融合了探索與利用——尚未被嘗試過的動作會有較大的不確定性，容易被取樣到；反之，被多次觀察後的動作若表現佳，會穩定地被選中。

這使得 Thompson Sampling 通常能取得很好的累積獎勵表現，特別是在 Bernoulli 類型的問題中。

---

### (3) 程式碼與圖表（Python + Matplotlib）

```python
import numpy as np
import matplotlib.pyplot as plt

class ThompsonSampling:
    def __init__(self, k_arm=10, steps=1000):
        self.k = k_arm
        self.steps = steps
        self.q_true = np.random.beta(2, 2, k_arm)  # 真實機率（0~1）
        self.alpha = np.ones(k_arm)
        self.beta = np.ones(k_arm)
        self.rewards = []

    def select_action(self):
        theta = np.random.beta(self.alpha, self.beta)
        return np.argmax(theta)

    def run(self):
        for _ in range(self.steps):
            action = self.select_action()
            reward = np.random.binomial(1, self.q_true[action])  # 獎勵為 0 或 1
            self.alpha[action] += reward
            self.beta[action] += 1 - reward
            self.rewards.append(reward)
        return self.rewards

# 執行與繪圖
np.random.seed(0)
agent = ThompsonSampling()
rewards = agent.run()
cumulative_rewards = np.cumsum(rewards)

plt.plot(cumulative_rewards)
plt.title("Thompson Sampling: 累積獎勵")
plt.xlabel("步驟")
plt.ylabel("累積獎勵")
plt.grid()
plt.show()
```
![image](https://github.com/user-attachments/assets/72ae3bf6-0020-474c-84a3-c9a05c3859f5)

---

### (4) 結果解釋
- 時間面向分析：Thompson Sampling 在初期會大量探索，但隨著參數收斂，它能快速集中於高回報動作。相比其他策略，它通常擁有更平滑與穩定的學習曲線。
- 空間面向分析：需要為每個動作維護 $(\alpha, \beta)$ 兩個參數，因此空間複雜度為 $O(k)$，與其他策略相近。
- 效果比較：
  - 優點：自然平衡探索與利用、不需人工調參（如 $\varepsilon$ 或 $\tau$），在多數實驗中表現優異。
  - 缺點：需根據問題類型選擇適當的貝葉斯分布；在非二元回報或未知分布下，實作可能較為複雜。
 
---

## 📊 四種 MAB 演算法統整比較表

| 演算法             | 探索方式                           | 是否機率性選擇 | 主要超參數         | 收斂速度     | 優點                                                       | 缺點                                                         |
|------------------|----------------------------------|----------------|------------------|------------|----------------------------------------------------------|------------------------------------------------------------|
| Epsilon-Greedy   | 機率性隨機探索 $\varepsilon$          | 否              | $\varepsilon$     | 中等         | 實作簡單，容易理解                                          | 探索與利用是硬切分， $\varepsilon$ 固定可能不靈活                 |
| UCB              | 基於置信區間，自動平衡探索與利用          | 否              | $c$（探索強度）     | 快           | 無需額外探索機率，自動調整探索程度                              | 對 $c$ 值敏感，設定不當可能導致過度探索或利用                        |
| Softmax          | 根據動作估值以 softmax 機率選擇         | 是              | $\tau$（溫度參數） | 中等         | 平滑控制探索強度，對估值接近的動作表現更穩定                        | 對 $\tau$ 敏感，若設定太小則近似貪婪策略；太大則過度探索              |
| Thompson Sampling | 從每個動作的後驗分布中取樣              | 是              | 初始先驗參數（如 $\alpha, \beta$） | 快           | 自然融合探索與利用、表現穩定、不需明確探索機率                       | 實作需根據問題設計合適的貝葉斯模型，非 0/1 獎勵時會較複雜              |

---

✅ **結論建議**：
- 若想快速上手且控制簡單，可從 **Epsilon-Greedy** 開始。
- 若需要穩定表現與自動化探索策略，**UCB** 與 **Thompson Sampling** 表現通常較佳。
- 若希望機率性探索，且有連續控制需求，可考慮 **Softmax**。

---



