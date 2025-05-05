
# 🧠 HW4-1: 靜態模式的樸素 DQN 理解報告

## ✅ 作業目標與內容說明

本次作業目標為理解並實作 Deep Q-Network（DQN）於靜態模式下的應用，掌握強化學習核心流程，並建立對 Replay Buffer、Epsilon-Greedy 策略、Q-value 更新機制的實作理解。

---

## 📚 理論背景與 DQN 起源

### 強化學習簡介

強化學習（Reinforcement Learning, RL）是一種基於回饋學習的決策方式，讓智能體（Agent）透過與環境互動獲取獎勵（Reward），進而學習最適策略（Policy）。傳統方法如 Q-Learning 需維護一張 Q-table，當狀態空間變大時將變得不可行。

### 深度 Q 網路（Deep Q-Network, DQN）

DQN 是 DeepMind 團隊於 2015 年提出的方法，透過神經網路逼近 Q 函數，使得 Q-Learning 可以處理高維度狀態空間，首次實現用單一架構玩多個 Atari 遊戲並達人類水準。其關鍵創新包括：
1. **Experience Replay**
2. **Target Network**
3. **梯度下降更新 Q 網路**

---

## 🔧 DQN 架構與核心模組實作說明

### 1. Q-Network 建構

```python
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```
- 使用三層全連接神經網路，適用於低維狀態空間。
- ReLU 激活函數能有效避免梯度消失。

### 2. 損失計算與 Q 更新

```python
q_target = reward + gamma * torch.max(q_next, dim=1)[0] * (1 - done)
q_eval = q_net(state).gather(1, action.unsqueeze(1)).squeeze()
loss = F.mse_loss(q_eval, q_target.detach())
```

- 若 `done` 為 True，則不考慮下一狀態的 Q 值。
- 使用 `.detach()` 避免 target value 影響梯度更新。

### 3. Epsilon-Greedy 策略探索

```python
if random.random() < epsilon:
    action = env.action_space.sample()
else:
    with torch.no_grad():
        action = q_net(state).argmax().item()
```

> 初期高探索，逐步衰減 epsilon 增加利用率。

---

## ♻️ Replay Buffer 緩衝區設計

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
```

- 經驗樣本以五元組 (s, a, r, s', done) 儲存。
- 抽樣時打亂順序，打破資料相關性，提升學習穩定性。

---

## 🧪 實驗流程與設定

### 訓練流程 Step-by-Step

1. 初始化 Q 網路與 Replay Buffer
2. 重複以下步驟直到完成訓練
   - 取得當前狀態並選擇動作（依 epsilon-greedy）
   - 執行動作並獲取環境回饋
   - 儲存五元組至 Replay Buffer
   - 隨機抽樣經驗並訓練 Q 網路
   - 遞減 epsilon（探索率）

### 超參數設定參考

| 參數         | 說明                        | 建議值或範圍       |
|--------------|-----------------------------|--------------------|
| learning rate| 學習率                      | 1e-3 ~ 1e-4         |
| gamma        | 折扣因子                    | 0.95 ~ 0.99         |
| epsilon      | 探索率初始值                | 1.0，線性衰減到 0.1 |
| buffer size  | Replay Buffer 大小          | 10000 ~ 50000       |
| batch size   | 每次訓練抽樣數量            | 32 或 64            |

---

## 🧩 改進方向與常見問題

| 問題 | 解法 |
|------|------|
| Q 值振盪不穩 | 引入 Target Network |
| Q 值高估 | 使用 Double DQN |
| 收斂速度慢 | 加快 epsilon 衰減、使用學習率調度器 |
| 探索效果不佳 | 採用 Boltzmann 策略或 Noisy Net |
| 樣本利用率低 | 使用 Prioritized Experience Replay |

---

## 💬 與 ChatGPT 討論重點摘要

- 深入探討 `.detach()` 的作用與 Backpropagation 行為。
- 對 Replay Buffer 記憶體結構與 sample 行為進行 trace。
- 測試不同 batch size 對學習穩定性的影響。
- 建議加入 reward shaping 改善策略偏差問題。

---

## 📊 可補上圖表與分析（預留位置）

- 📈 **學習曲線**（episode reward 對 episode 數）
- 📉 **epsilon 衰減視覺化**
- 🎯 **成功率變化趨勢**

---

## 📝 總結

本作業透過靜態模式的 DQN 實作，從底層理解強化學習各組件之間的關聯與設計目的。Replay Buffer 的設計大幅度提升樣本效率與學習穩定性，而 epsilon-greedy 策略展現了探索與利用的權衡哲學。若未來進一步導入 Target Network、Double DQN、Dueling DQN 等技巧，預期能顯著改善策略表現與學習速度。



---


# 🧠 HW4-1: 靜態模式的樸素 DQN 理解報告（含詳細程式碼說明）

## ✅ 作業目標與內容說明

本作業要求學生了解並實作 DQN（Deep Q-Network）演算法，並透過靜態模式學習其核心流程與實作細節。本報告將以程式碼為主軸，逐段說明各模組邏輯與其對學習的貢獻。

---

## 🧠 1. 建立 Q-Network 模型

```python
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
```

### 模型架構說明：
- `state_size`：輸入狀態空間維度，例如 (4,) 表示環境狀態向量長度為 4。
- `action_size`：動作空間個數，例如在 CartPole 中為 2（左移或右移）。
- `hidden_size`：隱藏層神經元個數，通常設定為 64、128 或 256。
- `fc1 ~ fc3`：三層全連接層（Linear Layer），最終輸出對每個動作的 Q 值估計。

```python
    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第一層輸入後進行 ReLU 激活
        x = F.relu(self.fc2(x))  # 第二層也使用 ReLU
        return self.fc3(x)       # 最終輸出對應各個動作的 Q 值向量
```

> ReLU 激活有助於避免梯度消失問題，使網路能更有效學習非線性關係。

---

## 💾 2. 經驗重播緩衝區（Replay Buffer）

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
```

- 使用 deque 實作 FIFO 緩衝區，`maxlen=capacity` 限制最大儲存數量，超過時自動移除最舊資料。

```python
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
```

- 每次與環境互動後，將經驗五元組記錄下來。

```python
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
```

- 隨機抽取樣本可打破時間序列依賴，有助於訓練穩定。
- `zip(*)` 解構 batch 為多個 array，`np.stack` 將它們轉為批次輸入格式。

---

## 🎯 3. 動作選擇（Epsilon-Greedy）

```python
if random.random() < epsilon:
    action = env.action_space.sample()
else:
    with torch.no_grad():
        action = q_net(state).argmax().item()
```

- 當隨機值 < epsilon 時，選擇隨機動作（探索）。否則使用 Q 網路選擇最大值動作（利用）。
- 使用 `with torch.no_grad()` 可避免計算圖構建，加快推論速度。

> 衰減 epsilon 可實現從「多探索」漸進轉向「多利用」，是一種典型的 exploration-exploitation 平衡方式。

---

## 🔁 4. Q 值更新流程

```python
q_next = target_net(next_state_tensor).detach()
q_target = reward_tensor + gamma * q_next.max(1)[0] * (1 - done_tensor)
q_eval = q_net(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze()
loss = F.mse_loss(q_eval, q_target)
```

- `q_next.max(1)[0]`：代表下個狀態中所有動作的最大 Q 值（對應 Bellman 最佳值估計）。
- `detach()`：避免 target Q 值參與梯度反傳。
- `gather(1, action)`：提取當前所選動作對應的 Q 值。

---

## 🏋️‍♂️ 5. 訓練與參數更新

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

- `zero_grad()`：每個 batch 訓練前重置梯度。
- `loss.backward()`：反向傳播誤差。
- `optimizer.step()`：根據梯度更新參數。

> 最常用的優化器為 Adam，具有自動調整學習率與梯度規範的優點。

---

## ✅ 建議補充的訓練主迴圈框架

```python
for episode in range(MAX_EPISODES):
    state = env.reset()
    total_reward = 0
    for step in range(MAX_STEPS):
        action = select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(buffer) >= BATCH_SIZE:
            update_q_network(buffer.sample(BATCH_SIZE))
        
        if done:
            break
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
```

- `MAX_EPISODES`：最大訓練回合數
- `total_reward` 可記錄當前 episode 成果
- 可依照 episode plot 出學習曲線

---

## 📘 總結：DQN 各模組的協作關係

- `QNetwork` 是核心估計器，持續學習 Q 值函數。
- `ReplayBuffer` 保留大量過往經驗以穩定訓練。
- `Epsilon-Greedy` 為策略平衡探索與利用。
- 損失計算根據 Bellman equation 指導學習方向。
- 使用 `detach()`、`optimizer` 等 PyTorch 特性使模型訓練更加穩定與高效。

以上說明不僅有助於理解靜態 DQN 實作，更為進一步導入改進架構（如 Dueling、Prioritized、Double DQN）打下基礎。
