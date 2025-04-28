# Homework 4: DQN and its variants（正式版詳細中文報告）

---

# 1. 作業要求確認

本次作業共分為三大部分：

1. ⚙️ HW4-1: 簡單環境下的基本 DQN 實現
   - 運行提供的程式碼或體驗緩衝區
   - 與 ChatGPT 討論代碼以澄清理解
   - 提交一份簡短的理解報告
   - 實作 Experience Replay Buffer

2. ⚖️ HW4-2: 玩家模式下的增強 DQN 變體
   - 實施並比較 Double DQN 與 Dueling DQN
   - 強調改進基本 DQN 方法的技巧

3. 🔁 HW4-3: 隨機模式下增強 DQN，並加入訓練技巧
   - 將 DQN 轉換為 keras 或 PyTorch Lightning
   - 整合訓練技術（如梯度削減、學習率調度等）以改善穩定性
   - 實作加分項目（穩定技巧）

---

# 2. HW4-1：Naive DQN for Static Mode

## 背景與目標
目標是實作最基礎的 Deep Q-Learning (DQN)，在玩家與目標位置固定的簡單環境中學習最佳策略。藉此熟悉 Experience Replay、Target Network 和 epsilon-greedy 策略。

## 訓練流程
1. 初始化 Static Mode 環境
2. 建立簡單 2-layer Fully Connected 的 DQN 網路
3. 設定 Replay Buffer、optimizer、loss function
4. 每回合依據 epsilon-greedy 策略選擇行動
5. 儲存經驗進 Replay Buffer，開始批次訓練
6. 定期同步 target network
7. 每50回合記錄 reward

## 技術說明
- **Experience Replay**：隨機取樣過去經驗，提升學習穩定性
- **Target Network**：穩定Q值估計
- **Epsilon Decay**：逐步減少探索，增強 exploitation

### 重要程式碼片段
```python
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```

## 訓練結果與曲線
![](1edbe2c9-7eff-44cf-81fc-8f447f7cfe31.png)

- 起初 reward 波動大
- 隨訓練次數增加，reward 趨於上升
- 最後 reward 穩定達成目標

## 遇到的問題與修正
- 初期 epsilon decay 太快 → 調整 decay 速度
- Replay Buffer 太小導致 overfitting → 增加 buffer size

## 小結與改進建議
- 若環境更複雜，可引入 Double DQN 防止 Q-value 過高估計
- 可嘗試加入 reward clipping 穩定學習

---

# 3. HW4-2：Enhanced DQN Variants for Player Mode

## 背景與目標
隨機初始位置的玩家使得任務更具挑戰性。為此需要更穩定且精確的 DQN 改良版。

## 訓練流程
1. 初始化 Player Mode 環境
2. 建立 Double DQN 與 Dueling DQN
3. 每回合根據 epsilon-greedy 選擇行動
4. 儲存並訓練經驗
5. 每N回合同步 target network
6. 分別收集 Double / Dueling 的 reward 曲線

## 技術說明
- **Double DQN**：將 action selection 與 target evaluation 分開，避免過高估計
- **Dueling DQN**：分開估計狀態價值與動作優勢，加速學習

### 重要程式碼片段
```python
# Double DQN Target
next_actions = policy_net(next_states).argmax(1)
next_q_values = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
```

```python
# Dueling DQN Forward
def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    value = self.value(x)
    advantage = self.advantage(x)
    return value + (advantage - advantage.mean(dim=1, keepdim=True))
```

## 訓練結果比較與曲線
![](d80e846c-76c2-4bf3-8716-0911fdb29db5.png)

- Double DQN 表現穩定，Dueling DQN 初期收斂速度較快
- Dueling Q-learning 在 early exploration 特別有效

## 遇到的問題與修正
- Dueling初期過度估計問題 → 增加 buffer 更新頻率
- Double在稀疏獎勵時收斂慢 → 延長 epsilon decay 時間

## 小結與改進建議
- 可嘗試結合 Double + Dueling
- 可以加上 Prioritized Replay 或 Multi-Step TD 改進取樣與更新方式

---

# 4. HW4-3：Enhance DQN for Random Mode with Training Tips

## 背景與目標
隨機環境變數增加，要求模型能在大範圍情境下穩定學習。引入訓練技術來穩定與提升效果。

## 訓練流程
1. 轉換 DQN 成 PyTorch Lightning 架構
2. 實作 Double DQN 與 Dueling DQN Lightning 版本
3. 加入 Gradient Clipping、Learning Rate Scheduler
4. 每回合記錄 reward，分析曲線表現

## 技術說明
- **PyTorch Lightning**：模組化訓練，提高易讀性與可擴展性
- **Gradient Clipping**：防止梯度爆炸
- **Learning Rate Scheduler**：動態調整學習率，促進收斂

### 重要程式碼片段
```python
self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
loss.backward()
torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
self.optimizer.step()
```

## 訓練結果比較與曲線
![](50ad5473-380e-4fa5-9477-36242673afe3.png)

- Double DQN 在 high-variance 隨機環境中表現較穩
- Dueling DQN early learning 稍微更快，但中期震盪較大

## 遇到的問題與修正
- Lightning初期 optimizer 錯誤 → 改用手動管理 optimizer
- Random mode收斂慢 → 減慢 epsilon decay 速度

## 小結與改進建議
- 隨機環境下使用 Lightning+穩定技術效果明顯
- 未來可以引入：
  - Distributional DQN
  - Rainbow DQN 綜合型改良
  - NoisyNet 增強 exploration

---
