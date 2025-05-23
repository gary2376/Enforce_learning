---
title: "Grid Navigation 強化學習專案完整指南"
tags: ["Reinforcement Learning", "Value Iteration", "Flask", "Grid Navigation"]
---

# **Grid Navigation 強化學習專案完整指南**
> 透過 **值迭代 (Value Iteration)** 演算法來求解最佳路徑，並以 Flask 及 JavaScript 視覺化顯示最佳策略與狀態值。本專案實作了一個基於 Flask (Python) 的 Value Iteration Grid 系統，主要應用
於 最短路徑尋找 及 決策規劃。該系統透過 強化學習 (Reinforcement
Learning) 的值迭代 (Value Iteration) 演算法，在網格環境中計算最佳行動策
略，並視覺化顯示 狀態價值函數 V(s) 及最佳行動策略 (policy)。

## **📌 1. 專案概述**
### **1.1 目標**
本專案的目標是使用 **強化學習 (Reinforcement Learning)** 的 **值迭代 (Value Iteration)** 方法，讓代理人（Agent）學會在網格 (Grid) 環境中找到最佳路徑，以最小的步數抵達目標點。

### **1.2 應用場景**
- **機器人導航**：機器人可以學習如何在迷宮中找到最佳移動策略。
- **遊戲 AI**：讓 NPC 自主學習如何走向目標。
- **物流與路徑規劃**：如智慧倉儲機器人的最佳行走規劃。
- **智慧交通**：規劃城市道路中的最短通行時間。

## **📚 2. 知識背景**
### **2.1 強化學習概念**
**強化學習 (Reinforcement Learning, RL)** 是機器學習的一個領域，透過與環境的互動來學習最優策略。核心概念：
- **狀態 (State, S)**：當前的環境，如棋盤的位置。
- **行動 (Action, A)**：在狀態 S 下可執行的操作，如上下左右移動。
- **獎勵 (Reward, R)**：執行動作後得到的回饋，如是否接近目標。
- **策略 (Policy, π)**：決定在某個狀態下該如何行動。
- **價值函數 (Value Function, V(s))**：衡量某狀態下的長期獎勵。

### **2.2 值迭代 (Value Iteration)**
值迭代是一種動態規劃 (Dynamic Programming) 方法，透過 **貝爾曼方程 (Bellman Equation)** 不斷更新狀態價值，直到收斂：

\[
V(s) = \max_a \sum_{s'} P(s' | s, a) [R(s, a, s') + \gamma V(s')]
\]

其中：
- \( V(s) \) 是狀態 \( s \) 的價值。
- \( P(s' | s, a) \) 是執行動作 \( a \) 後轉移到狀態 \( s' \) 的機率。
- \( R(s, a, s') \) 是即時獎勵。
- \( \gamma \) 是 **折扣因子 (Discount Factor)**，控制未來獎勵的影響力。

**基於 Value Iteration 的網格最佳路徑求解系統**
## **系統架構**
本系統由 **前端 (HTML + JavaScript)** 和 **後端 (Flask + Python)** 組成。

### **前端 (index.html)**
- 產生 **n × n** 的網格環境 (**範圍 5≤n≤9**)。
- 使用者可點擊設定：
  - **起點 (S)** - 綠色
  - **終點 (E)** - 紅色
  - **障礙物** - 灰色
- 透過 AJAX 呼叫 Flask API (**/solve**) 計算 **最佳策略**。
- 支援 **「逐步前進 / 回退顯示」最佳路徑**。

### **後端 (app.py)**
- 透過 **值迭代演算法 (Value Iteration Algorithm)** 計算：
  - **每個格子的價值函數 V(s)**
  - **最佳行動策略 (policy)**
  - **最佳行進路徑 (best_path)**

---
## **功能詳解**

### **1. 產生 n × n 網格**
- 使用者可選擇 **網格大小 (5 ≤ n ≤ 9)**。
- 每個格子代表一個狀態 (**state**)：
  - **S (Start) 起點** → 綠色
  - **E (End) 終點** → 紅色
  - **障礙物 (Obstacles)** → 灰色
  - **最佳行進路徑 (Best Path)** → 黃色
  - **當前步驟 (Current Step)** → 藍色

### **2. 設定環境**
- **第一個點擊**：設定 **起點 (S)** → 綠色。
- **第二個點擊**：設定 **終點 (E)** → 紅色。
- **其他點擊**：設定 **障礙物**，可設多個。
- **按下 Generate Grid**：清除網格並重新設定。

### **3. 執行 Value Iteration**
- **按下 Solve 按鈕**，前端會發送請求至 Flask 後端 (**/solve**)。
- 後端運行 **Value Iteration** 計算：
  - **每個格子的價值函數 V(s)**。
  - **最佳策略 (policy)**：每個格子該往哪裡移動 (**↑ ↓ ← →**)。
  - **最佳路徑 (best_path)**：**起點 → 終點** 的最短路徑。

### **4. 顯示最佳策略**
- 每個格子會顯示：
  - **最佳行動（方向箭頭 ↑ ↓ ← →）**。
  - **價值函數 V(s)**（顯示為數字）。

### **5. 路徑逐步顯示**
- **按下 Go Forward** → 按一次，顯示下一步，標記為 **藍色 (current-step)**。
- **按下 Go Backward** → 按一次，回到前一步，標記為 **藍色 (current-step)**。
- **所有已走過的步驟仍保持 path (黃色)**，確保視覺化完整。



##### 執行圖片
![image](https://github.com/user-attachments/assets/c0575fe5-0906-405b-af76-e6c98d61f218)

## **🔧 3. 系統架構**
### **3.1 Flask 後端**
後端 Flask 負責計算 Grid 環境的最佳路徑，並提供 API：
- **`/solve`**：接收網格大小、起點、終點、障礙物，回傳最佳路徑與策略。

### **3.2 前端視覺化**
前端透過 JavaScript 及 CSS 建立互動式 Grid，並動態顯示：
- **最佳行動策略**
- **各狀態的價值 (V-value)**
- **最佳移動路徑**

## **💻 4. 操作方式**
### **4.1 設定 Grid 環境**
1. **選擇 Grid 大小** (最小 5x5, 最大 9x9)。
2. **點擊設定**：
   - **第一個點擊：設定起點 (S)**
   - **第二個點擊：設定終點 (E)**
   - **後續點擊：設定障礙物 (黑色格子)**

### **4.2 執行值迭代演算法**
1. 點擊 **「Solve」**，後端會計算最佳策略。
2. 前端更新：
   - 各格子的 **價值 V(s)**
   - **最佳策略箭頭**
   - **標示最佳路徑**

## **📜 5. 程式碼解析**
### **5.1 Flask 後端 (`app.py`)**
#### **(1) 定義 Grid 環境**
```python
gamma = 0.9  # 折扣因子
theta = 0.0001  # 收斂標準
actions = ['U', 'D', 'L', 'R']
action_map = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
```

#### **(2) 值迭代計算**
```python
def value_iteration(n, start, goal, obstacles):
    V = np.zeros((n, n))  # 初始化價值函數
    while True:
        delta = 0
        new_V = V.copy()
        for i in range(n):
            for j in range(n):
                if (i, j) == goal or (i, j) in obstacles:
                    continue
                action_values = []
                for action in actions:
                    di, dj = action_map[action]
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n and (ni, nj) not in obstacles:
                        action_values.append(-1 + gamma * V[ni, nj])
                    else:
                        action_values.append(-1 + gamma * V[i, j])
                new_V[i, j] = max(action_values)
                delta = max(delta, abs(V[i, j] - new_V[i, j]))
        V = new_V.copy()
        if delta < theta:
            break
    return V
```

#### **(3) Flask API**
```python
@app.route('/solve', methods=['POST'])
def solve():
    data = request.json
    V = value_iteration(data['n'], tuple(data['start']), tuple(data['goal']), set(tuple(obstacle) for obstacle in data['obstacles']))
    return jsonify({'V': V.tolist()})
```


## 📊 結果示範

| 格子           | 內容          |
| -------------- | ------------- |
| 🟩             | 起點（S）     |
| 🟥             | 終點（E）     |
| ⬜             | 可行走區域    |
| 🟨             | 走過的區域    |
| ▒             | 障礙物        |
| →, ←, ↑, ↓ | 策略方向      |
| -5.2           | \( V(s) \) 值 |

---

## **🚀 6. 未來擴展**
- **Q-learning / DQN 強化學習方法**
- **動態障礙物場景**
- **3D 環境導航**
