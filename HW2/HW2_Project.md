---
title: "Grid Navigation 強化學習專案完整指南"
tags: ["Reinforcement Learning", "Value Iteration", "Flask", "Grid Navigation"]
---

# **Grid Navigation 強化學習專案完整指南**
> 透過 **值迭代 (Value Iteration)** 演算法來求解最佳路徑，並以 Flask 及 JavaScript 視覺化顯示最佳策略與狀態值。

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

## **🚀 6. 未來擴展**
- **Q-learning / DQN 強化學習方法**
- **動態障礙物場景**
- **3D 環境導航**

---
## 完整code
### app.py
'''python

from flask import Flask, render_template, jsonify, request
import numpy as np

app = Flask(__name__)

# 配置參數
gamma = 0.9  # 折扣因子
theta = 0.0001  # 收斂標準
actions = ['U', 'D', 'L', 'R']
action_map = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

# 方向箭頭對應
arrow_map = {
    'U': '↑',
    'D': '↓',
    'L': '←',
    'R': '→'
}

# Value Iteration 方法
def value_iteration(n, start, goal, obstacles):
    # 初始化每個格子的V(s)為 0
    V = np.zeros((n, n))
    V[goal[0], goal[1]] = 0  # 終點格子的V值為0
    policy = np.full((n, n), '', dtype=object)  # 儲存最佳策略

    while True:
        delta = 0
        new_V = V.copy()  # 複製一個新的V值來進行計算

        # 遍歷每個格子進行值迭代
        for i in range(n):
            for j in range(n):
                if (i, j) == goal or (i, j) in obstacles:
                    continue  # 目標點和障礙物不更新

                # 儲存每個行動的預期價值
                action_values = []

                for action in actions:
                    di, dj = action_map[action]
                    ni, nj = i + di, j + dj

                    # 如果移動有效，計算V值
                    if 0 <= ni < n and 0 <= nj < n and (ni, nj) not in obstacles:
                        action_values.append(-1 + gamma * V[ni, nj])  # 每一步的獎勳為-1
                    else:
                        action_values.append(-1 + gamma * V[i, j])  # 無效移動應該保持自身值的影響

                # 使用最大值更新V(s)
                new_V[i, j] = max(action_values)
                # 儲存最佳策略
                policy[i, j] = actions[np.argmax(action_values)]

                # 計算變化量
                delta = max(delta, abs(V[i, j] - new_V[i, j]))  # 記錄最大變化

        V = new_V.copy()  # 更新V(s)

        # 若最大變化小於閾值，則終止迭代
        if delta < theta:
            break

    # 計算最佳路徑
    best_path = find_best_path(n, start, goal, policy)

    # 將V值保留一位小數
    V = np.round(V, 1)

    # 將最佳策略轉換為箭頭
    policy_arrows = np.vectorize(lambda x: arrow_map.get(x, ''))(policy)

    return V.tolist(), policy_arrows.tolist(), best_path

def find_best_path(n, start, goal, policy):
    path = []
    current = start
    while current != goal:
        path.append(current)
        action = policy[current[0], current[1]]
        di, dj = action_map[action]
        current = (current[0] + di, current[1] + dj)
    path.append(goal)  # Add goal to the path
    return path

# 前端顯示頁面
@app.route('/')
def index():
    return render_template('index.html')

# 計算並返回結果
@app.route('/solve', methods=['POST'])
def solve():
    data = request.json
    n = data['n']
    start = tuple(data['start'])
    goal = tuple(data['goal'])
    obstacles = set(tuple(obstacle) for obstacle in data['obstacles'])

    # 計算每個格子的V值和最佳策略，並獲得最佳路徑
    V, policy, best_path = value_iteration(n, start, goal, obstacles)

    return jsonify({
        'V': V,
        'policy': policy,
        'best_path': best_path  # 返回最佳路徑
    })

if __name__ == '__main__':
    app.run(debug=True)

'''
