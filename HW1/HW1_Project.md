# 🏗 HW1: 網格地圖開發與價值函數評估 (Value Function Evaluation)

## 📌 專案目的

本專案的目標是開發一個基於 Flask 的網格地圖應用，讓使用者能夠互動式地設置起點、終點、障礙物，並利用 **價值函數評估 (Value Function Evaluation)** 方法計算每個格子的價值函數 \( V(s) \) 及對應策略。

本專案包含兩個主要部分：

1. **HW1-1: 網格地圖開發**

   - 使用者可設定網格大小 \( n 	imes n \)（\( 5 \leq n \leq 9 \)）。
   - 點擊設定 **起始點（綠色）**、**終點（紅色）**、**障礙物（灰色）**。
2. **HW1-2: 策略顯示與價值函數評估**

   - 針對每個格子，顯示隨機策略（上下左右）。
   - 計算價值函數 \( V(s) \) 來評估各格子的重要性。

---

## 🚀 專案執行流程

1. **伺服器啟動**

   - 啟動 Flask 伺服器，載入 `index.html` 作為前端界面。
2. **使用者互動**

   - 使用者可選擇網格大小 \( n \) 並點擊 **「Generate Grid」** 生成網格。
   - 透過滑鼠點擊設定 **起點**（S）、**終點**（E）、**障礙物**。
   - 點擊 **「Solve」** 來計算策略與價值函數。
3. **伺服器計算**

   - 生成隨機策略。
   - 進行 **價值函數評估 (Value Function Evaluation)** 來計算 \( V(s) \)。
   - 回傳策略箭頭與價值函數結果。
4. **網頁顯示結果**

   - 每個格子內部顯示 **策略方向**（箭頭）及 **數值 \( V(s) \)**。

##### 執行圖片
![image](https://github.com/user-attachments/assets/2fd6f948-b353-4e60-9319-d47bde676f35)
![image](https://github.com/user-attachments/assets/bf8a5d2a-86ea-4038-b70c-68a20b5f2f79)



---

## 🏗 背景與理論基礎

### 🎯 **強化學習與價值函數**

本專案採用 **價值函數 (Value Function)** 的概念來評估每個狀態的價值。價值函數 \( V(s) \) 表示從某個狀態 \( s \) 開始，依照當前策略執行動作時，期望獲得的累積報酬。

價值函數的更新方式如下：
\(
V(s) = R(s) + \gamma \sum P(s'|s, a) V(s')
\)
其中：

- \( R(s) \) 為當前狀態的即時回報 (reward)。
- \( \gamma \) 是折扣因子 (discount factor)。
- \( P(s'|s, a) \) 是從狀態 \( s \) 採取動作 \( a \) 轉移到新狀態 \( s' \) 的機率。
- \( V(s') \) 是新狀態的價值。

### 🤖 **動態規劃與策略評估**

我們使用 **動態規劃 (Dynamic Programming)** 方法來評估策略的好壞，透過 **價值函數評估 (Value Function Evaluation)** 來收斂至最優值。

---

## 📝 程式碼結構

本專案主要由 Flask 伺服器 (`app.py`) 及前端 (`index.html`) 組成。

### 🔹 Flask 伺服器 (`app.py`)

負責計算策略與價值函數：

```python
# 初始化 Flask
app = Flask(__name__)

# 主要 API
@app.route('/solve', methods=['POST'])
def solve():
    data = request.json
    n = data['n']
    start = tuple(data['start'])
    goal = tuple(data['goal'])
    obstacles = set(tuple(obstacle) for obstacle in data['obstacles'])

    # 計算 V(s) 與策略
    V, arrow_policy = policy_evaluation(n, start, goal, obstacles)

    return jsonify({'V': V, 'policy': arrow_policy.tolist()})
```

### 🔹 價值函數評估 (`policy_evaluation`)

```python
def policy_evaluation(n, start, goal, obstacles, policy):
    V = np.zeros((n, n))
    V[goal[0], goal[1]] = 0  # 終點價值 0

    while True:
        delta = 0
        new_V = V.copy()
        for i in range(n):
            for j in range(n):
                if (i, j) == goal or (i, j) in obstacles:
                    continue
                action = policy[i, j]
                di, dj = action_map[action]
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n and (ni, nj) not in obstacles:
                    new_V[i, j] = -1 + gamma * V[ni, nj]
                else:
                    new_V[i, j] = -1 + gamma * V[i, j]
                delta = max(delta, abs(V[i, j] - new_V[i, j]))
        V = new_V
        if delta < theta:
            break
    return np.round(V, 1).tolist()
```

---

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

## 🏆 結論

本專案透過 **Flask + JavaScript** 搭建了一個網格環境，並利用 **價值函數評估 (Value Function Evaluation)** 方法計算價值函數 \( V(s) \)。此系統可用於強化學習 (Reinforcement Learning) 的基礎應用，未來可以擴展至 **動態規劃 (Dynamic Programming)** 或 **最優策略學習 (Optimal Policy Learning)**。

🔥 **下一步優化**：

- **動態策略改進**（策略迭代）。
- **加入隨機動作機率**（馬可夫決策過程 MDP）。
- **改進 UI 顯示，增加動畫效果**。
- **加入 Q-Learning 或 SARSA** 以探索更進階的學習方法。

---

Chatgpt：https://chatgpt.com/share/67daa495-87fc-8010-bf81-92a53a1b9ebf
https://chatgpt.com/share/67dacbe3-fb58-8010-b2ef-92d32ddd1e1c


#### 以下為完整的code:

##### app.py

```python
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

# 策略評估方法
def policy_evaluation(n, start, goal, obstacles, policy):
    # 初始化V(s)
    V = np.zeros((n, n))
    V[goal[0], goal[1]] = 0  # 終點格子的V值為0

    while True:
        delta = 0
        new_V = V.copy()  # 複製一個新的V值來進行計算

        # 遍歷每個格子進行策略評估
        for i in range(n):
            for j in range(n):
                if (i, j) == goal or (i, j) in obstacles:
                    continue  # 目標點和障礙物不更新

                # 根據當前策略選擇動作
                action = policy[i, j]
                di, dj = action_map[action]
                ni, nj = i + di, j + dj

                # 如果移動有效，計算V值
                if 0 <= ni < n and 0 <= nj < n and (ni, nj) not in obstacles:
                    new_V[i, j] = -1 + gamma * V[ni, nj]  # 每一步的獎勳為-1
                else:
                    new_V[i, j] = -1 + gamma * V[i, j]  # 無效移動保持自身值

                # 計算變化量
                delta = max(delta, abs(V[i, j] - new_V[i, j]))  # 記錄最大變化

        V = new_V.copy()  # 更新V(s)

        # 若最大變化小於閾值，則終止迭代
        if delta < theta:
            break

    # 將V值保留一位小數
    V = np.round(V, 1)

    return V.tolist()

# 進行策略評估，不重新計算策略
def policy_iteration_with_evaluation(n, start, goal, obstacles):
    # 初始化V(s)為0
    V = np.zeros((n, n))
    V[goal[0], goal[1]] = 0  # 終點V值為0
    policy = np.full((n, n), '', dtype=object)  # 儲存初始策略

    # 初始隨機策略（隨便設一個）
    for i in range(n):
        for j in range(n):
            if (i, j) != goal and (i, j) not in obstacles:
                policy[i, j] = actions[np.random.choice(len(actions))]

    # 進行策略評估
    V = policy_evaluation(n, start, goal, obstacles, policy)

    # 將策略中的每個動作轉換為箭頭
    arrow_policy = np.vectorize(lambda x: arrow_map.get(x, ''))(policy)

    return V, arrow_policy

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

    # 計算每個格子的V值和策略（這裡策略不會變化，僅僅是策略評估）
    V, arrow_policy = policy_iteration_with_evaluation(n, start, goal, obstacles)

    return jsonify({
        'V': V,
        'policy': arrow_policy.tolist()  # 返回轉換為箭頭的策略
    })

if __name__ == '__main__':
    app.run(debug=True)

```

##### index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Value Iteration Grid</title>
    <style>
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, 50px);
            gap: 2px;
            margin: 20px auto;
        }
        .grid div {
            width: 50px;
            height: 50px;
            border: 1px solid #ccc;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            cursor: pointer;
        }
        .start { background-color: green; }
        .end { background-color: red; }
        .obstacle { background-color: gray; }
    </style>
</head>
<body>
    <h1>Value Iteration Grid</h1>
    <div>
        <label for="n">Grid Size (n x n): </label>
        <input type="number" id="n" value="5" min="5" max="9">
    </div>
    <div class="controls">
        <button onclick="generateGrid()">Generate Grid</button>
        <button onclick="solve()">Solve</button>
    </div>
    <div id="grid" class="grid"></div>

    <script>
        let gridData = [];
        let start = null;
        let end = null;
        let obstacles = [];
        let V = [];
        let policy = [];

        function generateGrid() {
            const n = parseInt(document.getElementById('n').value);
            gridData = [];
            obstacles = [];
            start = null;
            end = null;
            const grid = document.getElementById('grid');
            grid.innerHTML = '';
            for (let i = 0; i < n; i++) {
                const row = [];
                for (let j = 0; j < n; j++) {
                    const cell = document.createElement('div');
                    cell.dataset.x = i;
                    cell.dataset.y = j;
                    cell.onclick = () => onGridClick(cell, i, j);
                    grid.appendChild(cell);
                    row.push(cell);
                }
                gridData.push(row);
            }
            grid.style.gridTemplateColumns = `repeat(${n}, 50px)`;
        }

        function onGridClick(cell, i, j) {
            if (!start) {
                start = [i, j];
                cell.classList.add('start');
                cell.innerText = 'S';
            } else if (!end) {
                end = [i, j];
                cell.classList.add('end');
                cell.innerText = 'E';
            } else {
                if (obstacles.length < gridData.length - 2) {
                    obstacles.push([i, j]);
                    cell.classList.add('obstacle');
                }
            }
        }

        function solve() {
            const n = gridData.length;
            fetch('/solve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    n,
                    start,
                    goal: end,
                    obstacles
                })
            })
            .then(response => response.json())
            .then(data => {
                V = data.V;
                policy = data.policy;
                displayPolicyAndV();
            });
        }

        function displayPolicyAndV() {
            const n = gridData.length;
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    const cell = gridData[i][j];
                    if (cell.classList.contains('start') || cell.classList.contains('end') || cell.classList.contains('obstacle')) {
                        continue;
                    }
                    const policyArrow = policy[i][j] ? policy[i][j] : '';
                    cell.innerText = `${policyArrow}\n${V[i][j]}`;
                }
            }
        }
    </script>
</body>
</html>

```



---
