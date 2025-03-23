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
