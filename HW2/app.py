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
