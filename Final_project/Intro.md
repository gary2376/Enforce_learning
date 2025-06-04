## 1. 檔案概要

這份 Notebook 的主要目的在於：

1. **資料前處理**：讀取多支股票的歷史價格資料，並對缺失的交易日進行補值。
2. **分群 (Clustering)**：計算各股票的對數報酬率 (log return)，利用 Dynamic Time Warping (DTW) 建立距離矩陣，再透過 k-medoids 將股票分群。
3. **強化學習 (Reinforcement Learning, RL) 訓練**：為每個 DTW 分群結果建立獨立的股市交易環境（CustomStockTradingEnv），並以 PPO 演算法訓練多支股票的交易策略。
4. **回測示範 (注解)**：透過已訓練模型對單支股票或群組進行回測，計算績效指標（累積報酬、最大回撤、Sharpe Ratio）並繪製績效圖。

---

## 2. 資料讀取 (讀檔)

**程式位置：Cell 0 \~ Cell 1**

```python
#### 讀檔

import os
import pandas as pd

def load_all_stock_data(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            path = os.path.join(folder_path, filename)
            tic = filename.replace("converted_", "").replace(".csv", "")
            df = pd.read_csv(path)
            df['tic'] = tic  # 加上股票代碼欄位
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

# 📂 指定資料夾路徑
folder_path = "E:\\python_project\\class\\Reinforce_Learning\\RL\\code\\converted_stock"
raw_df = load_all_stock_data(folder_path)
```

* **`load_all_stock_data` 函式**：

  1. 掃描指定資料夾下所有 `.csv` 檔案。
  2. 讀取每一檔 CSV，並新增欄位 `tic` 代表該 CSV 對應的股票代號。
  3. 將所有 DataFrame 合併 (concatenate) 成一個大表，回傳 `raw_df`。
* **用途**：快速將「多支股票」分散在不同 CSV 的歷史價格，一次性讀取並合併起來，方便後續統一處理。

---

## 3. 缺失值補值 (補值)

**程式位置：Cell 2 \~ Cell 3**

```python
#### 補值

import pandas as pd
import numpy as np

def interpolate_stock_data(df, start_date="2020-01-01", end_date="2024-10-30"):
    # 建立完整日期範圍（僅工作日）
    full_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' = business day

    result_list = []
    for tic in df['tic'].unique():
        sub_df = df[df['tic'] == tic].copy()
        sub_df['date'] = pd.to_datetime(sub_df['date'])

        # 將 index 設為日期後 reindex，並對缺失記錄進行補值
        sub_df = sub_df.set_index('date').reindex(full_dates)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            # 先用線性插值，再用前向/後向補值
            sub_df[col] = sub_df[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        sub_df['tic'] = tic
        result_list.append(sub_df.reset_index().rename(columns={'index': 'date'}))

    # 合併所有股票的補值結果
    full_df = pd.concat(result_list, ignore_index=True)
    return full_df

# 執行補值
interpolated_df = interpolate_stock_data(raw_df)

# 檢查結果
print(interpolated_df.shape)
print(interpolated_df.head())
```

* **`interpolate_stock_data` 函式**：

  1. 先建立一個從 `start_date` 到 `end_date` 的完整工作日日期索引 `full_dates`。
  2. 對於每支股票 (`tic`)，將其原始 DataFrame 以 `date` 為索引後，呼叫 `reindex(full_dates)`，使得每個工作日都有一筆「空」紀錄。
  3. 使用 `interpolate(method='linear')` 先做線性內插，若還有缺值再分別以 `bfill`、`ffill` 補齊。
  4. 最後把所有子表 `reset_index()` 並合併成一個完整的 `full_df`。
* **用途**：確保每支股票在整個時間區間內（2020-01-01 到 2024-10-30）「每天」都有價格與成交量資料，以利後續計算報酬率或技術指標時不會遇到斷點。

---

## 4. DTW 分群 (DTW 分群)

**程式位置：Cell 4 \~ Cell 9**

1. **計算對數報酬率 (Log Return)**

   ```python
   def calculate_log_return(df):
       df['date'] = pd.to_datetime(df['date'])
       result = {}
       for tic in df['tic'].unique():
           sub_df = df[df['tic'] == tic].copy().set_index('date').sort_index()
           sub_df['log_return'] = np.log(sub_df['close'] / sub_df['close'].shift(1))
           result[tic] = sub_df['log_return'].dropna()
       return pd.DataFrame(result)
   ```

   * **功能**：

     1. 將 `date` 欄轉為 `datetime`，並依 `tic` (股票代號) 分組。
     2. 對每支股票，計算每日 `log_return = ln(close_t / close_{t-1})`。
     3. 回傳一個 DataFrame，欄位為各股票代號，索引為日期，值為對數報酬率。

2. **DTW 分群函式 (`dtw_cluster`)**

   ```python
   def dtw_cluster(log_return_df, k=3):
       series_array = log_return_df.T.values[..., np.newaxis]  # 轉成 shape=(n_stocks, time_steps, 1)
       dist_matrix = cdist_dtw(series_array)                  # 計算 DTW 距離矩陣
       initial_medoids = list(range(k))
       kmedoids_instance = kmedoids(dist_matrix, initial_medoids, data_type='distance_matrix', ccore=False)
       kmedoids_instance.process()
       clusters = kmedoids_instance.get_clusters()

       label_map = {}
       stock_list = list(log_return_df.columns)
       for i, cluster in enumerate(clusters):
           for idx in cluster:
               label_map[stock_list[idx]] = i
       return label_map
   ```

   * **關鍵步驟**：

     1. 將 `log_return_df` 轉為一個 3D 陣列 (`series_array`)，每筆時間序列保持原長度；
     2. 呼叫 `cdist_dtw` 計算所有股票兩兩之間的 DTW 距離，得到距離矩陣；
     3. 以 k-medoids (PAM) 方法 (使用 `pyclustering` 套件) 針對距離矩陣做分群，指定群數 `k`；
     4. 回傳一個 `label_map` 字典，mapping 每支股票代號到其所屬的群組 index。

3. **繪製分群結果 (`plot_clusters`)**

   ```python
   def plot_clusters(log_return_df, cluster_labels):
       grouped_stocks = {}
       for stock, group in cluster_labels.items():
           grouped_stocks.setdefault(group, []).append(stock)

       # 依群繪製該群所有股票的 log return 時序線圖
       for group_id, stocks in grouped_stocks.items():
           plt.figure(figsize=(12, 4))
           for stock in stocks:
               plt.plot(log_return_df.index, log_return_df[stock], label=stock)
           plt.title(f"group {group_id}: 共 {len(stocks)} 支")
           plt.xlabel("Date")
           plt.ylabel("Normalized Log Return")
           plt.legend(loc='upper right')
           plt.grid(True)
           plt.tight_layout()
           plt.show()
   ```

   * **功能**：

     1. 先將 `cluster_labels` 轉為以群組為鍵 (`group_id`) 的字典，值為該群所有股票代號的列表；
     2. 對每個群組，開新圖，將該群內所有股票的 `log_return` 時序畫在同張圖上，方便比較同群成員的走勢相似度。

---

### 4.1 整合流程範例

```python
# 讀取 & 補值
raw_df = load_all_stock_data(folder_path)
interpolated_df = interpolate_stock_data(raw_df)

# 計算 log return，並切出訓練集 (2023-12-31 以前)
log_return_df = calculate_log_return(interpolated_df)
train_log_return_df = log_return_df[log_return_df.index <= '2023-12-31']

# 以 k=4 做 DTW 分群
cluster_labels = dtw_cluster(train_log_return_df, k=4)

# 繪製每個群的 log return 時序
plot_clusters(train_log_return_df, cluster_labels)
```

---

## 5. 強化學習 (train model)

在完成分群後，會以每個群的股票資料作為輸入，訓練各自的 PPO 演算法交易策略。整段程式分為兩大子流程：

1. 建立自訂環境 (CustomStockTradingEnv) 并封裝成可給 Stable-Baselines3 使用的環境；
2. 以 DRLAgent 介面搭配 PPO 演算法，對每個群組逐一訓練並儲存模型。

---

### 5.1 自訂股市交易環境 (環境)

**程式位置：Cell 15 \~ Cell 17**

```python
import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv

class CustomStockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super(CustomStockTradingEnv, self).__init__()
        # --- 原始數據和參數 ---
        self.df = config['df']                # 輸入包含多支股票的 DataFrame (date, tic, open, close, indicators...)
        self.stock_dim = len(config['stock_dim_list'])  # 同一群組可交易的股票數量
        self.tech_indicator_list = config['tech_indicator_list']
        self.initial_amount = config['initial_amount']
        self.buy_cost_pct = config['buy_cost_pct']
        self.sell_cost_pct = config['sell_cost_pct']
        self.reward_scaling = config['reward_scaling']
        self.max_stock = config['max_stock']
        self.min_trade_unit = config['min_trade_unit']
        self.if_train = config['if_train']
        # 其他內部變數初始化 (現金、持股、當前天數、peak asset、歷史回報...等)

        # 建立 action_space、observation_space
        self.action_space = spaces.Box(low = -1, high = 1, shape=(self.stock_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1 + 2*self.stock_dim + len(self.tech_indicator_list)*self.stock_dim,), dtype=np.float32)

    def _calculate_reward(self):
        # 例如：報酬 = 當日總資產變動 * reward_scaling，並加入多項懲罰項 (交易成本、回撤懲罰、持倉成本等)
        pass

    def _get_observation(self):
        # 將現金、持股、當日股價、各技術指標串成一維向量，做為觀測值
        pass

    def reset(self):
        # 將環境重置：現金設為 initial_amount，買賣持股數 = 0，day=0, 重新設定 turbulence 等風險指標，以及初始 obs
        return self._get_observation()

    def step(self, action):
        # 根據 action (多支股票買/賣數量) 計算交易後現金、持股變化，計算該日 reward，更新當前 step info
        # 同時更新最大資產 (peak_asset)、record 回撤等
        return self._get_observation(), reward, done, info

    def render(self, mode='human'):
        # 如需畫圖或列印, 在這裡實作 (視情況開啟或關閉)
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
```

* **主要功能說明**

  1. **環境設定 (`__init__`)**：

     * 接收一個 `config` 字典，其中至少要包含：

       * `df`：一個 DataFrame，包含該群組所有股票的歷史資料 (例如：`date, tic, open, close, high, low, volume, 各技術指標...`)。
       * `stock_dim_list`：該群組內股票代號列表，用來決定可同時交易的股票維度。
       * `tech_indicator_list`：要加入觀測 (Observation) 的技術指標欄位清單。
       * `initial_amount`：代理人初始持有的現金。
       * `buy_cost_pct`, `sell_cost_pct`：買進/賣出時的手續費比例。
       * `reward_scaling`：回報縮放係數。
       * `max_stock`：每支股票可買的最大張數或單位數。
       * `min_trade_unit`：最小交易單位 (例如整股)。
       * `if_train`：布林值，代表目前是否為訓練模式 (決定是否記錄交易紀錄或產生額外噪音等)。

  2. **Action Space**：

     * `Box(-1, 1, shape=(stock_dim,))`，表示對群組中每支股票的「買 (正) / 賣 (負)」比例或數量。實際買賣數量會根據 `max_stock` 與 `min_trade_unit` 進行映射。

  3. **Observation Space**：

     * 一維向量，排列順序大致為：

       ```
       [現金餘額,  
        持股數目_股票1, 持股數目_股票2, ...,  
        當日股價_股票1, 當日股價_股票2, ...,  
        技術指標1_股票1, 技術指標1_股票2, ..., 技術指標N_股票M]
       ```
     * 觀測維度 = 1 (現金) + stock\_dim (持股) + stock\_dim (股價) + stock\_dim × len(tech\_indicator\_list)。

  4. **`_calculate_reward`**：

     * 目前尚未具體實作（留為 TODO），但程式中有註解提到：

       * 當日回報 = (當日總資產 − 前一日總資產) × `reward_scaling`
       * 若出現風險控管需求 (例如回撤超過 10%)，需要額外給予「最大回撤懲罰 (max drawdown penalty)」。
       * 會把交易成本、手續費、持倉成本等都考慮進最終 reward。

  5. **`reset` / `step`**：

     * `reset()`：

       * 重新初始化現金、持股、`day=0`、`peak_asset=initial_amount`，並讀取初始觀測值回傳。
     * `step(action)`：

       * 根據 `action` 對每支股票進行「買 / 賣」操作，計算交易後的持倉變化與現金變動。
       * 更新當日總資產 (`total_asset`)、歷史最高資產 (`peak_asset`)。
       * 若出現回撤 (drawdown) 超過 10%，依公式計算懲罰項 (公式中 `P_drawdown = |(total_asset − peak_asset) / peak_asset| × 1.0 × reward_scaling`)。
       * 最終輸出 `(observation, reward, done, info)`，其中 `done` 表示是否到達最後一天，`info` 可能包含額外細節（如手續費總額、當日交易紀錄等）。

  6. **`get_sb_env`**：

     * 將此自訂環境包裝成 SB3 可使用的向量化環境 (`DummyVecEnv`)，以便直接餵給 Stable-Baselines3 的演算法進行訓練。

---

### 5.2 建立環境函式

**程式位置：Cell 16**

```python
def create_env_for_stock_np(
    df,
    stock_tic,
    indicators,
    initial_amount=1e6,
    if_train=True,
    max_stock=1e4,
    slippage_pct=0.005,  # 滑價 ±0.5%
    min_trade_unit=1     # 最小交易單位（整數）
):
    # --- 篩選出該群組所有股票 ---
    df = df[df['tic'].isin(stock_tic)].copy()
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)

    # 這裡假設外部已經計算好各項技術指標並存在 df 中，若沒有則要先行加入

    env = CustomStockTradingEnv(
        {
            "df": df,
            "stock_dim_list": stock_tic,
            "tech_indicator_list": indicators,
            "if_add_stock_price": True,          # 是否將原始股價當作觀測
            "if_add_tech": True,                 # 是否加入技術指標
            "if_add_turbulence": False,          # 是否加入市場震盪指標
            "risk_indicator_col": "turbulence",  # 若使用風險指標，需指定該欄位名稱
            "initial_amount": initial_amount,
            "buy_cost_pct": 0.001,
            "sell_cost_pct": 0.001,
            "reward_scaling": 1e-4,
            "if_train": if_train,
            "max_stock": max_stock,             # 限制最大持倉
            "min_trade_unit": min_trade_unit    # 整數交易限制
        }
    )

    return env
```

* **功能**：

  1. 根據傳入的 `df` 與 `stock_tic` (該群組股票代號 list)，過濾出本次要訓練的多支股票資料；
  2. 將資料依 `date`、`tic` 排序，保證取值順序正確；
  3. 建立並回傳一個 `CustomStockTradingEnv` 實例。

* **常見參數**：

  * `indicators`：使用的技術指標名稱列表（例如：`['macd', 'rsi_30', 'cci_30', 'dx_30']`）；
  * `initial_amount`：初始資金（預設 1,000,000）；
  * `max_stock`, `slippage_pct`, `min_trade_unit`…等，用來控制交易限制與成本；
  * `if_train`：是否為訓練模式 (影響是否要保存過程資料 / 加入額外噪音等)。

---

### 5.3 分群訓練 (Cluster-wise Training)

**程式位置：Cell 12 \~ Cell 18（基礎範例，實務上可再包成迴圈）**

```python
import os
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import PPO
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
import numpy as np

# 假設已經有 cluster_labels (股票 -> 群組映射)，也準備好 full_df (包含所有股票歷史資料)
indicators = ['macd', 'rsi_30', 'cci_30', 'dx_30']
model_dir = "./trained_models/PPO"

for group_id in set(cluster_labels.values()):
    # 取得該群組所有股票代號
    stocks_in_group = [stock for stock, g in cluster_labels.items() if g == group_id]

    # 建立環境 (訓練用)
    env_train = create_env_for_stock_np(
        df=interpolated_df,
        stock_tic=stocks_in_group,
        indicators=indicators,
        initial_amount=1e6,
        if_train=True,
        max_stock=1e4,
        slippage_pct=0.005,
        min_trade_unit=1
    )
    env_train, _ = env_train.get_sb_env()

    # 使用 DRLAgent 介面初始化 PPO
    agent = DRLAgent(env=env_train)
    model_ppo = agent.get_model("ppo",
                                policy_kwargs = dict(net_arch=[256, 256], activation_fn="tanh"),
                                learning_rate=3e-4,
                                batch_size=256,
                                n_steps=2048,
                                gamma=0.99,
                                ent_coef=0.01,
                                vf_coef=0.5,
                                max_grad_norm=0.5,
                                gae_lambda=0.95,
                                clip_range=0.2)

    # 訓練模型
    timesteps = 1_000_000
    model_ppo.learn(total_timesteps=timesteps)

    # 儲存模型
    save_path = os.path.join(model_dir, f"PPO_group_{group_id}.zip")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_ppo.save(save_path)
    print(f"已儲存：{save_path}")
```

* **重點說明**：

  1. **環境創建**

     * 對於每個 `group_id`（DTW 分群結果），提取該群內所有股票代號 `stocks_in_group`；
     * 呼叫 `create_env_for_stock_np` 建立「訓練」環境 `env_train`，再以 `.get_sb_env()` 轉為 SB3 可用的 `DummyVecEnv`；

  2. **PPO 模型初始化**

     * 使用 FinRL 套件中的 `DRLAgent` 介面：`agent = DRLAgent(env=env_train)`；
     * 以 `agent.get_model("ppo", ...)` 方式定義 PPO 演算法的超參數：

       * `policy_kwargs`：隱藏層結構 (256,256)、激活函式 (tanh)；
       * `learning_rate`, `batch_size`, `n_steps`, `gamma`, `gae_lambda`, `clip_range` 等；
     * `model_ppo.learn(total_timesteps=timesteps)` 進行實際訓練，`timesteps` 可自行調整。

  3. **模型儲存**

     * 訓練完成後，以 `model_ppo.save(save_path)` 存成 `.zip`；
     * 路徑格式為 `./trained_models/PPO/PPO_group_{group_id}.zip`，方便後續載入。

---

## 6. 回測示範 (測試函式，已註解)

**程式位置：Cell 22 \~ Cell 28（均以註解形式存在）**

```python
def test_model_by_tic(cluster_labels, test_tic, df, indicators, model_dir, initial_amount=1e6):
     from stable_baselines3 import PPO
     from stable_baselines3.common.vec_env import DummyVecEnv
     import matplotlib.pyplot as plt
     import numpy as np
     import pandas as pd
     import os
     # 讀取該支股票所屬之群組
     group_id = cluster_labels[test_tic]
     # 載入已訓練之 PPO 模型
     model_path = os.path.join(model_dir, f"PPO_group_{group_id}.zip")
     model_ppo = PPO.load(model_path)

     # 篩選該支股票的後段測試資料 (例如：2024-01-01 以後)
     test_df = df[df['tic'] == test_tic].copy()
     # 計算技術指標、整理格式

     # 建立測試環境
     env_test = create_env_for_stock_np(
         df=test_df,
         stock_tic=[test_tic],
         indicators=indicators,
         initial_amount=initial_amount,
         if_train=False,
         max_stock=1e4,
         slippage_pct=0.005,
         min_trade_unit=1
     )
     env_test, obs = env_test.get_sb_env()

     # 開始回測：以模型預測 action，與環境互動
     done = False
     returns = []
     asset_memory = []

     while not done:
         action, _states = model_ppo.predict(obs)
         obs, rewards, done, info = env_test.step(action)
         # 記錄績效：當前回報、總資產、交易紀錄
         returns.append(rewards[0])
         asset_memory.append(info[0]['total_asset'])

     # 繪製回測績效圖
     dates = test_df['date'].unique()
     plt.figure(figsize=(10, 6))
     plt.plot(dates, asset_memory, label='Portfolio Value')
     plt.xticks(rotation=45)
     plt.title(f"PPO 回測績效：{test_tic}")
     plt.xlabel("Date")
     plt.ylabel("Portfolio Value (USD)")
     plt.legend()
     plt.grid(True)
     plt.tight_layout()
     plt.show()

     # 計算關鍵績效指標
     final_value = asset_memory[-1]
     cumulative_return = (asset_memory[-1] / initial_amount - 1)
     max_drawdown = np.max(np.maximum.accumulate(asset_memory) - asset_memory) / np.max(np.maximum.accumulate(asset_memory))
     sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)

     print(f"📌 Final Portfolio Value: ${final_value:,.2f}")
     print(f"📈 Cumulative Return: {cumulative_return*100:.2f}%")
     print(f"📉 Max Drawdown: {max_drawdown*100:.2f}%")
     print(f"📊 Sharpe Ratio: {sharpe_ratio:.2f}")
     print("Asset history (tail):", asset_memory[-5:])
```

* **說明**：

  1. 函式 `test_model_by_tic` 接收：

     * `cluster_labels`：由 DTW 得到的「股票 → 群組」對照表；
     * `test_tic`：指定要回測的單支股票；
     * `df`：包含歷史資料的 DataFrame；
     * `indicators`：技術指標列表；
     * `model_dir`：儲存模型的資料夾路徑；
     * `initial_amount`：初始資金。
  2. **流程**：

     * 先找出 `test_tic` 所屬的 `group_id`，載入對應的 PPO 模型；
     * 篩選出該支股票的測試期間資料 (例如 2024-01-01 以後)，並計算好技術指標；
     * 建立測試環境 `env_test`，使用 `model_ppo.predict(obs)` 逐步與環境互動；
     * 在回測過程中記錄每日 `returns` (日回報) 及 `asset_memory` (總資產)；
     * 最後繪製「資產凈值時序圖」，並計算：

       * 最終投資組合價值 (`final_value`)
       * 累積報酬 (`cumulative_return`)
       * 最大回撤 (`max_drawdown`)：使用 `np.maximum.accumulate` 計算歷史最高資產減去當前資產，再取最大值。
       * Sharpe Ratio：`mean(returns) / std(returns) * sqrt(252)`。

* **註**：該段程式目前皆為註解，讀者可依需求解除註解並補齊技術指標計算、測試資料處理的細節後使用。

---

## 7. 核心函式與參數索引

以下列出 Notebook 中重要的**自訂函式**以及常見參數，方便快速查閱：

1. **`load_all_stock_data(folder_path)`**

   * **輸入**：`folder_path` (字串)，指向存放多支股票 CSV 的資料夾。
   * **輸出**：`DataFrame`，含有所有 CSV 資料合併，並新增欄位 `tic`。

2. **`interpolate_stock_data(df, start_date, end_date)`**

   * **輸入**：

     * `df`：含多支股票原始資料 (至少要有 `date, open, high, low, close, volume, tic`)。
     * `start_date`, `end_date` (字串，格式 `"YYYY-MM-DD"`)：補值的時間範圍。
   * **輸出**：對所有缺測工作日做補值後的 `DataFrame`。

3. **`calculate_log_return(df)`**

   * **輸入**：經補值完畢、至少含欄位 `date, close, tic` 的 `DataFrame`。
   * **輸出**：一個 DataFrame，索引為日期，欄位為各股票 `tic`，值為每日對數報酬率。

4. **`dtw_cluster(log_return_df, k)`**

   * **輸入**：

     * `log_return_df`：由 `calculate_log_return` 回傳的對數報酬率表 (日期 × 股票)。
     * `k`：要分成幾群 (整數)。
   * **流程**：呼叫 `cdist_dtw` 計算 DTW 距離，再以 `kmedoids` 做分群。
   * **輸出**：`label_map` (字典)，鍵為 `tic`，值為所屬群組編號 (從 0 開始)。

5. **`plot_clusters(log_return_df, cluster_labels)`**

   * **輸入**：

     * `log_return_df`：對數報酬率 DataFrame。
     * `cluster_labels`：由 `dtw_cluster` 回傳的「股票 → 群組」字典。
   * **功能**：對每個群組畫出該群所有股票的對數報酬走勢。

6. **`CustomStockTradingEnv(config)`**

   * **輸入**：`config` (字典)，內容至少包含：

     * `"df"`：該群組所有股票的歷史資料 (含技術指標等)。
     * `"stock_dim_list"`：該群組內所有股票代號 (list)。
     * `"tech_indicator_list"`：要當作觀測特徵的技術指標名稱 (list)。
     * `"initial_amount"`：初始資金 (數值)。
     * `"buy_cost_pct"`, `"sell_cost_pct"`：買賣手續費比例。
     * `"reward_scaling"`：回報縮放係數。
     * `"max_stock"`：最大持股單位數。
     * `"min_trade_unit"`：最小交易單位 (整股)。
     * `"if_train"`：是否為訓練模式 (布林值)。
   * **功能**：建立一個可供 Stable-Baselines3 使用的多支股票交易環境。

7. **`create_env_for_stock_np(...)`**

   * **輸入**：

     * `df`：完整合併後的所有股票 DataFrame。
     * `stock_tic`：該次要訓練或測試的股票代號列表 (list)。
     * `indicators`：要加入觀測的技術指標清單 (list of str)。
     * 其餘參數：`initial_amount`, `if_train`, `max_stock`, `slippage_pct`, `min_trade_unit` 等。
   * **輸出**：剛包裝成 `CustomStockTradingEnv` 並返回。

8. **PPO 訓練相關**

   * **初始化範例** (透過 FinRL):

     ```python
     agent = DRLAgent(env=env_train)
     model_ppo = agent.get_model(
         "ppo",
         policy_kwargs = dict(net_arch=[256, 256], activation_fn="tanh"),
         learning_rate=3e-4,
         batch_size=256,
         n_steps=2048,
         gamma=0.99,
         ent_coef=0.01,
         vf_coef=0.5,
         max_grad_norm=0.5,
         gae_lambda=0.95,
         clip_range=0.2
     )
     model_ppo.learn(total_timesteps=1_000_000)
     model_ppo.save("PPO_group_{group_id}.zip")
     ```

---

## 8. 執行與注意事項

1. **資料格式要求**

   * 所有輸入 CSV 需至少包含：`date, open, high, low, close, volume`。
   * 建議在合併後的 `interpolated_df` 中，再額外計算各項技術指標 (如 MACD、RSI、CCI 等)，並將其併入 `df`。
   * `date` 欄位需為可轉為 `datetime` 的格式 (如 `"YYYY-MM-DD"`）；`tic` 欄位為股票代號。

2. **分群參數選擇**

   * `k` （群數）可依不同市場或策略需求調整，譬如常見取 3、4、5 等；
   * 若分群結果過於擁擠或稀疏，可先用 PCA、t-SNE 等方法做維度縮減後，再討論群數。

3. **環境與回報設計**

   * `CustomStockTradingEnv` 的 `_calculate_reward` 須依自身風險偏好做調整：

     * 是否要加入「最大回撤懲罰 (Max Drawdown Penalty)」、**多因子回報** (如夏普比率) 等；
     * 若要加入「滑價」或「影子成本 (Slippage)」，可進一步調整 `slippage_pct` 參數；
     * 技術指標是否要標準化 (Normalization) 或對數化 (Log)，視策略做法而定。

4. **訓練時間與硬體資源**

   * 每個群訓練至 `1e6` timesteps 以上，若股票數量眾多，建議分批次或多 GPU/多進程並行；
   * 訓練過程中可開啟 `tensorboard` 監控訓練曲線 (loss, reward) 變化，方便調參。

5. **回測注意**

   * 註解掉的 `test_model_by_tic` 需要自行補齊「技術指標計算」與「測試資料切分」部分；
   * 回測結果需與真實市場走勢對齊，若測試期間出現大波動 (2022-2023 疫情、Fed 干預等)，需特別標註。

---

## 9. 總結

整份 Notebook 的主要工作流程可以概括為：

1. **讀檔 → 補值**：將多支股票歷史資料一次性讀入並補齊缺失日期。
2. **計算對數報酬率 → DTW 分群 → 畫圖檢視**：理解不同股票間走勢相似度，將其同群。
3. **自訂股市環境 → 每群訓練 PPO**：針對每個群組建立多股交易環境，以 PPO 演算法訓練投資策略並儲存模型。
4. **(備選) 單支回測範例**：載入訓練好的模型，對單支股票做回測，計算績效指標並繪圖。

透過這樣的流程，可以針對「同類性比較高」的股票群組，訓練更專精的 RL 策略，而非一次針對全部股票混雜訓練，進而提升演算法在不同類股或產業輪動期間的穩定性與獲利表現。

---

