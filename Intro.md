
# FinRL 股票交易強化學習實踐 - 投資組合分配

專案 Notebook 連結: [FinRL Portfolio Allocation](https://colab.research.google.com/github/AI4Finance-LLC/FinRL-Library/blob/master/FinRL_portfolio_allocation.ipynb)

## 1. 專案概述

本 Notebook 旨在演示如何利用 FinRL 庫，一個專為量化金融而設計的深度強化學習 (DRL) 庫，來構建一個自動化的股票交易系統。該系統專注於投資組合分配問題，即如何在不同的股票之間分配資金以最大化收益。

### 1.1. 背景

傳統的金融市場分析和交易決策通常依賴於複雜的手工分析、統計模型和人為判斷。然而，DRL 提供了一種通過智能體與環境的互動學習最優策略的強大方法。FinRL 庫簡化了 DRL 在金融領域的應用，使研究人員和從業者能夠更輕鬆地開發和評估自動交易系統。

### 1.2. 目標

本專案的核心目標是：

- 使用 DRL 演算法訓練一個交易智能體，使其能夠學習有效的股票投資組合分配策略。
- 利用歷史股票市場資料來模擬交易環境，並對智能體的表現進行回測。
- 展示 FinRL 庫在構建和評估自動交易系統方面的能力。

## 2. 問題定義

### 2.1. 馬可夫決策過程 (MDP)

股票交易過程被建模為一個 MDP，它由以下要素組成：

- **狀態 (State)**: 智能體在每個時間步觀察到的市場資訊。
- **動作 (Action)**: 智能體採取的交易決策，例如買入、賣出或持有一定數量的股票。
- **獎勳 (Reward)**: 智能體根據其行動從環境中獲得的激勵，通常與投資組合價值的變化相關。
- **環境 (Environment)**: 股票市場，提供股票資料並響應智能體的交易行為。

### 2.2. 狀態空間

狀態空間描述了智能體用來做出交易決策的資訊。在本 Notebook 中，狀態空間包括：

- 股票價格: 開盤價、最高價、最低價、收盤價。
- 交易量: 成交量。
- 技術指標: 移動平均收斂散度 (MACD)、相對強弱指數 (RSI)、商品通道指數 (CCI)、動向指數 (DX)。
- 共變異數矩陣: 反映股票之間價格波動相關性的矩陣，用於衡量風險。

### 2.3. 動作空間

動作空間定義了智能體可以採取的交易行為。這裡使用連續動作空間，其中每個動作代表投資組合中每支股票的權重。權重被標準化，使其總和為 1，表示資金在不同股票之間的分配比例。

### 2.4. 獎勳函數

獎勳函數是 DRL 的關鍵組成部分，它引導智能體學習。在本 Notebook 中，獎勳被定義為投資組合價值的增長。智能體通過最大化累積獎勳來學習最大化投資組合的長期收益。

## 3. 準備工作

### 3.1. 安裝 FinRL 及其依賴

```bash
!pip install git+https://github.com/AI4Finance-LLC/FinRL-Library.git
```

這行程式碼使用 pip 命令從 GitHub 倉庫安裝 FinRL 庫。

### 3.2. 導入 Python 庫

```python
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import datetime

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_portfolio import StockPortfolioEnv

from finrl.model.models import DRLAgent
from finrl.trade.backtest import BackTestStats, BaselineStats, BackTestPlot
from finrl.trade.backtest import backtest_strat, baseline_strat
```

這段程式碼導入了必要的 Python 庫，包括：

- pandas: 用於資料處理。
- numpy: 用於數值計算。
- matplotlib: 用於資料視覺化。
- yfinance: 用於獲取股票資料。
- finrl: FinRL 庫，提供金融強化學習相關的功能。

### 3.3. 建立資料夾

```python
import os
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)
```

這段程式碼建立了用於儲存資料、訓練好的模型、TensorBoard 日誌和結果的資料夾。

## 4. 下載資料

```python
df = YahooDownloader(start_date = '2008-01-01',
                     end_date = '2021-01-01',
                     ticker_list = config.DOW_30_TICKER).fetch_data()
```

這段程式碼使用 FinRL 的 YahooDownloader 類從 Yahoo Finance API 下載 Dow 30 指數成分股在 2008 年 1 月 1 日到 2021 年 1 月 1 日期間的歷史資料。`config.DOW_30_TICKER` 是一個包含 Dow 30 成分股股票代碼的列表。

## 5. 資料前處理

### 5.1. 技術指標

```python
fe = FeatureEngineer(
                    use_technical_indicator=True,
                    use_turbulence=False,
                    user_defined_feature = False)

df = fe.preprocess_data(df)
這段程式碼使用 FinRL 的 FeatureEngineer 類計算技術指標。use_technical_indicator=True 表示計算 MACD、RSI、CCI 和 DX 等技術指標。use_turbulence=False 表示不計算金融擾動指數，該指數用於控制風險。
```
### 5.2. 共變異數矩陣
```python
df=df.sort_values(['date','tic'],ignore_index=True)
df.index = df.date.factorize()[0]

cov_list = []
lookback=252
for i in range(lookback,len(df.index.unique())):
  data_lookback = df.loc[i-lookback:i,:]
  price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
  return_lookback = price_lookback.pct_change().dropna()
  covs = return_lookback.cov().values 
  cov_list.append(covs)
  
df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list})
df = df.merge(df_cov, on='date')
df = df.sort_values(['date','tic']).reset_index(drop=True)
```
這段程式碼計算股票報酬率的共變異數矩陣。它首先對資料進行排序，然後使用一個滑動窗口 (lookback=252，表示一年) 計算每天的共變異數矩陣。共變異數矩陣反映了股票報酬率之間的相關性，是投資組合風險管理的重要因素。

## 6. 設計環境
```python
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv


class StockPortfolioEnv(gym.Env):
    """A single stock trading environment for OpenAI gym
    ...
    """

    def __init__(self, 
                df,
                stock_dim,
                hmax,
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                turbulence_threshold=None,
                lookback=252,
                day = 0):
        ...

    def step(self, actions):
        ...

    def reset(self):
        ...
    
    def render(self, mode='human'):
        ...
        
    def save_asset_memory(self):
        ...

    def save_action_memory(self):
        ...

```
這段程式碼定義了一個名為 StockPortfolioEnv 的自定義交易環境，它繼承自 OpenAI Gym 的 gym.Env 類。這個環境模擬了股票市場的運作，並允許智能體與市場進行互動。
- __init__: 建構函式，初始化環境的各種參數，例如：
    - df: 輸入資料。
    - stock_dim: 股票數量。
    - hmax: 最大交易股票數量。
    - initial_amount: 初始資金。
    - transaction_cost_pct: 交易成本百分比。
    - reward_scaling: 獎勳縮放因子。
    - state_space: 狀態空間的維度。
    - action_space: 動作空間的維度。
    - tech_indicator_list: 技術指標列表。
- step: 執行智能體的動作，並返回下一步的狀態、獎勳、是否結束等資訊。
- reset: 重置環境到初始狀態。
- render: 用於視覺化環境狀態 (此處返回狀態)。
- save_asset_memory: 保存每一步的資產價值。
- save_action_memory: 保存每一步的動作 (投資組合權重)。

## 7. 訓練 DRL 演算法
```python
agent = DRLAgent(env = env_train)

A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
model_a2c = agent.get_model(model_name="a2c", model_kwargs = A2C_PARAMS)

trained_a2c = agent.train_model(model=model_a2c, 
                                 tb_log_name='a2c',
                                 total_timesteps=50000)
```
這段程式碼使用 FinRL 提供的 DRL 演算法訓練智能體。
- agent = DRLAgent(env = env_train): 建立一個 DRL 代理，並傳入訓練環境。
- A2C_PARAMS = ...: 定義 A2C 演算法的超參數。
- model_a2c = agent.get_model(...): 建立 A2C 模型。
- trained_a2c = agent.train_model(...): 訓練 A2C 模型。

## 8. 回測
```python
e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)
env_trade, obs_trade = e_trade_gym.get_sb_env()

df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_a2c,
                        test_data = trade,
                        test_env = env_trade,
                        test_obs = obs_trade)
```
這段程式碼使用訓練好的模型在交易測試集上進行回測。DRL_prediction 函數返回每日報酬率和交易操作。

## 9. 回測結果分析
```python
from pyfolio import timeseries

DRL_strat = backtest_strat(df_daily_return)
perf_func = timeseries.perf_stats 
perf_stats_all = perf_func( returns=DRL_strat, 
                              factor_returns=DRL_strat, 
                                positions=None, transactions=None, turnover_denom="AGB")

print("==============DRL Strategy Stats===========")
perf_stats_all

print("==============Get Index Stats===========")
baesline_perf_stats=BaselineStats('^DJI',
                                  baseline_start = '2019-01-01',
                                  baseline_end = '2021-01-01')
```
這段程式碼使用 pyfolio 庫分析回測結果，並與基準策略 (Dow Jones Industrial Average, ^DJI) 進行比較。

## 10. 總結和改進方向
這份 Notebook 提供了一個使用 FinRL 庫構建股票投資組合分配系統的起點。為了滿足您的作業要求，您需要進一步擴展這份 Notebook，例如：
- 更深入的文獻研究: 在 NotebookLm 或 SciSpace 中搜尋並總結 2020 年之後關於 DRL 在股票交易中的最新研究，重點關注投資組合分配領域。
- 模型比較和分析: 嘗試不同的 DRL 演算法 (PPO, DDPG, SAC 等) 並比較它們的表現。分析不同演算法的優缺點，以及它們在不同市場條件下的適用性。
- 特徵工程的探索: 嘗試添加或修改輸入特徵，例如更多的技術指標、新聞情緒、總體經濟資料等，並分析它們對模型性能的影響。
- 風險管理策略: 加入更複雜的風險管理策略，例如動態調整倉位、停損策略等。
- 更全面的回測分析: 使用 pyfolio 庫進行更深入的回測分析，例如風險因子分析、持倉分析等。
- 論文展示準備: 基於您的研究和程式碼結果，準備一個清晰、有條理的論文展示，強調 DRL 在股票交易中的應用和潛在價值。


---
# 股票投資組合回測結果報告

## 1. 簡介

這份報告基於一個股票投資組合的回測結果，展示了策略在不同市場條件下的表現。回測圖表顯示了投資組合的總體回報、波動性、夏普比率等多個重要財務指標，並與基準指數進行了比較。以下是詳細的分析。

## 2. 回測結果概覽
![image](https://github.com/user-attachments/assets/39d841d3-b713-4cd9-8bb5-a6fd861749e0)


回測結果顯示，策略在不同時間範圍內的表現有顯著的波動。圖表顯示了**None**（沒有使用策略）和**Backtest**（使用策略）之間的回報差異。

### 2.1. 投資組合回報與基準比較

上面的圖顯示了投資組合的回報與基準回報的比較。**Backtest**（綠線）顯示了基於強化學習模型的回報，而**None**（灰線）代表沒有策略的回報。

- 從圖中可以看出，回測策略（Backtest）在大部分期間表現優於基準（None），但是在某些時期（如市場崩盤時），策略表現的波動性較大。

### 2.2. 投資組合波動性
![image](https://github.com/user-attachments/assets/12a0a928-afb2-4581-816c-6e30a826dc78)

波動性圖表展示了投資組合的風險（上下波動）的情況。波動性越高，表示投資回報的變動性越大。

- 在回測期間，我們可以看到策略所承擔的風險波動有時比基準指數（Benchmark volatility）高，但整體而言，策略在長期持有下表現較為穩定。

## 3. 重要指標分析

### 3.1. 夏普比率
![image](https://github.com/user-attachments/assets/36ff37de-fbe8-4772-baf3-157a830ecf71)

夏普比率圖顯示了每單位風險所獲得的回報。這是評估回報與風險之間關係的常用指標。

- 從圖表中可以看出，**Backtest** 的夏普比率在大部分時間內超過基準（None）。然而，某些極端的市場波動期間（如崩盤時期），策略的夏普比率大幅下滑。

### 3.2. 投資組合回報視覺化
![image](https://github.com/user-attachments/assets/03958133-d946-4ab8-bace-aea9a97cfe84)

這些圖表展示了在不同時間段內，策略的回報情況。您可以看到回報的波動，並根據市場的不同階段，分析投資組合的表現。

- 圖中呈現的區域顏色區分了不同的回報區間，這使得視覺上可以清晰地識別哪些時期策略表現較好，哪些時期表現較差。

## 4. 回測結果統計

### 4.1. 統計量圖表
![image](https://github.com/user-attachments/assets/72d8faa3-c231-404e-a936-b1e5af6e5b2c)

這些統計量圖表顯示了各項回報指標的基本統計數據，如中位數、均值、最小值和最大值。

- 從這些統計量來看，回測策略的表現均勻分佈，並且在回報的上下波動性方面，基準指數的表現更加一致，顯示出策略在某些時期的高度波動。

### 4.2. 箱型圖
![image](https://github.com/user-attachments/assets/e82bc83d-257f-49b7-8ff1-c1e2f6264a1e)

箱型圖進一步展示了回報的分佈情況，幫助分析投資組合在不同期間內的回報分佈範圍。

- 箱型圖顯示了**Backtest** 投資組合回報的中位數高於基準，顯示該策略在大部分時期內的回報較為穩定。

## 5. 總結與建議

回測結果顯示，該強化學習策略在整體上優於基準，但在極端市場條件下（例如市場崩盤時），策略表現有較大的波動性。這表明該策略在常規市場情況下具有良好的回報潛力，但在極端波動時期可能需要進行風險管理。

### 建議：

- **風險控制**: 在策略中加入風險控制措施，例如動態調整倉位、設定停損策略等，可能有助於減少極端市場條件下的損失。
- **策略改進**: 探索其他強化學習演算法或調整超參數，以改善在極端市場環境下的表現。

這些分析可以幫助進一步優化交易策略，使其在不同市場條件下均能保持穩定的表現。
