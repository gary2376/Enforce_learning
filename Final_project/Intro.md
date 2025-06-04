# Jupyter Notebook: Aktienhandelsstrategie mit PPO und DTW-Clustering

## Projektübersicht
Dieses Jupyter Notebook implementiert eine Aktienhandelsstrategie unter Verwendung von Reinforcement Learning (RL), speziell dem Proximal Policy Optimization (PPO) Algorithmus. Ein wesentliches Merkmal ist der Einsatz von Dynamic Time Warping (DTW) zum Clustering von Aktien basierend auf der Ähnlichkeit ihrer Log-Return-Zeitreihen. Für jedes Cluster wird anschließend ein separates PPO-Modell trainiert.

## Workflow des Notebooks

### 1. Datenverarbeitung und -vorbereitung
   - **Laden der Daten (`load_all_stock_data`):**
     - Liest mehrere CSV-Dateien mit Aktiendaten aus einem spezifizierten Ordner.
     - Fügt jeder Aktie eine `tic`-Spalte (Ticker-Symbol) hinzu.
     - Kombiniert alle Daten in einem einzigen Pandas DataFrame.
   - **Interpolation fehlender Werte (`interpolate_stock_data`):**
     - Stellt sicher, dass für einen definierten Zeitraum (Standard: 2020-01-01 bis 2024-10-30) Daten für alle Geschäftstage vorhanden sind.
     - Fehlende Werte für OHLCV-Daten (Open, High, Low, Close, Volume) werden durch lineare Interpolation und anschließendes Forward/Backward-Fill aufgefüllt.
   - **Hinzufügen technischer Indikatoren (`add_technical_indicators`):**
     - Verwendet `stockstats`, um gängige technische Indikatoren zu den Daten hinzuzufügen.
     - Standardindikatoren: "macd", "rsi_30", "cci_30", "wr_14".

### 2. Aktien-Clustering mittels DTW
   - **Log-Returns berechnen (`calculate_log_return`):**
     - Berechnet die täglichen logarithmischen Renditen der Schlusskurse für jede Aktie.
   - **DTW-basiertes Clustering (`dtw_cluster`):**
     - Gruppiert Aktien basierend auf der Ähnlichkeit ihrer Log-Return-Zeitreihen.
     - Verwendet Dynamic Time Warping (DTW) als Distanzmaß zwischen den Zeitreihen.
     - Nutzt den k-Medoids-Algorithmus (`pyclustering`) zur eigentlichen Clusterbildung.
     - Im Notebook wird für die Trainingsdaten (bis 2023-12-31) mit `k=4` Clustern gearbeitet.
   - **Visualisierung der Cluster (`plot_clusters`):**
     - Stellt die Log-Return-Zeitreihen der Aktien innerhalb jedes Clusters grafisch dar.

### 3. Reinforcement Learning (RL) Umgebung (`CustomStockTradingEnv`)
   - Eine benutzerdefinierte Handelsumgebung, die von `gym.Env` erbt und speziell auf die Gegebenheiten des taiwanesischen Aktienmarktes zugeschnitten ist.
   - **Wichtige Umgebungsparameter und -logik:**
     - **Input-Daten:** Numpy-Arrays für Preise, technische Indikatoren, optional Handelsvolumina und Marktindex-Preise.
     - **Kapital:** `initial_amount` (Startkapital).
     - **Handelskosten (Taiwan):**
       - Kaufprovision: `buy_brokerage_pct` (0.1425%).
       - Verkaufsprovision: `sell_brokerage_pct` (0.1425%).
       - Transaktionssteuer (Verkauf): `transaction_tax_pct` (0.3%).
     - **Aktionsraum:** Kontinuierlich (`spaces.Box`), Werte zwischen -1 (alles verkaufen) und 1 (maximal kaufen) für jede Aktie, repräsentiert gewünschte Portfolio-Gewichtungsänderungen.
     - **Beobachtungsraum:** Umfasst Bargeld, Aktienbestände, aktuelle Kurse, technische Indikatoren und optional den Marktindex.
     - **Handelsbeschränkungen:**
       - Kurslimits (漲跌停 ±10%): Verhindert Käufe bei Limit-Down und Verkäufe bei Limit-Up.
       - Liquiditätsreduktion: Bei Erreichen der Kurslimits wird das handelbare Volumen für die Aktion auf 5% des `max_stock` reduziert.
       - Mindesthandelvolumen (`min_trade_unit`).
     - **Stop-Loss-Mechanismus:** Verkauft Aktien automatisch, wenn der Kurs um mehr als 10% gegenüber dem Vortag fällt (außer bei Limit-Down).
     - **Risikokontrolle im Training:** Beendet Episoden vorzeitig bei zu hoher Tagesrendite (>15%) oder zu großem maximalen Drawdown (< -40%).
   - **Hilfsfunktion (`create_env_for_stock_np`):**
     - Bereitet die Daten für eine spezifische Aktiengruppe vor (pivotieren, NaN-Handling).
     - Fügt optional Slippage (Kursabweichung) für das Training hinzu.
     - Instanziiert die `CustomStockTradingEnv`.

### 4. Modelltraining
   - **Algorithmus:** Proximal Policy Optimization (PPO) von `stable_baselines3`, integriert über die `DRLAgent`-Klasse von `finrl`.
   - **Strategie:** Für jedes zuvor identifizierte Aktiencluster wird ein separates PPO-Modell trainiert.
     - `train_model_for_cluster`: Trainiert ein Modell für ein einzelnes Cluster.
     - `train_all_models`: Iteriert über alle Cluster und führt das Training durch.
   - **Trainingsdaten:** Zeitraum vom 2020-01-01 bis 2023-12-31.
   - **Trainingsschritte:** `10_000` Timesteps pro Modell.
   - **Modellspeicherung:** Die trainierten Modelle werden als `.zip`-Dateien im angegebenen `model_dir` gespeichert.

### 5. Modelltest und Analyse (im Notebook auskommentiert)
   - **Backtesting (`test_model_by_tic`):**
     - Lädt ein trainiertes Modell basierend auf dem Cluster der Testaktie.
     - Führt einen Backtest auf Testdaten (2024-01-01 bis 2024-10-30) durch.
     - Protokolliert detaillierte Handelsaktionen und Zustände für jede Aktie im getesteten Cluster.
     - Visualisiert die Vermögensentwicklung und gibt Performancemetriken aus (kumulative Rendite, max. Drawdown, Sharpe Ratio).
   - **Batch-Testing (`test_multiple_stocks_by_tic_list`):**
     - War vorgesehen, um `test_model_by_tic` für eine Liste von Aktien auszuführen.
   - **Ergebnisvisualisierung aus Logs:**
     - Ein Abschnitt war vorgesehen, um die Log-CSV-Dateien (z.B. `1101.TW_log.csv`) einzulesen und Preis, Gesamtvermögen sowie Belohnungen über die Zeit zu plotten.

### 6. Belohnungsfunktion (Reward)
Die Belohnung des Agenten ist eine Summe aus fünf Komponenten, die alle durch `reward_scaling` skaliert werden:
   $$ \text{Reward} = R_{\text{asset}} - P_{\text{cost}} - P_{\text{volatility}} - P_{\text{cash\_util}} - P_{\text{drawdown}} $$
   - $R_{\text{asset}}$: Belohnung für Vermögenszuwachs.
   - $P_{\text{cost}}$: Strafe für Transaktionskosten (Provisionen, Steuern).
   - $P_{\text{volatility}}$: Strafe für die Volatilität der täglichen Renditen (Standardabweichung der letzten 5 Tage).
   - $P_{\text{cash\_util}}$: Strafe für die Abweichung von einer Ziel-Cash-Quote (z.B. 5%).
   - $P_{\text{drawdown}}$: Strafe, wenn der maximale Drawdown 10% übersteigt.

## Wichtige Bibliotheken
- `os`, `pandas`, `numpy`: Basis-Datenverarbeitung.
- `sklearn.preprocessing.MinMaxScaler`: (Importiert, aber nicht direkt im gezeigten Code verwendet).
- `tslearn.metrics.cdist_dtw`: Für Dynamic Time Warping.
- `pyclustering.cluster.kmedoids`: Für K-Medoids Clustering.
- `matplotlib`: Für Visualisierungen.
- `finrl.agents.stablebaselines3.models.DRLAgent`: Für RL-Agenten-Management.
- `stable_baselines3.PPO`: PPO-Algorithmus Implementierung.
- `gym`: Basis für RL-Umgebungen.
- `stockstats.StockDataFrame`: Für die Berechnung technischer Indikatoren.

## Zusammenfassung und nächste Schritte
Das Notebook demonstriert einen umfassenden Ansatz für den algorithmischen Aktienhandel, der Datenvorverarbeitung, Merkmalsextraktion (technische Indikatoren, Log-Returns), Clustering von Vermögenswerten zur Strategiedifferenzierung und das Training von PPO-Agenten in einer realitätsnahen, benutzerdefinierten Umgebung umfasst.

**Mögliche nächste Schritte bei der Einarbeitung:**
-   Den auskommentierten Test- und Analyse-Code aktivieren und durchführen.
-   Die Ergebnisse der verschiedenen Cluster-Modelle vergleichen.
-   Hyperparameter der PPO-Modelle und der DTW-Clusteranalyse optimieren.
-   Weitere oder andere technische Indikatoren und Marktinformationen (z.B. Volumina, Indexpreise) in die Beobachtung und als Features für das Clustering einbeziehen.
-   Die Robustheit der Cluster über verschiedene Zeiträume untersuchen.
-   Die Belohnungsfunktion weiter verfeinern.
-   Die Auswirkungen der Risikokontrollmechanismen (Stop-Loss, Trainingsabbruch) analysieren.
