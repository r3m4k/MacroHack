# Спецификация: 3 альтернативных модели прогнозирования кривой доходности

## Общий контекст

Три модели решают одну задачу: прогноз 9 теноров кривой доходности на 6 месяцев вперёд. Каждая модель реализуется в двух вариантах: M1 (без IV) и M2 (с IV-фичами). Все три модели объединяет отказ от рекурсивного прогноза (который расходится) в пользу стабильных стратегий.

**Входные данные одинаковы для всех моделей.**

---

## Входные данные

### Кривая доходности

**Файл:** `Problem_1_yield_curve_train.xlsx`

```
                  O/N         1W         2W  ...         6M         1Y         2Y
Month
2019-03-31  17.353072  18.029116  18.356644  ...  26.943709  36.723876  55.907225
...
2025-09-30  27.018171  27.808797  28.025353  ...  37.626871  47.167868  67.606288
[79 rows x 9 columns]
```

- Индекс: `Month`, даты конца месяца
- 79 строк (2019-03 → 2025-09), 9 столбцов
- Значения анонимизированы (монотонное преобразование), диапазон ~17–93

**Загрузка:**
```python
yield_df = pd.read_excel('Problem_1_yield_curve_train.xlsx', index_col='Month', parse_dates=True)
yield_df.index = yield_df.index.to_period('M').to_timestamp()  # конец → начало месяца
```

### Поверхность IV

**Файл:** `Problem_1_IV_train.xlsx`

```
            Date Maturity  Maturity (year fraction)  Strike  Volatility
0     2019-03-01       1M                  0.082192     3.0    1.202842
...
37859 2025-09-01      10Y                 10.000000    45.0    8.062223
[37860 rows x 5 columns]
```

- Long-формат: каждая строка = одна точка (дата, тенор, страйк, волатильность)
- Даты — начало месяца

**Загрузка и агрегация:**
```python
iv_raw = pd.read_excel('Problem_1_IV_train.xlsx')
iv_raw['Date'] = pd.to_datetime(iv_raw['Date'])
IV_TENORS = ["1M", "2M", "3M", "6M", "9M", "1Y", "2Y", "3Y", "4Y",
             "5Y", "6Y", "7Y", "8Y", "9Y", "10Y"]
iv_df = iv_raw.groupby(['Date', 'Maturity'])['Volatility'].mean().unstack('Maturity')[IV_TENORS]
# Результат: DataFrame (79, 15)
```

### Константы

```python
YIELD_TENORS = ["O/N", "1W", "2W", "1M", "2M", "3M", "6M", "1Y", "2Y"]
BACKTEST_SPLIT = "2025-03"   # обучение до этой даты для backtesting
FULL_TRAIN_END = "2025-09"   # все доступные данные
N_FORECAST = 6               # горизонт прогноза

# Веса метрики
WEIGHT_ON = 0.4
WEIGHT_OTHER = 0.075  # = 0.6 / 8
```

### Метрика (одинаковая для всех)

```python
def weighted_rmse(y_true, y_pred):
    """y_true, y_pred: shape (T, 9), столбцы в порядке YIELD_TENORS"""
    T = y_true.shape[0]
    weights = np.array([0.4, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075])
    sq_errors = (y_true - y_pred) ** 2
    return np.sqrt(np.sum(weights[np.newaxis, :] * sq_errors) / T)
```

### Разделение данных (одинаковое для всех)

```python
yield_bt_train = yield_df.loc[:"2025-03"]     # ~72 строки — обучение для backtesting
yield_bt_test  = yield_df.loc["2025-04":"2025-09"]  # 6 строк — проверка backtesting
yield_full     = yield_df.loc[:"2025-09"]      # 79 строк — финальное обучение

iv_bt_train = iv_df.loc[:"2025-03"]
iv_full     = iv_df.loc[:"2025-09"]
```

### Выход (одинаковый для всех)

Файл `Problem_1_yield_curve_predict.xlsx`, 2 вкладки (M1, M2), каждая 6×9, без NaN.
Даты: 2025-10, 2025-11, 2025-12, 2026-01, 2026-02, 2026-03.

Плюс backtesting графики: поверхности дельт для M1 и M2.

---

## Зависимости

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.linear_model import Ridge, LassoCV, ElasticNet, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# Для Модели 1 (ARIMA):
from pmdarima import auto_arima
# pip install pmdarima

# Для Модели 2 (Theta):
from statsforecast import StatsForecast
from statsforecast.models import Theta
# pip install statsforecast
```

---

# ═══════════════════════════════════════════════════════════════════════
# МОДЕЛЬ 1: ARIMA + Direct Forecast
# ═══════════════════════════════════════════════════════════════════════

## Идея

Два компонента:
- **ARIMA** (per-tenor): стабильный статистический baseline, прогнозирует каждый тенор отдельно, не расходится
- **Direct Ridge** (per-tenor, per-horizon): 6 отдельных Ridge-моделей для каждого горизонта, без рекурсии

Финальный прогноз = взвешенное среднее ARIMA и Direct Ridge.

M1 использует yield-фичи. M2 добавляет IV-фичи. ARIMA одинакова для M1 и M2 (не принимает экзогенные фичи). Разница M1 vs M2 — только через Direct Ridge.

## Компонента 1: ARIMA

### Обучение

```python
from pmdarima import auto_arima

arima_models = {}

for tenor in YIELD_TENORS:
    series = yield_train[tenor].values

    model = auto_arima(
        series,
        start_p=0, max_p=4,
        start_q=0, max_q=4,
        d=None,              # auto-detect differencing
        seasonal=False,       # месячные данные, нет явной сезонности
        stepwise=True,        # быстрый поиск
        suppress_warnings=True,
        error_action='ignore',
        max_order=8,          # ограничить суммарный порядок p+q
        information_criterion='aic',
    )
    arima_models[tenor] = model
    print(f"  {tenor}: ARIMA{model.order}, AIC={model.aic():.2f}")
```

### Прогноз

```python
arima_forecasts = {}

for tenor in YIELD_TENORS:
    forecast = arima_models[tenor].predict(n_periods=N_FORECAST)
    # forecast: np.array длиной 6
    arima_forecasts[tenor] = forecast

# Собрать в матрицу (6, 9)
arima_pred = np.column_stack([arima_forecasts[t] for t in YIELD_TENORS])
```

**Важно:** ARIMA прогнозирует все 6 шагов за один вызов `predict(n_periods=6)`. Внутри она делает рекурсию, но её параметры (AR-коэффициенты < 1) гарантируют стабильность.

## Компонента 2: Direct Ridge

### Ключевое отличие от рекурсивного

Вместо одной модели, которая прогнозирует y(t+1) и рекурсивно подставляет свой прогноз, обучаем **6 отдельных моделей**:

```
Модель h=1: y(t+1) = f₁(features(t))
Модель h=2: y(t+2) = f₂(features(t))
Модель h=3: y(t+3) = f₃(features(t))
Модель h=4: y(t+4) = f₄(features(t))
Модель h=5: y(t+5) = f₅(features(t))
Модель h=6: y(t+6) = f₆(features(t))
```

Каждая модель fₕ обучена на парах (features(t), y(t+h)) из обучающей выборки. При прогнозе: features вычисляются из последнего доступного месяца (сентябрь 2025), и каждая модель даёт свой прогноз напрямую. **Никакой рекурсии.**

### Feature Engineering (сокращённый набор)

Учитывая проблемы с расхождением из-за большого числа фичей, использовать **минимальный набор**:

```python
def make_features_direct(df, iv_df=None):
    """Сокращённый набор фичей для direct forecast."""
    features = pd.DataFrame(index=df.index)

    # Лаги (только lag1 и lag2 — lag3 избыточен при direct)
    for lag in [1, 2]:
        shifted = df[YIELD_TENORS].shift(lag)
        shifted.columns = [f"{c}_lag{lag}" for c in YIELD_TENORS]
        features = pd.concat([features, shifted], axis=1)
    # 18 столбцов

    # Дельта месячная
    delta = df[YIELD_TENORS].diff(1)
    delta.columns = [f"{c}_delta1" for c in YIELD_TENORS]
    features = pd.concat([features, delta], axis=1)
    # +9 = 27 столбцов

    # Ключевые спреды (только 3 штуки)
    features['spread_2Y_ON'] = df['2Y'] - df['O/N']
    features['spread_1Y_3M'] = df['1Y'] - df['3M']
    features['spread_6M_1M'] = df['6M'] - df['1M']
    # +3 = 30 столбцов

    # IV-фичи (только для M2)
    if iv_df is not None:
        # Лагированные IV (shift=1, только 5 ключевых теноров)
        key_iv = ['1M', '3M', '1Y', '5Y', '10Y']
        for t in key_iv:
            if t in iv_df.columns:
                features[f'iv_{t}_lag1'] = iv_df[t].shift(1)
        # +5 = 35 столбцов

        # Спред термструктуры IV
        features['iv_spread_10Y_1M'] = iv_df['10Y'].shift(1) - iv_df['1M'].shift(1)
        # +1 = 36 столбцов

        # Vol-of-vol
        features['iv_vol_of_vol'] = iv_df.rolling(3).std().shift(1).mean(axis=1)
        # +1 = 37 столбцов

    return features
```

**M1:** ~30 фичей. **M2:** ~37 фичей. Это в 2–4 раза меньше, чем раньше (79/132).

### Подготовка обучающих данных для direct forecast

```python
def prepare_direct_data(features_df, yield_df, horizon):
    """
    Для горизонта h: сдвинуть таргет на h шагов назад.
    X(t) → y(t+h)

    Аргументы:
        features_df: DataFrame с фичами
        yield_df:    DataFrame с целевыми значениями
        horizon:     int (1..6)

    Возвращает:
        X: DataFrame — фичи (строки, для которых есть таргет)
        y: DataFrame — сдвинутый таргет
    """
    y_shifted = yield_df[YIELD_TENORS].shift(-horizon)
    # shift(-h): строка t получает значение y(t+h)
    # Последние h строк → NaN (нет будущего)

    common_idx = features_df.dropna().index.intersection(y_shifted.dropna().index)
    X = features_df.loc[common_idx]
    y = y_shifted.loc[common_idx]
    return X, y
```

**Пример:** Для horizon=3 и обучающей выборки до 2025-03:
- X содержит фичи за 2019-06..2024-12 (~67 строк)
- y содержит значения за 2019-09..2025-03 (сдвинуто на 3 месяца)
- Последние 3 строки (2025-01..2025-03) не имеют таргета → выпадают

### Обучение Direct Ridge

```python
def train_direct_ridge(features_df, yield_df, iv_features_df=None, model_type="M1"):
    """
    Обучает 6 × 9 = 54 модели Ridge (6 горизонтов × 9 теноров).

    Возвращает:
        models: dict {horizon: {tenor: Ridge_model}}
        scalers: dict {horizon: StandardScaler}
        selected: dict {horizon: {tenor: [feature_names]}}
    """
    if model_type == "M2" and iv_features_df is not None:
        X_all = pd.concat([features_df, iv_features_df], axis=1)
    else:
        X_all = features_df

    models = {}
    scalers = {}
    selected = {}

    for h in range(1, N_FORECAST + 1):  # h = 1..6
        X, y = prepare_direct_data(X_all, yield_df, horizon=h)

        # Масштабирование
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scalers[h] = scaler

        # Lasso-отбор фичей (для каждого горизонта отдельно!)
        selected[h] = {}
        models[h] = {}

        for tenor in YIELD_TENORS:
            lasso = LassoCV(alphas=np.logspace(-5, 1, 100), cv=5, max_iter=20000)
            lasso.fit(X_scaled, y[tenor].values)

            mask = np.abs(lasso.coef_) > 1e-8
            sel_features = [name for name, s in zip(X.columns, mask) if s]

            # Fallback
            if len(sel_features) == 0:
                sel_features = [f"{tenor}_lag1"]

            selected[h][tenor] = sel_features

            # Ridge на отобранных фичах
            sel_idx = [list(X.columns).index(f) for f in sel_features]
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_scaled[:, sel_idx], y[tenor].values)
            models[h][tenor] = ridge

        print(f"    Горизонт h={h}: обучено, "
              f"фичей: {[len(selected[h][t]) for t in YIELD_TENORS]}")

    return models, scalers, selected
```

### Прогноз Direct Ridge

```python
def predict_direct_ridge(models, scalers, selected, features_last_row, feature_names):
    """
    Прогноз на 6 шагов вперёд.

    features_last_row: pd.Series или 1D array — фичи последнего месяца (сентябрь 2025)
    feature_names: list — имена фичей (для индексации)

    Возвращает: np.array (6, 9)
    """
    predictions = np.zeros((N_FORECAST, len(YIELD_TENORS)))

    for h in range(1, N_FORECAST + 1):
        x = features_last_row.values.reshape(1, -1)
        x_scaled = scalers[h].transform(x)

        for j, tenor in enumerate(YIELD_TENORS):
            sel_idx = [feature_names.index(f) for f in selected[h][tenor]]
            predictions[h - 1, j] = models[h][tenor].predict(x_scaled[:, sel_idx])[0]

    return predictions
```

**ВАЖНО:** Все 6 горизонтов используют **одни и те же фичи** из последнего известного месяца. Нет рекурсии, нет подстановки промежуточных прогнозов.

## Ансамбль ARIMA + Direct Ridge

```python
# arima_pred: np.array (6, 9) — прогноз ARIMA
# ridge_pred: np.array (6, 9) — прогноз Direct Ridge

# Вес подбирается на backtesting (2025-04..2025-09)
# Перебор w от 0 до 1 с шагом 0.05
# Для каждого w: pred = w * ridge + (1-w) * arima → weighted_rmse
# best_w = argmin(rmse)

final_pred = best_w * ridge_pred + (1 - best_w) * arima_pred
```

**Ожидаемый результат:** w_ridge ≈ 0.3–0.5. ARIMA даёт стабильную базу, Ridge добавляет коррекцию через фичи.

Для M2: Ridge использует IV-фичи → его прогноз (и вклад) отличается от M1. ARIMA одинакова.

---

# ═══════════════════════════════════════════════════════════════════════
# МОДЕЛЬ 2: Theta + Huber Regressor
# ═══════════════════════════════════════════════════════════════════════

## Идея

- **Theta method**: стабильный статистический baseline с минимумом параметров. Победитель M3-competition. Не переобучается по определению
- **Huber Regressor**: ML-модель, устойчивая к выбросам (ковид-2020, шоки-2022). Direct forecast (не рекурсивный)

Финальный прогноз = взвешенное среднее Theta и Huber.

## Компонента 1: Theta

### Обучение и прогноз

```python
from statsforecast import StatsForecast
from statsforecast.models import Theta

def theta_forecast(yield_train, n_forecast=6):
    """
    Theta-прогноз для каждого тенора.
    Возвращает: np.array (n_forecast, 9)
    """
    predictions = np.zeros((n_forecast, len(YIELD_TENORS)))

    for j, tenor in enumerate(YIELD_TENORS):
        series = yield_train[tenor].values

        # Подготовка данных в формате StatsForecast
        sf_df = pd.DataFrame({
            'unique_id': ['series'] * len(series),
            'ds': yield_train.index,
            'y': series
        })

        sf = StatsForecast(
            models=[Theta(season_length=1)],  # без сезонности
            freq='MS',                         # monthly start
            n_jobs=1
        )
        sf.fit(sf_df)
        forecast = sf.predict(h=n_forecast)
        predictions[:, j] = forecast['Theta'].values

    return predictions
```

**Альтернативная реализация без statsforecast (если установка проблематична):**

```python
def theta_manual(series, n_forecast=6):
    """
    Упрощённая реализация Theta method.
    Theta = 0.5 * SES_forecast + 0.5 * linear_drift

    SES = Simple Exponential Smoothing
    linear_drift = последнее значение + средний рост за историю
    """
    n = len(series)
    last = series[-1]

    # Компонента 1: SES (alpha подбирается)
    best_alpha = 0.3
    best_sse = np.inf
    for alpha in np.arange(0.05, 1.0, 0.05):
        level = series[0]
        sse = 0
        for t in range(1, n):
            level = alpha * series[t] + (1 - alpha) * level
            sse += (series[t] - level) ** 2
        if sse < best_sse:
            best_sse = sse
            best_alpha = alpha

    # SES-прогноз (плоская линия = последний сглаженный уровень)
    level = series[0]
    for t in range(1, n):
        level = best_alpha * series[t] + (1 - best_alpha) * level
    ses_forecast = np.full(n_forecast, level)

    # Компонента 2: Linear drift
    drift = (series[-1] - series[0]) / (n - 1)
    drift_forecast = last + drift * np.arange(1, n_forecast + 1)

    # Theta = среднее двух компонент
    return 0.5 * ses_forecast + 0.5 * drift_forecast


def theta_forecast_manual(yield_train, n_forecast=6):
    predictions = np.zeros((n_forecast, len(YIELD_TENORS)))
    for j, tenor in enumerate(YIELD_TENORS):
        predictions[:, j] = theta_manual(yield_train[tenor].values, n_forecast)
    return predictions
```

## Компонента 2: Huber Regressor (Direct)

### Почему Huber, а не Ridge

Huber loss = MSE для малых ошибок, MAE для больших:
```
L(e) = 0.5 * e²        если |e| ≤ ε
L(e) = ε * |e| - 0.5ε² если |e| > ε
```

В наших данных есть резкие скачки (2020 — ковид, 2022 — геополитика, 2024 — рост ключевой ставки до 21%). Ridge подгоняется под эти выбросы. Huber ослабляет их влияние.

### Feature Engineering

Тот же сокращённый набор, что и в Модели 1 (`make_features_direct`).

### Обучение (Direct, per-horizon, per-tenor)

```python
def train_direct_huber(features_df, yield_df, iv_features_df=None, model_type="M1"):
    """
    6 горизонтов × 9 теноров = 54 модели HuberRegressor.
    """
    if model_type == "M2" and iv_features_df is not None:
        X_all = pd.concat([features_df, iv_features_df], axis=1)
    else:
        X_all = features_df

    models = {}
    scalers = {}
    selected = {}

    for h in range(1, N_FORECAST + 1):
        X, y = prepare_direct_data(X_all, yield_df, horizon=h)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scalers[h] = scaler

        selected[h] = {}
        models[h] = {}

        for tenor in YIELD_TENORS:
            # Lasso-отбор
            lasso = LassoCV(alphas=np.logspace(-5, 1, 100), cv=5, max_iter=20000)
            lasso.fit(X_scaled, y[tenor].values)
            mask = np.abs(lasso.coef_) > 1e-8
            sel_features = [name for name, s in zip(X.columns, mask) if s]
            if len(sel_features) == 0:
                sel_features = [f"{tenor}_lag1"]
            selected[h][tenor] = sel_features

            # Huber на отобранных фичах
            sel_idx = [list(X.columns).index(f) for f in sel_features]
            huber = HuberRegressor(
                epsilon=1.35,    # стандартный порог (95% эффективности при нормальном распределении)
                max_iter=500,
                alpha=0.01       # слабая L2-регуляризация
            )
            huber.fit(X_scaled[:, sel_idx], y[tenor].values)
            models[h][tenor] = huber

    return models, scalers, selected
```

### Прогноз

Идентичен `predict_direct_ridge` из Модели 1 — та же логика direct forecast.

## Ансамбль Theta + Huber

```python
final_pred = best_w * huber_pred + (1 - best_w) * theta_pred
# Подбор w на backtesting (2025-04..2025-09)
```

**Ожидаемый результат:** w_huber ≈ 0.3–0.5. Theta стабильнее (меньше параметров), Huber ловит нелинейности.

---

# ═══════════════════════════════════════════════════════════════════════
# МОДЕЛЬ 3: Last Value + Delta Forecast
# ═══════════════════════════════════════════════════════════════════════

## Идея

Вместо прогноза абсолютного значения ставки прогнозируем **изменение** (дельту):

```
ŷ(t+h) = y(t) + Δ̂(t, h)
```

где `y(t)` — последнее известное значение (факт), `Δ̂(t, h)` — прогнозируемое изменение за h месяцев.

**Почему это стабильно:** Даже если модель дельт ошибётся (скажем, предскажет Δ = +2 вместо Δ = −1), прогноз будет `27 + 2 = 29` вместо `27 - 1 = 26`. Ошибка = 3 единицы. При рекурсивном прогнозе абсолютных значений ошибка может быть 10–20 единиц из-за накопления.

## Подготовка таргета

```python
def prepare_delta_targets(yield_df):
    """
    Для каждого горизонта h создаёт таргет: y(t+h) - y(t)

    Возвращает: dict {h: DataFrame с дельтами}
    """
    delta_targets = {}

    for h in range(1, N_FORECAST + 1):
        shifted = yield_df[YIELD_TENORS].shift(-h)
        delta = shifted - yield_df[YIELD_TENORS]
        # delta(t) = y(t+h) - y(t)
        delta_targets[h] = delta.dropna()

    return delta_targets
```

**Пример:** Для h=3, строка за июнь 2024 содержит дельту `y(сентябрь 2024) - y(июнь 2024)`.

## Feature Engineering

### Yield-фичи (для M1 и M2)

Фичи тоже должны быть **в терминах изменений**, чтобы соответствовать таргету:

```python
def make_delta_features(df, iv_df=None):
    """Фичи в терминах изменений — для прогноза дельт."""
    features = pd.DataFrame(index=df.index)

    # Последние значения (текущая точка, от которой считаем дельту)
    for tenor in YIELD_TENORS:
        features[f'{tenor}_current'] = df[tenor]
    # 9 столбцов

    # Дельта за 1 месяц (momentum)
    delta1 = df[YIELD_TENORS].diff(1)
    delta1.columns = [f"{c}_delta1" for c in YIELD_TENORS]
    features = pd.concat([features, delta1], axis=1)
    # +9 = 18

    # Дельта за 3 месяца (квартальный тренд)
    delta3 = df[YIELD_TENORS].diff(3)
    delta3.columns = [f"{c}_delta3" for c in YIELD_TENORS]
    features = pd.concat([features, delta3], axis=1)
    # +9 = 27

    # Спреды (форма кривой)
    features['spread_2Y_ON'] = df['2Y'] - df['O/N']
    features['spread_1Y_3M'] = df['1Y'] - df['3M']
    features['spread_6M_1M'] = df['6M'] - df['1M']
    # +3 = 30

    # Изменение спредов за месяц (динамика формы)
    features['d_spread_2Y_ON'] = features['spread_2Y_ON'].diff(1)
    features['d_spread_1Y_3M'] = features['spread_1Y_3M'].diff(1)
    # +2 = 32

    # Отклонение от MA6 (mean-reversion signal)
    for tenor in YIELD_TENORS:
        features[f'{tenor}_dev_ma6'] = df[tenor] - df[tenor].rolling(6).mean()
    # +9 = 41

    # IV-фичи (только для M2)
    if iv_df is not None:
        # Изменение IV (не уровень, а дельта — согласовано с таргетом)
        key_iv = ['1M', '3M', '1Y', '5Y', '10Y']
        for t in key_iv:
            if t in iv_df.columns:
                features[f'iv_{t}_delta1'] = iv_df[t].diff(1).shift(1)
        # +5 = 46

        # Уровень IV (лагированный)
        for t in key_iv:
            if t in iv_df.columns:
                features[f'iv_{t}_lag1'] = iv_df[t].shift(1)
        # +5 = 51

        # IV term spread
        features['iv_spread_10Y_1M'] = iv_df['10Y'].shift(1) - iv_df['1M'].shift(1)
        # +1 = 52

    return features
```

**M1:** ~41 фича. **M2:** ~52 фичи.

**Ключевое отличие от предыдущих подходов:** Фича `{tenor}_dev_ma6` — отклонение текущего значения от 6-месячного среднего. Это сигнал mean-reversion: если ставка сильно выше MA6, дельта скорее будет отрицательной (возврат к среднему).

## Обучение

```python
def train_delta_model(features_df, yield_df, iv_features_df=None, model_type="M1"):
    """
    Direct forecast дельт: 6 горизонтов × 9 теноров.
    Таргет: y(t+h) - y(t) вместо y(t+h).
    Модель: ElasticNet (L1+L2, сам отбирает фичи).
    """
    if model_type == "M2" and iv_features_df is not None:
        X_all = pd.concat([features_df, iv_features_df], axis=1)
    else:
        X_all = features_df

    delta_targets = prepare_delta_targets(yield_df)

    models = {}
    scalers = {}

    for h in range(1, N_FORECAST + 1):
        deltas = delta_targets[h]
        common_idx = X_all.dropna().index.intersection(deltas.index)
        X = X_all.loc[common_idx]
        y = deltas.loc[common_idx]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scalers[h] = scaler

        models[h] = {}
        for tenor in YIELD_TENORS:
            # ElasticNet: объединяет Lasso-отбор и Ridge-регуляризацию
            model = ElasticNet(
                alpha=0.1,       # общая сила регуляризации
                l1_ratio=0.7,    # 70% L1 (отбор) + 30% L2 (стабильность)
                max_iter=10000
            )
            model.fit(X_scaled, y[tenor].values)
            models[h][tenor] = model

            n_nonzero = np.sum(np.abs(model.coef_) > 1e-8)
            # Опционально: print(f"  h={h} {tenor}: {n_nonzero} фичей")

    return models, scalers
```

**Почему ElasticNet, а не Lasso+Ridge:**
При прогнозе дельт (которые часто малы и шумны) отдельный Lasso-этап может отбросить все фичи. ElasticNet мягче: L1-часть обнуляет самые слабые, L2-часть стабилизирует оставшиеся.

## Прогноз

```python
def predict_delta(models, scalers, features_last_row, last_known_values, feature_names):
    """
    Прогноз через дельты:
    ŷ(t+h) = y(t) + Δ̂(t, h)

    Аргументы:
        features_last_row: фичи за последний месяц (сентябрь 2025)
        last_known_values: np.array (9,) — y(t) для каждого тенора (последний факт)

    Возвращает: np.array (6, 9) — абсолютные прогнозные значения
    """
    predictions = np.zeros((N_FORECAST, len(YIELD_TENORS)))

    x = features_last_row.values.reshape(1, -1)

    for h in range(1, N_FORECAST + 1):
        x_scaled = scalers[h].transform(x)

        for j, tenor in enumerate(YIELD_TENORS):
            delta_pred = models[h][tenor].predict(x_scaled)[0]
            predictions[h - 1, j] = last_known_values[j] + delta_pred

    return predictions
```

**КРИТИЧНО:** `last_known_values` — это **факт** за последний месяц обучающей выборки (сентябрь 2025). Берётся из `yield_df.iloc[-1][YIELD_TENORS].values`. Прогноз всегда привязан к факту, не к предыдущему прогнозу.

## Дополнительная стабилизация: clipping дельт

```python
# Ограничить дельты на основе исторического диапазона
historical_deltas = yield_df[YIELD_TENORS].diff(1)
max_delta = historical_deltas.abs().quantile(0.95).max()  # 95-й перцентиль

# При прогнозе:
delta_pred = np.clip(delta_pred, -max_delta * h, max_delta * h)
# h-кратный лимит: на горизонте 6 месяцев допускается большее изменение
```

Это предотвращает ситуацию, когда модель предсказывает нереалистично большую дельту.

## M1 vs M2

M1 и M2 отличаются только набором фичей:
- M1: `make_delta_features(df, iv_df=None)` — 41 фича
- M2: `make_delta_features(df, iv_df=iv_df)` — 52 фичи

Модель (ElasticNet), таргет (дельты), стратегия (direct) — одинаковы.

---

# ═══════════════════════════════════════════════════════════════════════
# ОБЩЕЕ: BACKTESTING И ВИЗУАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════════

## Backtesting (одинаковый для всех 3 моделей)

```
Для каждой из 3 моделей:
    1. Обучить на данных до 2025-03
    2. Прогноз на 2025-04..2025-09 (6 шагов)
    3. Сравнить с фактом
    4. Посчитать weighted_rmse для M1 и M2
    5. Построить поверхности дельт
```

## Визуализация поверхности дельт

```python
def plot_backtest(delta_m1, delta_m2, actual, pred_m1, pred_m2,
                  rmse_m1, rmse_m2, model_name, filename):
    """
    4-панельный график для backtesting.

    Панель 1: Heatmap дельт M1 (6×9, цвет = прогноз − факт)
    Панель 2: Heatmap дельт M2
    Панель 3: |ошибка M1| − |ошибка M2| (зелёный = M2 лучше)
    Панель 4: RMSE по тенорам M1 vs M2 (grouped bar chart)
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f'{model_name}\n'
        f'Backtesting 2025-04..2025-09 | '
        f'wRMSE M1={rmse_m1:.4f}  M2={rmse_m2:.4f}  '
        f'Δ={(rmse_m1-rmse_m2)/rmse_m1*100:+.1f}%',
        fontsize=14, fontweight='bold'
    )
    bt_dates = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']

    # Панель 1: Дельта M1
    ax = axes[0, 0]
    vmax = max(np.max(np.abs(delta_m1)), np.max(np.abs(delta_m2)))
    im = ax.imshow(delta_m1, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(9)); ax.set_xticklabels(YIELD_TENORS, fontsize=8)
    ax.set_yticks(range(6)); ax.set_yticklabels(bt_dates)
    ax.set_title('M1: прогноз − факт')
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(6):
        for j in range(9):
            ax.text(j, i, f'{delta_m1[i,j]:.1f}', ha='center', va='center', fontsize=7)

    # Панель 2: Дельта M2
    ax = axes[0, 1]
    im = ax.imshow(delta_m2, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(9)); ax.set_xticklabels(YIELD_TENORS, fontsize=8)
    ax.set_yticks(range(6)); ax.set_yticklabels(bt_dates)
    ax.set_title('M2: прогноз − факт')
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(6):
        for j in range(9):
            ax.text(j, i, f'{delta_m2[i,j]:.1f}', ha='center', va='center', fontsize=7)

    # Панель 3: Преимущество M2
    ax = axes[1, 0]
    advantage = np.abs(delta_m1) - np.abs(delta_m2)
    vmax_a = np.max(np.abs(advantage))
    im = ax.imshow(advantage, cmap='RdYlGn', aspect='auto', vmin=-vmax_a, vmax=vmax_a)
    ax.set_xticks(range(9)); ax.set_xticklabels(YIELD_TENORS, fontsize=8)
    ax.set_yticks(range(6)); ax.set_yticklabels(bt_dates)
    ax.set_title('Преимущество M2 (зелёный = M2 точнее)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Панель 4: RMSE по тенорам
    ax = axes[1, 1]
    rmse_t_m1 = np.sqrt(np.mean(delta_m1**2, axis=0))
    rmse_t_m2 = np.sqrt(np.mean(delta_m2**2, axis=0))
    x = np.arange(9)
    ax.bar(x - 0.18, rmse_t_m1, 0.35, label='M1', color='#3498db')
    ax.bar(x + 0.18, rmse_t_m2, 0.35, label='M2', color='#e74c3c')
    ax.set_xticks(x); ax.set_xticklabels(YIELD_TENORS)
    ax.set_title('RMSE по тенорам'); ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
```

---

# ═══════════════════════════════════════════════════════════════════════
# MAIN: ЗАПУСК ВСЕХ 3 МОДЕЛЕЙ
# ═══════════════════════════════════════════════════════════════════════

```
main():
    # ═══ ЗАГРУЗКА ═══
    yield_df = загрузить кривую доходности
    iv_df = загрузить и агрегировать IV

    # ═══ РАЗДЕЛЕНИЕ ═══
    yield_bt_train = yield_df.loc[:"2025-03"]
    yield_bt_test  = yield_df.loc["2025-04":"2025-09"]
    iv_bt_train    = iv_df.loc[:"2025-03"]

    # ═══════════════════════════════════════════
    # МОДЕЛЬ 1: ARIMA + Direct Ridge
    # ═══════════════════════════════════════════

    # Backtesting
    arima_bt = train_arima(yield_bt_train) → predict(6)
    ridge_bt_m1 = train_direct_ridge(yield_bt_train, model="M1") → predict(6)
    ridge_bt_m2 = train_direct_ridge(yield_bt_train, iv_bt_train, model="M2") → predict(6)
    подобрать w_m1, w_m2 на bt_test
    bt_pred_m1 = w_m1 * ridge_bt_m1 + (1-w_m1) * arima_bt
    bt_pred_m2 = w_m2 * ridge_bt_m2 + (1-w_m2) * arima_bt
    посчитать и вывести rmse
    построить графики → "backtest_model1.png"

    # Финальный прогноз
    arima_final = train_arima(yield_full) → predict(6)
    ridge_final_m1 = train_direct_ridge(yield_full, model="M1") → predict(6)
    ridge_final_m2 = train_direct_ridge(yield_full, iv_full, model="M2") → predict(6)
    pred_m1 = w_m1 * ridge_final_m1 + (1-w_m1) * arima_final
    pred_m2 = w_m2 * ridge_final_m2 + (1-w_m2) * arima_final
    сохранить → "predict_model1.xlsx"

    # ═══════════════════════════════════════════
    # МОДЕЛЬ 2: Theta + Huber
    # ═══════════════════════════════════════════

    # Backtesting
    theta_bt = theta_forecast(yield_bt_train)
    huber_bt_m1 = train_direct_huber(yield_bt_train, model="M1") → predict(6)
    huber_bt_m2 = train_direct_huber(yield_bt_train, iv_bt_train, model="M2") → predict(6)
    подобрать w_m1, w_m2
    bt_pred_m1 = w_m1 * huber_bt_m1 + (1-w_m1) * theta_bt
    bt_pred_m2 = w_m2 * huber_bt_m2 + (1-w_m2) * theta_bt
    rmse, графики → "backtest_model2.png"

    # Финальный прогноз → "predict_model2.xlsx"

    # ═══════════════════════════════════════════
    # МОДЕЛЬ 3: Last Value + Delta
    # ═══════════════════════════════════════════

    # Backtesting
    last_values_bt = yield_bt_train.iloc[-1][YIELD_TENORS].values
    delta_bt_m1 = train_delta_model(yield_bt_train, model="M1") → predict(6)
    delta_bt_m2 = train_delta_model(yield_bt_train, iv_bt_train, model="M2") → predict(6)
    # Нет ансамбля — delta forecast самодостаточна
    bt_pred_m1 = delta_bt_m1  # уже включает last_value + delta
    bt_pred_m2 = delta_bt_m2
    rmse, графики → "backtest_model3.png"

    # Финальный прогноз → "predict_model3.xlsx"

    # ═══ ИТОГОВОЕ СРАВНЕНИЕ ═══
    print таблицу:
    ┌────────────────────────┬──────────┬──────────┬──────────┐
    │ Модель                 │ BT M1    │ BT M2    │ Δ M2/M1  │
    ├────────────────────────┼──────────┼──────────┼──────────┤
    │ 1. ARIMA+Direct Ridge  │ X.XXXX   │ X.XXXX   │ +XX.X%   │
    │ 2. Theta+Huber         │ X.XXXX   │ X.XXXX   │ +XX.X%   │
    │ 3. Delta Forecast      │ X.XXXX   │ X.XXXX   │ +XX.X%   │
    └────────────────────────┴──────────┴──────────┴──────────┘
    Лучшая модель: ... (по min BT RMSE)

    # Сохранить лучшую как Problem_1_yield_curve_predict.xlsx
```

---

## Ожидаемые результаты

| Модель | Сильные стороны | Слабые стороны | Ожидаемый RMSE |
|---|---|---|---|
| ARIMA + Direct Ridge | ARIMA стабильна, Ridge ловит фичи, direct не расходится | ARIMA univariate, не видит межтенорные связи | Средний |
| Theta + Huber | Theta минимум параметров, Huber устойчив к выбросам | Theta простая (может недооценить тренд) | Лучший для стабильности |
| Delta Forecast | Привязка к факту, не расходится, экономически осмыслен | Может не поймать перелом тренда | Лучший для точности |

**Модель 3 (Delta)** — фаворит. Привязка к последнему факту + direct forecast + mean-reversion фичи. Единственный риск — если тренд резко меняет направление, дельта-модель будет отставать на 1–2 шага. Для этого есть фича `{tenor}_dev_ma6` — сигнал mean-reversion.

---

## Чеклист

- [ ] Все 3 модели реализованы для M1 и M2
- [ ] Backtesting: обучение до 2025-03, прогноз на 2025-04..2025-09
- [ ] Поверхности дельт (heatmap) для каждой модели (3 PNG файла)
- [ ] Итоговая таблица сравнения 3 моделей по backtesting RMSE
- [ ] Лучшая модель сохранена как Problem_1_yield_curve_predict.xlsx
- [ ] Direct forecast (не рекурсивный) во всех ML-компонентах
- [ ] Сокращённый набор фичей (30–52 вместо 79–132)
- [ ] Все IV-фичи сдвинуты на shift(1)
- [ ] Даты yield_df синхронизированы с iv_df
- [ ] Нет NaN в финальных прогнозах