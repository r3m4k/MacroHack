# Техническое задание: Ridge + VAR ансамбль для прогнозирования кривой доходности

## Цель документа

Этот файл — полная спецификация для написания production-ready кода. AI-агент должен по этому документу создать один Python-файл, который:

1. Загружает данные (кривая доходности + поверхность IV)
2. Строит фичи для M1 (без IV) и M2 (с IV)
3. Отбирает фичи через Lasso (L1)
4. Обучает ансамбль Ridge + VAR
5. Делает рекурсивный прогноз на 6 месяцев
6. Считает взвешенный RMSE
7. Сохраняет результат в Excel

---

## Стек и зависимости

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR
import warnings
warnings.filterwarnings("ignore")
```

Все зависимости стандартные: `numpy`, `pandas`, `scikit-learn`, `statsmodels`. Не использовать LightGBM, XGBoost, PyTorch, TensorFlow — они не нужны.

---

## Константы и конфигурация

```python
YIELD_TENORS = ["O/N", "1W", "2W", "1M", "2M", "3M", "6M", "1Y", "2Y"]  # 9 целевых теноров
IV_TENORS = ["1M", "2M", "3M", "6M", "9M", "1Y", "2Y", "3Y", "4Y",
             "5Y", "6Y", "7Y", "8Y", "9Y", "10Y"]  # 15 теноров IV

TRAIN_START = "2019-03"
TRAIN_END = "2025-03"           # обучение заканчивается в марте 2025

VAL_START = "2025-04"           # валидационный период для проверки точности
VAL_END = "2025-09"
N_VAL_MONTHS = 6                # 6 месяцев валидации (апрель–сентябрь 2025)

TEST_START = "2025-10"          # финальный прогноз
TEST_END = "2026-03"
N_TEST_MONTHS = 6               # горизонт прогноза

WEIGHT_ON = 0.4                           # вес O/N в метрике
WEIGHT_OTHER = 0.6 / (len(YIELD_TENORS) - 1)  # = 0.075, вес каждого из остальных 8 теноров
```

---

## Блок 1. Загрузка данных

### 1.1. Кривая доходности

**Вход:** файл `yield_curve_filled.xlsx` (или CSV).

**Формат:**
- Индекс: столбец `Month` (даты, monthly frequency, например `2019-03-01`)
- 9 столбцов: `O/N`, `1W`, `2W`, `1M`, `2M`, `3M`, `6M`, `1Y`, `2Y`
- Значения: числа с плавающей точкой (анонимизированные, строго монотонное преобразование от реальных ставок)
- Размер: 79 строк (март 2019 — сентябрь 2025), из них для обучения используются 73 строки (до 2025-03)

**Код загрузки:**
```python
yield_df = pd.read_excel('yield_curve_filled.xlsx', index_col='Month', parse_dates=True)
```

**Проверки после загрузки:**
- `assert yield_df.shape == (79, 9)` или близко к этому (79 строк: март 2019 — сентябрь 2025)
- `assert yield_df.isnull().sum().sum() == 0` — пропусков быть не должно (файл называется `filled`)
- `assert yield_df.index.is_monotonic_increasing` — даты отсортированы

### 1.2. Поверхность вменённой волатильности

**Вход:** файл с IV (формат уточняется при загрузке).

**Ожидаемая структура:**
- Трёхмерные данные: (месяц, тенор_IV, страйк) → значение σ
- 79 месяцев × 15 теноров × переменное число страйков (для обучения: до 2025-03)
- Всего ~38000 точек

**Предобработка IV (выполняется при загрузке):**

Поскольку количество страйков различается от месяца к месяцу и от тенора к тенору, необходимо агрегировать данные по страйкам. Три стратегии агрегации (реализовать все три, использовать лучшую по CV):

1. **ATM (at-the-money):** Взять значение IV для страйка, ближайшего к текущей спотовой ставке. Результат: DataFrame 78×15.
2. **Среднее по страйкам:** Для каждого (месяц, тенор) посчитать `mean(σ)` по всем доступным страйкам. Результат: DataFrame 78×15.
3. **Квантили:** Для каждого (месяц, тенор) посчитать 25-й, 50-й, 75-й перцентили σ по страйкам. Результат: DataFrame 78×45 (15 теноров × 3 квантиля).

**Основная стратегия:** Начать со стратегии 2 (среднее) как baseline. Если формат данных позволяет определить ATM — использовать стратегию 1. Стратегию 3 использовать только если CV показывает улучшение.

**Результат загрузки IV:**
```python
iv_df  # DataFrame, index = даты (те же, что yield_df), columns = IV_TENORS или расширенный набор
```

**Проверки:**
- Индексы `yield_df` и `iv_df` должны совпадать (или iv_df должен содержать все даты yield_df)
- Нет NaN (если есть — заполнить forward fill: `iv_df.fillna(method='ffill')`)

---

## Блок 2. Feature Engineering

Реализовать как класс `FeatureEngineer` с двумя методами: `yield_features()` и `iv_features()`.

### 2.1. Фичи из кривой доходности (для M1 и M2)

#### 2.1.1. Лагированные значения

```
Для каждого тенора из YIELD_TENORS:
    Для lag in [1, 2, 3]:
        создать столбец "{tenor}_lag{lag}" = yield_df[tenor].shift(lag)
```

- Итого: 9 теноров × 3 лага = **27 столбцов**
- `shift(lag)` сдвигает ряд вниз на `lag` позиций, первые `lag` строк становятся NaN
- Экономический смысл: авторегрессионная память — значения ставок за 1, 2, 3 месяца назад

#### 2.1.2. Спреды между тенорами

```
spread_2Y_ON  = yield_df["2Y"] - yield_df["O/N"]    # полный наклон кривой
spread_1Y_ON  = yield_df["1Y"] - yield_df["O/N"]    # наклон до 1 года
spread_2Y_1Y  = yield_df["2Y"] - yield_df["1Y"]     # наклон длинного конца
spread_6M_3M  = yield_df["6M"] - yield_df["3M"]     # середина кривой
spread_3M_1M  = yield_df["3M"] - yield_df["1M"]     # короткий конец
```

Плюс лагированные спреды:
```
Для lag in [1, 2]:
    Для spread in [spread_2Y_ON, spread_1Y_ON, spread_2Y_1Y]:
        создать "{spread_name}_lag{lag}" = spread.shift(lag)
```

- Итого: 5 спредов + 6 лагированных = **11 столбцов**
- Экономический смысл: форма кривой и её динамика. Отрицательный spread_2Y_ON = инверсия (рынок ждёт снижения ставки)

#### 2.1.3. Скользящие средние

```
Для window in [3, 6]:
    Для каждого тенора из YIELD_TENORS:
        создать "{tenor}_ma{window}" = yield_df[tenor].rolling(window).mean()
```

- Итого: 9 теноров × 2 окна = **18 столбцов**
- `rolling(3).mean()` — среднее за последние 3 месяца (включая текущий)
- `rolling(6)` создаёт NaN в первых 5 строках
- Экономический смысл: тренд и отклонение от тренда

#### 2.1.4. Дельты (изменения)

```
Для diff_period in [1, 3]:
    Для каждого тенора из YIELD_TENORS:
        создать "{tenor}_delta{diff_period}" = yield_df[tenor].diff(diff_period)
```

- Итого: 9 теноров × 2 периода = **18 столбцов**
- `diff(1)` = месячное изменение, `diff(3)` = квартальное
- Экономический смысл: momentum (скорость и направление движения)

#### 2.1.5. Кривизна

```
curvature_1 = 2 * yield_df["1M"] - yield_df["O/N"] - yield_df["6M"]
curvature_2 = 2 * yield_df["1Y"] - yield_df["3M"] - yield_df["2Y"]
```

- Итого: **2 столбца**
- Формула `2×mid - short - long` — дискретный аналог второй производной
- Экономический смысл: нелинейность формы кривой (горб vs впадина)

#### 2.1.6. PCA-компоненты кривой

```python
pca = PCA(n_components=3)
pca_values = pca.fit_transform(yield_df[YIELD_TENORS].values)
# pca_level     = pca_values[:, 0]   # ~80% дисперсии, общий уровень ставок
# pca_slope     = pca_values[:, 1]   # ~10-15%, наклон кривой
# pca_curvature = pca_values[:, 2]   # ~3-5%, кривизна
```

- Итого: **3 столбца**
- **ВАЖНО:** PCA обучается (fit) на обучающей выборке. При прогнозе используется `transform` с тем же объектом PCA. Нельзя делать `fit_transform` на тестовых данных
- Экономический смысл: сжатое представление кривой в 3 независимых фактора (Litterman-Scheinkman)

#### Итого фичей M1: 27 + 11 + 18 + 18 + 2 + 3 = **~79 столбцов**

---

### 2.2. Фичи из поверхности IV (только для M2)

**КРИТИЧЕСКИЙ ПРИНЦИП:** Все IV-фичи должны быть сдвинуты на 1 месяц через `.shift(1)`. Это предотвращает look-ahead bias — мы используем IV прошлого месяца для прогноза текущего.

#### 2.2.1. Лагированные значения IV

```
Для каждого тенора из IV_TENORS:
    создать "iv_{tenor}_lag1" = iv_df[tenor].shift(1)
```

- Итого: **15 столбцов**
- shift(1) — не просто лаг, а защита от подглядывания в будущее

#### 2.2.2. Спреды термструктуры IV

```
iv_spread_10Y_1M = iv_df["10Y"].shift(1) - iv_df["1M"].shift(1)
iv_spread_5Y_1Y  = iv_df["5Y"].shift(1)  - iv_df["1Y"].shift(1)
iv_spread_2Y_6M  = iv_df["2Y"].shift(1)  - iv_df["6M"].shift(1)
```

- Итого: **3 столбца**
- Экономический смысл: форма распределения неопределённости по горизонтам

#### 2.2.3. Дельты IV

```
Для каждого тенора из IV_TENORS:
    создать "iv_{tenor}_delta1" = iv_df[tenor].diff(1).shift(1)
```

- Итого: **15 столбцов**
- `diff(1)` считает изменение, затем `shift(1)` предотвращает look-ahead
- Экономический смысл: резкий рост IV → рынок ожидает движение ставок

#### 2.2.4. Скользящие средние IV

```
Для каждого тенора из IV_TENORS:
    создать "iv_{tenor}_ma3" = iv_df[tenor].rolling(3).mean().shift(1)
```

- Итого: **15 столбцов**
- Экономический смысл: тренд волатильности

#### 2.2.5. PCA поверхности IV

```python
pca_iv = PCA(n_components=3)
iv_clean = iv_df.dropna()
pca_iv_values = pca_iv.fit_transform(iv_clean.values)
# iv_pca_1, iv_pca_2, iv_pca_3
# Затем shift(1) для предотвращения look-ahead
```

- Итого: **3 столбца**
- **ВАЖНО:** PCA обучается на обучающей выборке, shift(1) после проекции
- Экономический смысл: сжатые факторы волатильности

#### 2.2.6. Vol-of-vol

```python
iv_std3 = iv_df.rolling(3).std().shift(1)
iv_vol_of_vol_short = iv_std3.mean(axis=1)  # среднее std по всем 15 тенорам

iv_std6 = iv_df.rolling(6).std().shift(1)
iv_vol_of_vol_long = iv_std6.mean(axis=1)
```

- Итого: **2 столбца**
- Экономический смысл: нестабильность самой волатильности (мета-мера нервозности)

#### Итого фичей IV: 15 + 3 + 15 + 15 + 3 + 2 = **~53 столбца**

#### Итого фичей M2: 79 (yield) + 53 (IV) = **~132 столбца**

---

## Блок 3. Отбор фичей через Lasso (L1)

### 3.1. Логика отбора

Отбор фичей выполняется **для каждого тенора отдельно** и **для M1 и M2 отдельно**. Итого: 9 теноров × 2 модели = 18 запусков Lasso.

### 3.2. Алгоритм

```
Для каждой модели (M1, M2):
    X = соответствующая матрица фичей (M1: yield-фичи, M2: yield + IV)
    Удалить строки с NaN из X и y одновременно (dropna по пересечению индексов)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    selected_features = {}   # словарь {тенор: [список отобранных фичей]}

    Для каждого тенора из YIELD_TENORS:
        y = yield_df[тенор].values (на тех же индексах)

        lasso_cv = LassoCV(
            alphas=np.logspace(-5, 1, 120),  # 120 значений alpha от 0.00001 до 10
            cv=5,                             # 5-fold cross-validation
            max_iter=20000,                   # достаточно для сходимости
        )
        lasso_cv.fit(X_scaled, y)

        # Фичи с |коэффициент| > 1e-8 считаются отобранными
        mask = np.abs(lasso_cv.coef_) > 1e-8
        selected_features[тенор] = [name for name, sel in zip(X.columns, mask) if sel]
```

### 3.3. Результат отбора

- `selected_features_m1`: словарь `{тенор: [список фичей]}` для M1
- `selected_features_m2`: словарь `{тенор: [список фичей]}` для M2
- Ожидаемое количество отобранных фичей: 5–20 для каждого тенора
- Если для какого-то тенора отобрано 0 фичей — использовать fallback: `["{tenor}_lag1"]`

### 3.4. Важные детали

- `StandardScaler` обучается (`fit`) только на обучающей выборке
- `LassoCV` использует внутреннюю кросс-валидацию (5 фолдов) для выбора alpha
- Lasso не используется как финальная модель — только для отбора фичей

---

## Блок 4. Обучение модели

### 4.1. Компонента 1: Ridge Regression (per-tenor)

```
Для каждой модели (M1, M2):
    X = полная матрица фичей (до отбора)
    Удалить NaN

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    ridge_models = {}  # {тенор: обученная Ridge-модель}

    Для каждого тенора из YIELD_TENORS:
        features = selected_features[тенор]  # из Блока 3
        X_subset = X_train_scaled[:, индексы отобранных фичей]

        # Подбор alpha через CV
        best_alpha = подобрать_alpha_cv(X_subset, y_train[тенор])

        model = Ridge(alpha=best_alpha)
        model.fit(X_subset, y_train[тенор])
        ridge_models[тенор] = model
```

**Подбор alpha для Ridge:**

```python
def find_best_ridge_alpha(X, y, alphas=np.logspace(-3, 3, 50)):
    tscv = TimeSeriesSplit(n_splits=5)
    best_alpha = 1.0
    best_score = np.inf

    for alpha in alphas:
        scores = []
        for train_idx, val_idx in tscv.split(X):
            model = Ridge(alpha=alpha)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            scores.append(np.mean((y[val_idx] - pred) ** 2))
        mean_score = np.mean(scores)
        if mean_score < best_score:
            best_score = mean_score
            best_alpha = alpha

    return best_alpha
```

- Перебираем 50 значений alpha от 0.001 до 1000
- Используем TimeSeriesSplit (не обычный KFold!)
- Критерий: MSE на валидационных фолдах

### 4.2. Компонента 2: VAR (Vector Autoregression)

VAR прогнозирует все 9 теноров одновременно с учётом их взаимных лагированных влияний.

```python
# Обучение VAR
yield_train = yield_df.loc[TRAIN_START:TRAIN_END]  # ~73 строки × 9 столбцов (до 2025-03)
var_model = VAR(yield_train)

# Автоматический подбор лага по информационному критерию AIC
lag_order_results = var_model.select_order(maxlags=6)
best_lag = lag_order_results.aic  # число (например, 2)

# Обучение с выбранным лагом
var_fitted = var_model.fit(best_lag)
```

**Детали VAR:**
- Вход: сырая матрица ~73×9 (без feature engineering, без масштабирования, данные до 2025-03)
- Максимальный лаг: 6 (не больше, при 78 строках и 9 переменных VAR(6) уже имеет ~330 параметров)
- Подбор лага: AIC автоматически балансирует сложность и точность
- VAR не использует фичи из IV — он работает только с сырой кривой

### 4.3. VAR для M2

VAR остаётся **одинаковым** для M1 и M2. По условию задачи, M2 отличается от M1 **только добавлением IV-фичей**. VAR не принимает экзогенные фичи в стандартной формулировке (для этого есть VARX, но при 78 строках это overfitting). Поэтому:

- Ridge M1 использует yield-фичи → Ridge M2 использует yield + IV-фичи (ОТЛИЧИЕ)
- VAR M1 = VAR M2 (одинаковый)
- Разница M1 vs M2 обеспечивается только через Ridge-компоненту

---

## Блок 5. Рекурсивный прогноз

### 5.1. Общая логика

Прогноз на 6 месяцев выполняется **пошагово**: прогнозируем октябрь → используем его для прогноза ноября → и т.д. На каждом шаге оба компонента (Ridge и VAR) дают свой прогноз, затем они усредняются с весами.

### 5.2. Прогноз Ridge

```
df_extended = yield_df.copy()  # начинаем с полных исторических данных

Для step = 0, 1, ..., 5 (6 шагов):
    1. Пересоздать FeatureEngineer(df_extended [, iv_df])
    2. Сгенерировать фичи
    3. Взять ПОСЛЕДНЮЮ строку фичей (iloc[-1])
    4. Оставить только отобранные фичи (selected_features[тенор])
    5. Масштабировать с СОХРАНЁННЫМ scaler (transform, НЕ fit_transform)
    6. Предсказать 9 значений (по одному Ridge на тенор)
    7. Добавить прогноз как новую строку в df_extended
    8. Перейти к step + 1
```

**КРИТИЧЕСКИЕ ДЕТАЛИ:**
- Scaler (StandardScaler) обучен на train. На каждом шаге прогноза используется `scaler.transform()`, а не `fit_transform()`. Иначе — утечка данных
- Фичи пересчитываются на каждом шаге, потому что лаги и MA зависят от предыдущих прогнозов
- IV-фичи: при рекурсивном прогнозе у нас нет IV за тестовые месяцы. Решение: IV-фичи «замораживаются» на последних доступных значениях. NaN заполняются через `fillna(method='ffill')` или `fillna(0)`

### 5.3. Прогноз VAR

```python
# VAR имеет встроенный метод forecast
last_observations = yield_train.values[-best_lag:]  # последние p строк (где p = лаг VAR)
var_forecast = var_fitted.forecast(last_observations, steps=6)
# var_forecast: массив 6×9
```

- VAR прогнозирует все 6 шагов за один вызов (не рекурсивно вручную — statsmodels делает рекурсию внутри)
- Результат: массив (6, 9) — 6 месяцев × 9 теноров

### 5.4. Ансамблирование

```python
# ridge_forecast: массив (6, 9) — результат рекурсивного Ridge-прогноза
# var_forecast:   массив (6, 9) — результат VAR-прогноза
# w: вес Ridge-компоненты (0 < w < 1)

final_forecast = w * ridge_forecast + (1 - w) * var_forecast
```

**Подбор веса w:**

```
Выполняется на последнем фолде TimeSeriesSplit обучающей выборки:

1. Разделить train на sub_train (первые ~60 строк) и sub_val (последние ~12 строк)
2. Обучить Ridge и VAR на sub_train
3. Получить прогнозы Ridge и VAR на sub_val
4. Перебрать w от 0.0 до 1.0 с шагом 0.05
5. Для каждого w посчитать weighted_rmse(y_true, w * ridge_pred + (1-w) * var_pred)
6. Выбрать w с минимальным RMSE

Типичный результат: w ≈ 0.5–0.7 (Ridge обычно сильнее VAR на малых выборках)
```

**ВАЖНО:** Вес w подбирается **одинаковый для всех теноров**. Не нужно подбирать 9 отдельных весов — при 12 точках валидации это overfitting.

---

## Блок 6. Метрика качества

### 6.1. Взвешенный RMSE (по условию задачи)

```python
def weighted_rmse(y_true, y_pred):
    """
    y_true, y_pred: массивы shape (T, 9)
    T = 6 (тестовых месяцев)
    9 столбцов в порядке YIELD_TENORS

    Веса:
        O/N (столбец 0): 0.4
        Остальные 8 (столбцы 1-8): 0.075 каждый
        Сумма весов: 0.4 + 8 × 0.075 = 1.0
    """
    T = y_true.shape[0]
    weights = np.array([0.4, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075])

    sq_errors = (y_true - y_pred) ** 2                    # (T, 9)
    weighted_sq_errors = weights[np.newaxis, :] * sq_errors  # broadcasting (1,9) × (T,9)
    rmse = np.sqrt(np.sum(weighted_sq_errors) / T)

    return rmse
```

### 6.2. Финальная метрика

```python
rmse_total = 0.5 * rmse_m1 + 0.5 * rmse_m2
```

Обе модели вносят одинаковый вклад. Оптимизировать нужно обе.

---

## Блок 7. Кросс-валидация

### 7.1. Оценка качества модели

Используем `TimeSeriesSplit(n_splits=5)`. Обычный KFold запрещён — он нарушает временной порядок.

```
Для каждой модели (M1, M2):
    Для каждого фолда (train_idx, val_idx) в TimeSeriesSplit:
        1. Обучить scaler на train_idx
        2. Отобрать фичи Lasso на train_idx
        3. Обучить Ridge на train_idx (с отобранными фичами)
        4. Обучить VAR на train_idx
        5. Подобрать вес ансамбля на внутренней валидации
        6. Предсказать на val_idx
        7. Посчитать weighted_rmse

    Вернуть список RMSE по фолдам и среднее
```

**ВАЖНО:** Отбор фичей Lasso должен быть **внутри** кросс-валидации, а не снаружи. Если отобрать фичи на всех данных, а потом валидировать — это утечка информации. Правильный pipeline:

```
[  train fold  ] → Lasso отбор → Ridge обучение → [ val fold ] → RMSE
```

А не:

```
[ все данные ] → Lasso отбор → [train fold] → Ridge обучение → [val fold] → RMSE  ← НЕПРАВИЛЬНО
```

---

## Блок 8. Валидация на out-of-sample данных (2025-04 — 2025-09)

### 8.1. Логика валидации

Модели обучаются на данных до 2025-03 включительно. Затем строится рекурсивный прогноз на 6 месяцев вперёд (2025-04 — 2025-09). Поскольку фактические данные за этот период доступны, можно вычислить точность модели.

### 8.2. Дельта-поверхность

Для каждой модели (M1, M2) строится **поверхность дельты** — разность между спрогнозированной кривой доходности и фактическими значениями:

```python
delta_m1 = pred_val_m1 - actual_val   # DataFrame (6, 9): разница M1-прогноз vs факт
delta_m2 = pred_val_m2 - actual_val   # DataFrame (6, 9): разница M2-прогноз vs факт
```

- Ось X: теноры (O/N, 1W, 2W, 1M, 2M, 3M, 6M, 1Y, 2Y)
- Ось Y: даты (2025-04, 2025-05, ..., 2025-09)
- Ось Z (цвет): величина дельты (положительная = переоценка, отрицательная = недооценка)

### 8.3. Визуализация

Строятся два 3D-графика (или heatmap) рядом: дельта M1 и дельта M2. Это позволяет визуально оценить, где модели ошибаются и добавляет ли IV информативность.

```python
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Heatmap дельты M1
sns.heatmap(delta_m1, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=axes[0])
axes[0].set_title('Дельта M1 (прогноз − факт)')

# Heatmap дельты M2
sns.heatmap(delta_m2, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=axes[1])
axes[1].set_title('Дельта M2 (прогноз − факт)')
```

### 8.4. Метрики валидации

```python
val_rmse_m1 = weighted_rmse(actual_val.values, pred_val_m1.values)
val_rmse_m2 = weighted_rmse(actual_val.values, pred_val_m2.values)
print(f"Validation RMSE M1: {val_rmse_m1:.4f}")
print(f"Validation RMSE M2: {val_rmse_m2:.4f}")
```

---

## Блок 9. Сохранение результатов

### 8.1. Формат выходного файла

```python
# pred_m1: DataFrame (6, 9) — прогноз M1
# pred_m2: DataFrame (6, 9) — прогноз M2
# Индекс: даты [2025-10, 2025-11, 2025-12, 2026-01, 2026-02, 2026-03]
# Столбцы: YIELD_TENORS

with pd.ExcelWriter('Problem_1_yield_curve_predict.xlsx', engine='openpyxl') as writer:
    pred_m1.to_excel(writer, sheet_name='M1', index_label='date')
    pred_m2.to_excel(writer, sheet_name='M2', index_label='date')
```

### 8.2. Проверки валидности (перед сохранением)

```python
# 1. Нет NaN
assert pred_m1.isnull().sum().sum() == 0, "M1 содержит NaN!"
assert pred_m2.isnull().sum().sum() == 0, "M2 содержит NaN!"

# 2. Размеры правильные
assert pred_m1.shape == (6, 9), f"M1 shape {pred_m1.shape}, ожидалось (6, 9)"
assert pred_m2.shape == (6, 9), f"M2 shape {pred_m2.shape}, ожидалось (6, 9)"

# 3. Даты правильные
expected_dates = pd.date_range('2025-10', periods=6, freq='MS')
# (проверить, что индекс pred_m1 соответствует expected_dates)

# 4. Значения в разумном диапазоне
# (не обязательно — данные анонимизированы, диапазон неизвестен)
```

---

## Блок 10. Основной скрипт (main)

### 10.1. Последовательность вызовов

```
main():
    # --- ЗАГРУЗКА ---
    yield_df_full = загрузить_кривую()           # все данные 2019-03 ... 2025-09
    iv_df_full = загрузить_iv()                   # все данные 2019-03 ... 2025-09

    yield_train = yield_df_full.loc[:TRAIN_END]   # обучение: до 2025-03
    yield_val   = yield_df_full.loc[VAL_START:VAL_END]  # факт: 2025-04 ... 2025-09
    iv_train    = iv_df_full[iv_df_full['Date'] <= TRAIN_END]

    # --- FEATURE ENGINEERING ---
    X_m1 = make_yield_features(yield_train)
    X_m2 = concat(make_yield_features(yield_train), make_iv_features(iv_train))
    Удалить NaN, выровнять индексы

    # --- ОТБОР ФИЧЕЙ (на обучающей выборке до 2025-03) ---
    selected_m1 = lasso_select(X_m1, yield_train)
    selected_m2 = lasso_select(X_m2, yield_train)

    # --- ОБУЧЕНИЕ ФИНАЛЬНЫХ МОДЕЛЕЙ ---
    ridge_models_m1, scaler_m1 = train_ridge(X_m1, yield_train, selected_m1)
    ridge_models_m2, scaler_m2 = train_ridge(X_m2, yield_train, selected_m2)
    var_fitted, best_lag = train_var(yield_train)

    # --- ПОДБОР ВЕСОВ АНСАМБЛЯ ---
    w_m1 = find_ensemble_weight(...)
    w_m2 = find_ensemble_weight(...)

    # --- ВАЛИДАЦИОННЫЙ ПРОГНОЗ (2025-04 ... 2025-09) ---
    pred_val_m1 = recursive_predict(ridge_models_m1, var_fitted, w_m1, yield_train, n_steps=6)
    pred_val_m2 = recursive_predict(ridge_models_m2, var_fitted, w_m2, yield_train, n_steps=6, iv_df=iv_train)

    # --- ДЕЛЬТА-ПОВЕРХНОСТЬ ---
    delta_m1 = pred_val_m1 - yield_val   # разница прогноз vs факт
    delta_m2 = pred_val_m2 - yield_val
    plot_delta_surfaces(delta_m1, delta_m2)
    print(validation RMSE M1, M2)

    # --- ФИНАЛЬНЫЙ ПРОГНОЗ (2025-10 ... 2026-03) ---
    # Дообучение на полных данных (до 2025-09) для финального прогноза
    pred_m1 = recursive_predict(...)   # 6 шагов вперёд
    pred_m2 = recursive_predict(...)

    # --- СОХРАНЕНИЕ ---
    save_predictions(pred_m1, pred_m2)
    save_delta_surfaces(delta_m1, delta_m2)
```

### 9.2. Формат вывода в консоль

```
======================================================================
PIPELINE: Ridge + VAR ансамбль
======================================================================

[1/8] Загрузка данных...
  Кривая доходности: (78, 9)
  Поверхность IV:    (78, 15)

[2/8] Feature engineering...
  M1: 79 фичей
  M2: 132 фичи (79 yield + 53 IV)

[3/8] Кросс-валидация...
  M1 CV RMSE: [0.18, 0.15, 0.12, 0.09, 0.07]  среднее = 0.122
  M2 CV RMSE: [0.16, 0.13, 0.10, 0.08, 0.06]  среднее = 0.106

[4/8] Отбор фичей (Lasso)...
  M1: O/N=7, 1W=4, 2W=4, 1M=4, 2M=4, 3M=6, 6M=4, 1Y=5, 2Y=7
  M2: O/N=10, 1W=5, 2W=6, 1M=5, 2M=5, 3M=8, 6M=6, 1Y=12, 2Y=8

[5/8] Обучение Ridge...
  M1: alpha по тенорам: {O/N: 1.2, 1W: 0.8, ...}
  M2: alpha по тенорам: {O/N: 0.6, 1W: 0.5, ...}

[6/8] Обучение VAR...
  Лаг по AIC: 2
  Параметров VAR: 171

[7/8] Подбор весов ансамбля...
  M1: w_ridge = 0.65, w_var = 0.35
  M2: w_ridge = 0.60, w_var = 0.40

[8/8] Прогноз и сохранение...
  Файл: Problem_1_yield_curve_predict.xlsx

======================================================================
ИТОГ
======================================================================
  CV RMSE M1:           0.1220
  CV RMSE M2:           0.1060
  Улучшение M2 vs M1:  +13.1%
  → IV добавляет прогностическую ценность
======================================================================
```

---

## Возможные ошибки и их решение

### Ошибка 1: Singular matrix в VAR
**Причина:** Мультиколлинеарность — некоторые теноры почти линейно зависимы.
**Решение:** Уменьшить лаг VAR (maxlags=3 вместо 6) или удалить один из коррелированных теноров (например, 2W, если он сильно коррелирует с 1W).

### Ошибка 2: Lasso отбирает 0 фичей
**Причина:** Слишком сильная регуляризация (alpha слишком большой).
**Решение:** Расширить диапазон alpha в LassoCV: `alphas=np.logspace(-6, 0, 150)`. Fallback: если 0 фичей — использовать `["{tenor}_lag1"]`.

### Ошибка 3: Рекурсивный прогноз расходится
**Причина:** Накопление ошибок, прогнозы уходят за пределы обучающего диапазона.
**Решение:** Добавить clipping: `pred = np.clip(pred, min_train_value, max_train_value)`. Значения не должны выходить за диапазон, наблюдавшийся в обучении (±10% запаса).

### Ошибка 4: NaN в прогнозе M2
**Причина:** IV-фичи при рекурсивном прогнозе содержат NaN (нет IV за тестовые месяцы).
**Решение:** Перед масштабированием: `X_last = X_last.fillna(method='ffill').fillna(0)`. Forward fill сначала, нули — как последний fallback.

### Ошибка 5: StandardScaler выдаёт предупреждение о division by zero
**Причина:** Столбец с нулевой дисперсией (например, все значения одинаковы).
**Решение:** Удалить такие столбцы перед масштабированием: `X = X.loc[:, X.std() > 1e-10]`.

---

## Структура файлов проекта

```
project/
├── yield_curve_filled.xlsx          # Входные данные: кривая доходности
├── iv_surface.xlsx                  # Входные данные: поверхность IV
├── pipeline.py                      # Основной скрипт (создаётся по этой спецификации)
├── Problem_1_yield_curve_predict.xlsx  # Выходной файл с прогнозами
└── selected_features.csv            # Опционально: таблица отобранных фичей
```

---

## Чеклист перед отправкой

- [ ] `Problem_1_yield_curve_predict.xlsx` содержит 2 вкладки: M1 и M2
- [ ] Каждая вкладка: 6 строк × 9 столбцов, без NaN
- [ ] Даты: 2025-10, 2025-11, 2025-12, 2026-01, 2026-02, 2026-03
- [ ] Столбцы: O/N, 1W, 2W, 1M, 2M, 3M, 6M, 1Y, 2Y
- [ ] M2 отличается от M1 только добавлением IV-фичей (не новых источников данных)
- [ ] Отбор фичей выполнен внутри кросс-валидации (нет утечки)
- [ ] Все IV-фичи сдвинуты на shift(1) — нет look-ahead bias
- [ ] StandardScaler обучен только на train, на test используется transform
- [ ] VAR лаг ≤ 6, подобран по AIC
- [ ] Вес ансамбля подобран на валидационной выборке, а не на тесте