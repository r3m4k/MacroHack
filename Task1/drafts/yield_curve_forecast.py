"""
================================================================================
ПРОГНОЗИРОВАНИЕ КРИВОЙ ДОХОДНОСТИ (RUONIA + ROISfix)
================================================================================
Задание 1: Модель M1 (без IV) и M2 (с фичами из поверхности вменённой волатильности)

Стек: Python — sklearn, statsmodels, pandas, numpy
Автор: Аналитик ЦБ РФ
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
# from statsmodels.tsa.api import VAR  # раскомментируй при наличии statsmodels
import warnings
warnings.filterwarnings("ignore")


# ==============================================================================
# 0. КОНФИГУРАЦИЯ
# ==============================================================================

# Теноры целевой кривой доходности (9 штук)
YIELD_TENORS = ["O/N", "1W", "2W", "1M", "2M", "3M", "6M", "1Y", "2Y"]

# Теноры поверхности IV (15 штук)
IV_TENORS = ["1M", "2M", "3M", "6M", "9M", "1Y", "2Y", "3Y", "4Y",
             "5Y", "6Y", "7Y", "8Y", "9Y", "10Y"]

# Периоды
TRAIN_START = "2019-03"
TRAIN_END = "2025-09"
TEST_START = "2025-10"
TEST_END = "2026-03"

# Веса для метрики RMSE
WEIGHT_ON = 0.4                                 # вес O/N
WEIGHT_OTHER = 0.6 / (len(YIELD_TENORS) - 1)    # вес каждого из остальных 8 теноров


# ==============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ==============================================================================

def load_yield_curve(filepath: str) -> pd.DataFrame:
    """
    Загрузка рублёвой кривой спотовой доходности.

    Ожидаемый формат CSV/Excel:
        date | O/N | 1W | 2W | 1M | 2M | 3M | 6M | 1Y | 2Y
        2019-03 | ... | ... | ...

    Returns:
        pd.DataFrame с DatetimeIndex (monthly) и столбцами = YIELD_TENORS
    """
    # --- ПОДСТАВЬ СВОЙ КОД ЗАГРУЗКИ ---
    # df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    # df = pd.read_excel(filepath, index_col=0)

    # Заглушка: генерируем синтетические данные для тестирования pipeline
    dates = pd.date_range(TRAIN_START, TRAIN_END, freq="MS")
    np.random.seed(42)
    base = np.linspace(7.0, 9.0, len(YIELD_TENORS))  # восходящая кривая
    data = np.tile(base, (len(dates), 1)) + np.random.randn(len(dates), len(YIELD_TENORS)) * 0.5
    df = pd.DataFrame(data, index=dates, columns=YIELD_TENORS)
    return df


def load_iv_surface(filepath: str) -> pd.DataFrame:
    """
    Загрузка поверхности вменённой волатильности.

    Ожидаемый формат: мультииндексный DataFrame или wide-формат.
    Столбцы: (tenor, strike) — значения σ(t, τ, K).

    Для простоты предполагаем, что данные агрегированы по страйкам
    (например, ATM IV или среднее по доступным страйкам).

    Returns:
        pd.DataFrame с DatetimeIndex и столбцами = IV_TENORS
    """
    # --- ПОДСТАВЬ СВОЙ КОД ЗАГРУЗКИ ---
    # Заглушка
    dates = pd.date_range(TRAIN_START, TRAIN_END, freq="MS")
    np.random.seed(123)
    data = np.random.randn(len(dates), len(IV_TENORS)) * 2 + 15
    df = pd.DataFrame(data, index=dates, columns=IV_TENORS)
    return df


# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================

class FeatureEngineer:
    """Построение признаков для M1 и M2."""

    def __init__(self, yield_df: pd.DataFrame, iv_df: pd.DataFrame = None):
        self.yield_df = yield_df.copy()
        self.iv_df = iv_df.copy() if iv_df is not None else None

    # ---------- Фичи из кривой доходности (для M1 и M2) ----------

    def yield_curve_features(self) -> pd.DataFrame:
        """
        Признаки, извлечённые из кривой доходности:
        1. Лагированные значения (lag 1, 2, 3 месяца)
        2. Спреды между тенорами (наклон кривой)
        3. Скользящие средние (3м, 6м)
        4. Изменения (дельты) месяц к месяцу
        5. Кривизна (curvature): 2×mid - short - long
        6. PCA-компоненты кривой (level, slope, curvature)
        """
        df = self.yield_df.copy()
        features = pd.DataFrame(index=df.index)

        # --- 2.1 Лагированные значения ---
        for lag in [1, 2, 3]:
            lagged = df.shift(lag)
            lagged.columns = [f"{col}_lag{lag}" for col in df.columns]
            features = pd.concat([features, lagged], axis=1)

        # --- 2.2 Спреды (наклон кривой) ---
        features["spread_2Y_ON"] = df["2Y"] - df["O/N"]       # полный наклон
        features["spread_1Y_ON"] = df["1Y"] - df["O/N"]       # короткий-длинный
        features["spread_2Y_1Y"] = df["2Y"] - df["1Y"]        # длинный конец
        features["spread_6M_3M"] = df["6M"] - df["3M"]        # середина
        features["spread_3M_1M"] = df["3M"] - df["1M"]        # короткий конец

        # Лагированные спреды
        for lag in [1, 2]:
            for col in ["spread_2Y_ON", "spread_1Y_ON", "spread_2Y_1Y"]:
                features[f"{col}_lag{lag}"] = features[col].shift(lag)

        # --- 2.3 Скользящие средние ---
        for window in [3, 6]:
            rolled = df.rolling(window).mean()
            rolled.columns = [f"{col}_ma{window}" for col in df.columns]
            features = pd.concat([features, rolled], axis=1)

        # --- 2.4 Дельты (месячные изменения) ---
        delta1 = df.diff(1)
        delta1.columns = [f"{col}_delta1" for col in df.columns]
        features = pd.concat([features, delta1], axis=1)

        delta3 = df.diff(3)
        delta3.columns = [f"{col}_delta3" for col in df.columns]
        features = pd.concat([features, delta3], axis=1)

        # --- 2.5 Кривизна ---
        # curvature = 2 * средний тенор - короткий - длинный
        features["curvature_1"] = 2 * df["1M"] - df["O/N"] - df["6M"]
        features["curvature_2"] = 2 * df["1Y"] - df["3M"] - df["2Y"]

        # --- 2.6 PCA-компоненты кривой ---
        pca = PCA(n_components=3)
        pca_vals = pca.fit_transform(df.values)
        features["pca_level"] = pca_vals[:, 0]
        features["pca_slope"] = pca_vals[:, 1]
        features["pca_curvature"] = pca_vals[:, 2]

        return features

    # ---------- Фичи из поверхности IV (только для M2) ----------

    def iv_surface_features(self) -> pd.DataFrame:
        """
        Признаки из поверхности вменённой волатильности:
        1. ATM implied volatility по каждому тенору (или среднее по страйкам)
        2. Термструктура IV: спреды между тенорами IV
        3. PCA-компоненты поверхности IV
        4. Скользящие средние и дельты IV
        5. IV skew и term spread
        """
        if self.iv_df is None:
            raise ValueError("IV data not provided")

        iv = self.iv_df.copy()
        features = pd.DataFrame(index=iv.index)

        # --- 3.1 Текущие значения IV (лагированные, чтобы не было look-ahead) ---
        iv_lagged = iv.shift(1)
        iv_lagged.columns = [f"iv_{col}_lag1" for col in iv.columns]
        features = pd.concat([features, iv_lagged], axis=1)

        # --- 3.2 Термструктура IV (спреды) ---
        features["iv_spread_10Y_1M"] = iv["10Y"].shift(1) - iv["1M"].shift(1)
        features["iv_spread_5Y_1Y"] = iv["5Y"].shift(1) - iv["1Y"].shift(1)
        features["iv_spread_2Y_6M"] = iv["2Y"].shift(1) - iv["6M"].shift(1)

        # --- 3.3 Дельты IV ---
        iv_delta = iv.diff(1).shift(1)
        iv_delta.columns = [f"iv_{col}_delta1" for col in iv.columns]
        features = pd.concat([features, iv_delta], axis=1)

        # --- 3.4 Скользящие средние IV ---
        iv_ma3 = iv.rolling(3).mean().shift(1)
        iv_ma3.columns = [f"iv_{col}_ma3" for col in iv.columns]
        features = pd.concat([features, iv_ma3], axis=1)

        # --- 3.5 PCA-компоненты IV ---
        iv_clean = iv.dropna()
        if len(iv_clean) > 5:
            pca_iv = PCA(n_components=3)
            pca_vals = pca_iv.fit_transform(iv_clean.values)
            pca_df = pd.DataFrame(
                pca_vals,
                index=iv_clean.index,
                columns=["iv_pca_1", "iv_pca_2", "iv_pca_3"]
            )
            # Лагируем, чтобы избежать look-ahead bias
            pca_df = pca_df.shift(1)
            features = features.join(pca_df, how="left")

        # --- 3.6 Волатильность волатильности (vol of vol) ---
        iv_std3 = iv.rolling(3).std().shift(1)
        features["iv_vol_of_vol_short"] = iv_std3.mean(axis=1)
        iv_std6 = iv.rolling(6).std().shift(1)
        features["iv_vol_of_vol_long"] = iv_std6.mean(axis=1)

        return features


# ==============================================================================
# 3. МЕТРИКА КАЧЕСТВА
# ==============================================================================

def weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Взвешенный RMSE по формуле из условия задачи.

    y_true, y_pred: shape (T, 9) — строки = месяцы, столбцы = теноры (O/N, 1W, ..., 2Y)

    Веса:
      - O/N: 0.4
      - остальные 8 теноров: 0.6/8 = 0.075 каждый
    """
    T = y_true.shape[0]
    weights = np.array([WEIGHT_ON] + [WEIGHT_OTHER] * (len(YIELD_TENORS) - 1))

    sq_errors = (y_true - y_pred) ** 2  # (T, 9)
    weighted_sum = np.sum(weights[np.newaxis, :] * sq_errors)
    rmse = np.sqrt(weighted_sum / T)
    return rmse


def rmse_total(rmse_m1: float, rmse_m2: float, delta: float = 0.5) -> float:
    """Финальная метрика: RMSE_total = δ * RMSE^M1 + (1-δ) * RMSE^M2."""
    return delta * rmse_m1 + (1 - delta) * rmse_m2


# ==============================================================================
# 4. МОДЕЛЬ M1 — без IV
# ==============================================================================

class YieldCurveModelM1:
    """
    Модель прогнозирования кривой доходности без поверхности IV.

    Стратегия: MultiOutput Ridge Regression.
    Каждый тенор прогнозируется отдельной Ridge-моделью (или одной MultiOutput).
    Альтернатива: VAR-модель из statsmodels.
    """

    def __init__(self, method: str = "ridge"):
        """
        method: 'ridge' | 'var' | 'elasticnet'
        """
        self.method = method
        self.models = {}
        self.scalers = {}
        self.var_model = None
        self.feature_engineer = None

    def fit(self, yield_df: pd.DataFrame):
        """Обучение модели M1."""
        self.feature_engineer = FeatureEngineer(yield_df)
        X = self.feature_engineer.yield_curve_features()
        y = yield_df.copy()

        # Выровняем индексы и уберём NaN
        common_idx = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        if self.method == "var":
            self._fit_var(yield_df)
        else:
            self._fit_sklearn(X, y)

    def _fit_sklearn(self, X: pd.DataFrame, y: pd.DataFrame):
        """Обучение отдельной модели на каждый тенор."""
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        self.scalers["X"] = scaler_X

        for tenor in YIELD_TENORS:
            if self.method == "ridge":
                model = Ridge(alpha=1.0)
            elif self.method == "elasticnet":
                model = ElasticNet(alpha=0.1, l1_ratio=0.5)
            else:
                model = Ridge(alpha=1.0)

            model.fit(X_scaled, y[tenor].values)
            self.models[tenor] = model

        # Сохраняем данные для последующего прогноза
        self._last_yield_df = y
        self._last_X = X

    def _fit_var(self, yield_df: pd.DataFrame):
        """Обучение VAR-модели."""
        df_clean = yield_df.dropna()
        self.var_model = VAR(df_clean)
        # Выбор лага по AIC
        results = self.var_model.select_order(maxlags=6)
        best_lag = results.aic
        self.var_fitted = self.var_model.fit(best_lag)
        print(f"VAR: выбран лаг = {best_lag} по AIC")

    def predict(self, yield_df_full: pd.DataFrame, n_steps: int = 6) -> pd.DataFrame:
        """
        Прогноз на n_steps месяцев вперёд.

        Для sklearn: рекурсивный прогноз (подставляем прогноз как вход).
        Для VAR: встроенный forecast.
        """
        if self.method == "var":
            return self._predict_var(yield_df_full, n_steps)
        else:
            return self._predict_sklearn(yield_df_full, n_steps)

    def _predict_sklearn(self, yield_df_full: pd.DataFrame,
                         n_steps: int = 6) -> pd.DataFrame:
        """Рекурсивный прогноз для sklearn-моделей."""
        # Работаем с копией, чтобы не мутировать оригинал
        df = yield_df_full.copy()
        predictions = []
        pred_dates = pd.date_range(
            start=pd.Period(TRAIN_END, freq="M").end_time + pd.Timedelta(days=1),
            periods=n_steps,
            freq="MS"
        )

        for step in range(n_steps):
            fe = FeatureEngineer(df)
            X_new = fe.yield_curve_features()

            # Берём последнюю строку фичей
            X_last = X_new.iloc[[-1]]
            X_scaled = self.scalers["X"].transform(X_last)

            pred_row = {}
            for tenor in YIELD_TENORS:
                pred_row[tenor] = self.models[tenor].predict(X_scaled)[0]

            predictions.append(pred_row)

            # Добавляем прогноз в DataFrame для рекурсии
            new_row = pd.DataFrame(pred_row, index=[pred_dates[step]])
            df = pd.concat([df, new_row])

        return pd.DataFrame(predictions, index=pred_dates, columns=YIELD_TENORS)

    def _predict_var(self, yield_df_full: pd.DataFrame,
                     n_steps: int = 6) -> pd.DataFrame:
        """Прогноз VAR-модели."""
        forecast = self.var_fitted.forecast(
            yield_df_full.dropna().values[-self.var_fitted.k_ar:],
            steps=n_steps
        )
        pred_dates = pd.date_range(
            start=pd.Period(TRAIN_END, freq="M").end_time + pd.Timedelta(days=1),
            periods=n_steps,
            freq="MS"
        )
        return pd.DataFrame(forecast, index=pred_dates, columns=YIELD_TENORS)


# ==============================================================================
# 5. МОДЕЛЬ M2 — M1 + фичи из IV
# ==============================================================================

class YieldCurveModelM2(YieldCurveModelM1):
    """
    Модель M2: всё то же, что M1, плюс признаки из поверхности IV.
    Допускается перенастройка гиперпараметров.
    НЕ допускается добавление новых источников данных.
    """

    def __init__(self, method: str = "ridge"):
        super().__init__(method=method)
        self.iv_feature_engineer = None

    def fit(self, yield_df: pd.DataFrame, iv_df: pd.DataFrame):
        """Обучение модели M2 с фичами из IV."""
        fe = FeatureEngineer(yield_df, iv_df)

        X_yield = fe.yield_curve_features()
        X_iv = fe.iv_surface_features()

        # Объединяем фичи
        X = pd.concat([X_yield, X_iv], axis=1)
        y = yield_df.copy()

        # Выровняем индексы
        common_idx = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        # Скалирование
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        self.scalers["X"] = scaler_X

        # Обучение (с перенастроенными гиперпараметрами)
        for tenor in YIELD_TENORS:
            if self.method == "ridge":
                model = Ridge(alpha=0.5)  # перенастроенный alpha
            elif self.method == "elasticnet":
                model = ElasticNet(alpha=0.05, l1_ratio=0.3)
            else:
                model = Ridge(alpha=0.5)

            model.fit(X_scaled, y[tenor].values)
            self.models[tenor] = model

        self._last_yield_df = y
        self._last_X = X
        self._iv_df = iv_df

    def predict(self, yield_df_full: pd.DataFrame, iv_df_full: pd.DataFrame,
                n_steps: int = 6) -> pd.DataFrame:
        """Рекурсивный прогноз M2 (с IV-фичами)."""
        df = yield_df_full.copy()
        iv = iv_df_full.copy()
        predictions = []
        pred_dates = pd.date_range(
            start=pd.Period(TRAIN_END, freq="M").end_time + pd.Timedelta(days=1),
            periods=n_steps,
            freq="MS"
        )

        for step in range(n_steps):
            fe = FeatureEngineer(df, iv)
            X_yield = fe.yield_curve_features()
            X_iv = fe.iv_surface_features()
            X_new = pd.concat([X_yield, X_iv], axis=1)

            X_last = X_new.iloc[[-1]]

            # Заполняем NaN нулями для фичей, которые не удалось вычислить
            X_last = X_last.fillna(0)
            X_scaled = self.scalers["X"].transform(X_last)

            pred_row = {}
            for tenor in YIELD_TENORS:
                pred_row[tenor] = self.models[tenor].predict(X_scaled)[0]

            predictions.append(pred_row)

            new_row = pd.DataFrame(pred_row, index=[pred_dates[step]])
            df = pd.concat([df, new_row])

        return pd.DataFrame(predictions, index=pred_dates, columns=YIELD_TENORS)


# ==============================================================================
# 6. КРОСС-ВАЛИДАЦИЯ (TimeSeriesSplit)
# ==============================================================================

def cross_validate_model(yield_df: pd.DataFrame, iv_df: pd.DataFrame = None,
                         model_type: str = "M1", n_splits: int = 5,
                         method: str = "ridge") -> list:
    """
    Кросс-валидация с TimeSeriesSplit.
    Возвращает список RMSE для каждого фолда.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores = []

    fe = FeatureEngineer(yield_df, iv_df)
    X_yield = fe.yield_curve_features()

    if model_type == "M2" and iv_df is not None:
        X_iv = fe.iv_surface_features()
        X = pd.concat([X_yield, X_iv], axis=1)
    else:
        X = X_yield

    y = yield_df.copy()
    common_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        y_pred = np.zeros_like(y_test.values)
        for j, tenor in enumerate(YIELD_TENORS):
            model = Ridge(alpha=1.0) if model_type == "M1" else Ridge(alpha=0.5)
            model.fit(X_train_s, y_train[tenor].values)
            y_pred[:, j] = model.predict(X_test_s)

        rmse = weighted_rmse(y_test.values, y_pred)
        rmse_scores.append(rmse)

    return rmse_scores


# ==============================================================================
# 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ==============================================================================

def save_predictions(pred_m1: pd.DataFrame, pred_m2: pd.DataFrame,
                     output_path: str = "Problem_1_yield_curve_predict.xlsx"):
    """
    Сохранение прогнозов в формате xlsx (шаблон из условия задачи).
    """
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pred_m1.to_excel(writer, sheet_name="M1", index_label="date")
        pred_m2.to_excel(writer, sheet_name="M2", index_label="date")
    print(f"Прогнозы сохранены в {output_path}")


# ==============================================================================
# 8. MAIN — ЗАПУСК PIPELINE
# ==============================================================================

def main():
    print("=" * 70)
    print("PIPELINE: Прогнозирование кривой доходности")
    print("=" * 70)

    # --- Загрузка данных ---
    print("\n[1/6] Загрузка данных...")
    yield_df = load_yield_curve("yield_curve.csv")      # <- подставь свой путь
    iv_df = load_iv_surface("iv_surface.csv")            # <- подставь свой путь
    print(f"  Кривая доходности: {yield_df.shape}")
    print(f"  Поверхность IV:    {iv_df.shape}")

    # --- Кросс-валидация ---
    print("\n[2/6] Кросс-валидация M1...")
    cv_m1 = cross_validate_model(yield_df, model_type="M1")
    print(f"  RMSE по фолдам: {[f'{x:.4f}' for x in cv_m1]}")
    print(f"  Средний RMSE:   {np.mean(cv_m1):.4f}")

    print("\n[3/6] Кросс-валидация M2...")
    cv_m2 = cross_validate_model(yield_df, iv_df, model_type="M2")
    print(f"  RMSE по фолдам: {[f'{x:.4f}' for x in cv_m2]}")
    print(f"  Средний RMSE:   {np.mean(cv_m2):.4f}")

    # --- Обучение финальных моделей ---
    print("\n[4/6] Обучение M1...")
    model_m1 = YieldCurveModelM1(method="ridge")
    model_m1.fit(yield_df)

    print("[5/6] Обучение M2...")
    model_m2 = YieldCurveModelM2(method="ridge")
    model_m2.fit(yield_df, iv_df)

    # --- Прогноз ---
    print("\n[6/6] Прогнозирование на тестовый период...")
    pred_m1 = model_m1.predict(yield_df, n_steps=6)
    pred_m2 = model_m2.predict(yield_df, iv_df, n_steps=6)

    print("\nПрогноз M1:")
    print(pred_m1.round(4))
    print("\nПрогноз M2:")
    print(pred_m2.round(4))

    # --- Сохранение ---
    save_predictions(pred_m1, pred_m2)

    # --- Итоговый отчёт ---
    print("\n" + "=" * 70)
    print("ИТОГ")
    print("=" * 70)
    improvement = (np.mean(cv_m1) - np.mean(cv_m2)) / np.mean(cv_m1) * 100
    print(f"  CV RMSE M1: {np.mean(cv_m1):.4f}")
    print(f"  CV RMSE M2: {np.mean(cv_m2):.4f}")
    print(f"  Улучшение M2 vs M1: {improvement:+.2f}%")
    if improvement > 0:
        print("  → IV добавляет прогностическую ценность!")
    else:
        print("  → IV не улучшает прогноз (нужна другая экстракция фичей)")


if __name__ == "__main__":
    main()
