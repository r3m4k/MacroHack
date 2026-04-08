# Индекс неопределённости ДКП (MPU) на основе вменённой волатильности процентных ставок
## Полное теоретическое руководство с Python-реализацией
### Применительно к российскому рынку (ОФЗ, RUONIA, ключевая ставка ЦБ РФ)

---

> **Навигация по документу**
> 1. [Теория RND/RWD и методы извлечения](#раздел-1-теория-rndрwd-и-методы-извлечения)
> 2. [Построение MPU индекса](#раздел-2-построение-mpu-индекса-пошаговое-руководство)
> 3. [Эконометрика: тесты и модели](#раздел-3-эконометрика-тесты-и-макромодели)
> 4. [Чеклист по 4 подзаданиям](#раздел-4-мастер-чеклист-по-всем-4-подзаданиям)
> 5. [Ключевые источники: разбор методологий](#раздел-5-ключевые-источники-разбор-методологий)

---

## РАЗДЕЛ 1. Теория RND/RWD и методы извлечения

### 1.1 Фундамент: что такое Risk-Neutral Distribution (RND)?

**Экономический смысл.** Цена любого дериватива в безарбитражном мире равна ожидаемой выплате под риск-нейтральной мерой $\mathbb{Q}$, дисконтированной по безрисковой ставке:

$$C(K, T) = e^{-rT} \int_0^{\infty} \max(S_T - K, 0)\, q(S_T)\, dS_T$$

где $q(S_T)$ — риск-нейтральная плотность (RND). Это **не** реальное распределение, а взвешенное по ценообразующему ядру $m(S_T)$:

$$q(S_T) = m(S_T) \cdot p(S_T)$$

Для процентных ставок аналогичная логика: «базовый актив» — будущая ставка $r_T$ (например, RUONIA за 3 месяца), колл-опцион (кэп) имеет выплату $\max(r_T - K, 0)$.

---

### 1.2 Формула Бридена–Литценбергера (Breeden & Litzenberger, 1978)

**Центральный результат для извлечения RND.**

Для европейского колл-опциона на актив $S$ со страйком $K$ и погашением $T$:

$$\boxed{q(K) = e^{rT} \frac{\partial^2 C(K, T)}{\partial K^2}}$$

**Вывод:**

1. Начнём с формулы цены колл-опциона:
$$C(K) = e^{-rT} \int_K^{\infty} (S_T - K)\, q(S_T)\, dS_T$$

2. Первая производная по $K$:
$$\frac{\partial C}{\partial K} = -e^{-rT} \int_K^{\infty} q(S_T)\, dS_T = -e^{-rT}(1 - F(K))$$

где $F(K)$ — функция распределения $S_T$ под $\mathbb{Q}$.

3. Вторая производная:
$$\frac{\partial^2 C}{\partial K^2} = e^{-rT} q(K)$$

Откуда немедленно следует $q(K) = e^{rT} \frac{\partial^2 C}{\partial K^2}$.

**Практические следствия:**
- Нисходящий наклон улыбки волатильности → отрицательная асимметрия RND
- «Толстые» хвосты улыбки → эксцесс > 0 в RND
- Для российских процентных деривативов: вместо $C(K)$ используем цены **кэплетов** (caplets) на RUONIA

**Для процентных ставок: модель Блэка (Black's model)**

Цена кэплета:
$$\text{Caplet}(K, T) = e^{-rT} \tau \left[F \Phi(d_1) - K \Phi(d_2)\right]$$

$$d_1 = \frac{\ln(F/K) + \frac{1}{2}\sigma^2 T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

где $F$ — форвардная ставка, $\tau$ — длина периода (например, 0.25 для квартального кэплета), $\sigma$ — вменённая волатильность (flat Black vol).

**RND ставки через формулу Блэка:** применяем Бридена–Литценбергера к ценам кэплетов по страйку $K$.

---

### 1.3 Поверхность вменённой волатильности (IV Surface)

Структура данных для российского рынка (Московская биржа / OTC):

| Тенор (T) | Strike (K, % годовых) | IV (%, Black vol) |
|-----------|----------------------|-------------------|
| 3M        | 12                   | 18.5              |
| 3M        | 15 (ATM)             | 16.0              |
| 3M        | 18                   | 17.2              |
| 6M        | 12                   | 19.1              |
| ...       | ...                  | ...               |

**Метрики поверхности:**
- $\sigma_{ATM}(T)$ — ATM IV для каждого тенора
- **Risk Reversal (RR):** $RR_{25} = \sigma_{25\Delta\text{call}} - \sigma_{25\Delta\text{put}}$ — мера асимметрии (наклон улыбки)
- **Butterfly (BF):** $BF_{25} = \frac{1}{2}(\sigma_{25\Delta\text{call}} + \sigma_{25\Delta\text{put}}) - \sigma_{ATM}$ — мера «улыбки» (кривизна)
- **Term Structure Slope:** $\sigma_{ATM}(12M) - \sigma_{ATM}(3M)$ — наклон срочной структуры

---

### 1.4 Методы извлечения RND

#### Метод 1: Метод Шимко / Кубический сплайн (Shimko 1993 + Breeden-Litzenberger)

**Шаги:**
1. Наблюдаемые пары $(K_i, \sigma_i^{IV})$ для нескольких страйков
2. Кубический сплайн по $\sigma^{IV}(K)$
3. Перевод в цены колл-опционов через формулу Блэка: $C_i = C^{Black}(K_i, \sigma_i^{IV})$
4. Численная вторая производная $\frac{\partial^2 C}{\partial K^2}$
5. Нормализация плотности

**Python-реализация:**

```python
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import norm
import matplotlib.pyplot as plt

def black_call_price(F, K, T, sigma, r=0.0):
    """Цена кэплета по модели Блэка (упрощённо)."""
    if sigma <= 0 or T <= 0:
        return max(F - K, 0) * np.exp(-r * T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))

def extract_rnd_spline(strikes, ivols, F, T, r=0.0, n_points=200):
    """
    Извлечение RND через кубический сплайн + Breeden-Litzenberger.
    
    Parameters
    ----------
    strikes : array-like, страйки (% годовых или доли)
    ivols   : array-like, вменённые волатильности (доли, не %)
    F       : float, форвардная ставка
    T       : float, срок до экспирации (в годах)
    r       : float, безрисковая ставка для дисконтирования
    
    Returns
    -------
    K_fine : np.ndarray, сетка страйков
    rnd    : np.ndarray, RND плотность (нормализована)
    """
    strikes = np.array(strikes)
    ivols   = np.array(ivols)
    
    # Шаг 1: сплайн по волатильности
    cs = CubicSpline(strikes, ivols, bc_type='natural')
    
    # Шаг 2: тонкая сетка страйков
    K_min = strikes[0] * 0.7
    K_max = strikes[-1] * 1.3
    K_fine = np.linspace(K_min, K_max, n_points)
    
    # Шаг 3: IV и цены колл-опционов на тонкой сетке
    iv_fine = cs(K_fine)
    iv_fine = np.clip(iv_fine, 1e-4, 5.0)  # обрезаем нефизичные значения
    
    C_fine = np.array([black_call_price(F, K, T, sig, r) 
                       for K, sig in zip(K_fine, iv_fine)])
    
    # Шаг 4: вторая производная (Breeden-Litzenberger)
    dK = K_fine[1] - K_fine[0]
    d2C = np.gradient(np.gradient(C_fine, dK), dK)
    
    rnd = np.exp(r * T) * d2C
    
    # Шаг 5: нормализация (обнуляем отрицательные значения)
    rnd = np.maximum(rnd, 0)
    area = np.trapz(rnd, K_fine)
    if area > 0:
        rnd = rnd / area
    
    return K_fine, rnd


# ===================== ПРИМЕР ИСПОЛЬЗОВАНИЯ =====================
# Данные для конкретной даты: опционы на 3M RUONIA, исторические
strikes_example = np.array([0.10, 0.12, 0.14, 0.16, 0.18, 0.20])  # страйки
ivols_example   = np.array([0.22, 0.19, 0.17, 0.17, 0.18, 0.21])  # IV (доли)
F_example = 0.155  # форвардная ставка
T_example = 0.25   # 3 месяца

K_grid, rnd_density = extract_rnd_spline(
    strikes_example, ivols_example, F_example, T_example
)

plt.figure(figsize=(10, 5))
plt.plot(K_grid * 100, rnd_density, 'b-', lw=2, label='RND (сплайн + B-L)')
plt.axvline(F_example * 100, color='r', ls='--', label=f'Форвард F={F_example*100:.1f}%')
plt.xlabel('Ставка RUONIA (%)')
plt.ylabel('Плотность вероятности')
plt.title('Risk-Neutral Distribution для 3M RUONIA')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('rnd_example.png', dpi=150)
plt.show()
```

---

#### Метод 2: Смесь логнормальных распределений (Melick & Thomas, 1997)

**Идея:** RND = взвешенная смесь двух (или трёх) логнормальных распределений. Удобно для процентных ставок с несколькими режимами (например, «ставка повышается» vs «ставка снижается»).

$$q(S) = \lambda_1 \cdot LN(\mu_1, \sigma_1) + \lambda_2 \cdot LN(\mu_2, \sigma_2)$$

где $\lambda_1 + \lambda_2 = 1$, $\lambda_i > 0$.

**Параметры для оценки:** $\theta = (\mu_1, \sigma_1, \mu_2, \sigma_2, \lambda)$.

**Цена колл-опциона в смешанной модели:**

$$C^{mix}(K) = \lambda_1 C^{LN}(K; \mu_1, \sigma_1) + \lambda_2 C^{LN}(K; \mu_2, \sigma_2)$$

где $C^{LN}$ — цена Блэка–Шоулза с параметрами конкретного компонента.

**Оценка: минимизация суммы квадратов отклонений** (или максимизация правдоподобия):

$$\min_\theta \sum_i \left[C^{obs}(K_i) - C^{mix}(K_i; \theta)\right]^2$$

```python
from scipy.optimize import minimize
from scipy.stats import lognorm

def lognormal_call(F, K, T, mu, sigma2, r=0.0):
    """
    Цена колл-опциона при логнормальном распределении с параметрами mu, sigma2.
    Используем формулу Блэка: sigma_eff = sqrt(sigma2/T)
    """
    sigma_eff = np.sqrt(sigma2 / T) if T > 0 else 1e-6
    return black_call_price(F, K, T, sigma_eff, r)

def mixture_call(F, K, T, lam, mu1, sig1, mu2, sig2, r=0.0):
    """Цена колл-опциона в двухкомпонентной смеси."""
    c1 = lognormal_call(F, K, T, mu1, sig1, r)
    c2 = lognormal_call(F, K, T, mu2, sig2, r)
    return lam * c1 + (1 - lam) * c2

def fit_mixture_rnd(strikes, call_prices, F, T, r=0.0):
    """
    Подгонка двухкомпонентной смеси логнормальных распределений к ценам опционов.
    
    Returns
    -------
    params : dict с оптимальными параметрами
    rnd_func : callable, плотность RND
    """
    def objective(params):
        lam, mu1, sig1_sq, mu2, sig2_sq = params
        # ограничения: lam in (0,1), sigma > 0
        if not (0 < lam < 1 and sig1_sq > 0 and sig2_sq > 0):
            return 1e10
        errors = []
        for K, C_obs in zip(strikes, call_prices):
            C_pred = mixture_call(F, K, T, lam, mu1, sig1_sq, mu2, sig2_sq, r)
            errors.append((C_obs - C_pred)**2)
        return sum(errors)
    
    # Начальные значения
    x0 = [0.5,
          np.log(F) - 0.02, 0.03,   # компонент 1: чуть ниже форварда
          np.log(F) + 0.02, 0.05]   # компонент 2: чуть выше форварда
    
    bounds = [(0.01, 0.99), (None, None), (1e-6, None),
              (None, None), (1e-6, None)]
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 2000})
    
    lam_opt, mu1_opt, sig1_opt, mu2_opt, sig2_opt = result.x
    
    def rnd_density(x):
        """Плотность смешанного распределения."""
        eps = 1e-10
        x = np.maximum(x, eps)
        # Логнормальная плотность: ln x ~ N(mu, sigma2)
        def ln_pdf(x, mu, sigma2):
            return (1 / (x * np.sqrt(2*np.pi*sigma2))) * \
                   np.exp(-0.5 * (np.log(x) - mu)**2 / sigma2)
        return lam_opt * ln_pdf(x, mu1_opt, sig1_opt) + \
               (1 - lam_opt) * ln_pdf(x, mu2_opt, sig2_opt)
    
    params = {
        'lambda': lam_opt, 'mu1': mu1_opt, 'sigma1_sq': sig1_opt,
        'mu2': mu2_opt, 'sigma2_sq': sig2_opt,
        'success': result.success, 'loss': result.fun
    }
    
    return params, rnd_density
```

---

#### Метод 3: SABR-модель (Hagan et al., 2002)

**Самый популярный метод** для рынков процентных ставок (используется банками для опционов на LIBOR/SOFR/RUONIA).

**SDE системы:**

$$dF = \sigma F^\beta\, dW_1$$
$$d\sigma = \nu \sigma\, dW_2$$
$$\mathbb{E}[dW_1 dW_2] = \rho\, dt$$

где $F$ — форвардная ставка, $\sigma$ — стохастическая волатильность, $\beta \in [0,1]$ (обычно 0 или 0.5), $\nu$ — «волотильность волатильности», $\rho$ — корреляция.

**Приближённая формула IV (Hagan et al., 2002):**

$$\sigma_{IV}^{SABR}(K) \approx \frac{\alpha}{(FK)^{(1-\beta)/2}} \cdot \frac{z}{\chi(z)} \cdot \left[1 + \left(\frac{(1-\beta)^2}{24}\frac{\alpha^2}{(FK)^{1-\beta}} + \frac{\rho\beta\nu\alpha}{4(FK)^{(1-\beta)/2}} + \frac{2-3\rho^2}{24}\nu^2\right)T\right]$$

где:
$$z = \frac{\nu}{\alpha}(FK)^{(1-\beta)/2}\ln\frac{F}{K}, \quad \chi(z) = \ln\frac{\sqrt{1-2\rho z + z^2} + z - \rho}{1-\rho}$$

**Для ATM ($K = F$):**
$$\sigma_{ATM}^{SABR} \approx \frac{\alpha}{F^{1-\beta}}\left[1 + \left(\frac{(1-\beta)^2}{24}\frac{\alpha^2}{F^{2-2\beta}} + \frac{\rho\beta\nu\alpha}{4 F^{1-\beta}} + \frac{2-3\rho^2}{24}\nu^2\right)T\right]$$

```python
import numpy as np
from scipy.optimize import minimize

def sabr_vol(F, K, T, alpha, beta, rho, nu):
    """
    Вменённая волатильность по формуле SABR (Hagan et al., 2002).
    
    Parameters
    ----------
    F     : форвардная ставка
    K     : страйк
    T     : срок (годы)
    alpha : начальная волатильность (уровень)
    beta  : CEV параметр (0 <= beta <= 1)
    rho   : корреляция (-1 <= rho <= 1)
    nu    : вол-оф-вол (> 0)
    
    Returns
    -------
    sigma_iv : вменённая Black vol
    """
    if abs(F - K) < 1e-8:  # ATM случай
        FK_mid = F
        A = alpha / (FK_mid**(1 - beta))
        B = 1 + ((1-beta)**2 / 24 * alpha**2 / FK_mid**(2*(1-beta)) +
                  rho * beta * nu * alpha / (4 * FK_mid**(1-beta)) +
                  (2 - 3*rho**2) / 24 * nu**2) * T
        return A * B
    
    FK = (F * K)**0.5
    log_FK = np.log(F / K)
    
    # Вычисление z и chi(z)
    z = nu / alpha * FK**(1 - beta) * log_FK
    
    disc = np.sqrt(1 - 2*rho*z + z**2)
    chi = np.log((disc + z - rho) / (1 - rho))
    
    if abs(chi) < 1e-8:
        z_over_chi = 1.0
    else:
        z_over_chi = z / chi
    
    # Основной множитель
    A = alpha / (FK**(1 - beta) * (1 + (1-beta)**2/24 * log_FK**2 +
                                     (1-beta)**4/1920 * log_FK**4))
    
    # Поправочный множитель
    B = 1 + ((1-beta)**2 / 24 * alpha**2 / FK**(2*(1-beta)) +
              rho * beta * nu * alpha / (4 * FK**(1-beta)) +
              (2 - 3*rho**2) / 24 * nu**2) * T
    
    return A * z_over_chi * B


def calibrate_sabr(strikes, ivols_market, F, T, beta=0.5, verbose=False):
    """
    Калибровка SABR по рыночным котировкам IV.
    
    Returns
    -------
    params : dict {'alpha', 'beta', 'rho', 'nu'}
    """
    def objective(x):
        alpha, rho, nu = x
        if not (alpha > 0 and -1 < rho < 1 and nu > 0):
            return 1e10
        errors = []
        for K, iv_mkt in zip(strikes, ivols_market):
            try:
                iv_model = sabr_vol(F, K, T, alpha, beta, rho, nu)
                errors.append((iv_mkt - iv_model)**2)
            except Exception:
                errors.append(1e6)
        return sum(errors)
    
    # Начальные приближения: ATM IV как начальное alpha
    atm_iv = np.interp(F, strikes, ivols_market)
    x0 = [atm_iv * F**(1-beta), 0.0, 0.3]
    bounds = [(1e-5, 5.0), (-0.999, 0.999), (1e-5, 5.0)]
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 5000, 'ftol': 1e-12})
    
    alpha_opt, rho_opt, nu_opt = result.x
    
    if verbose:
        print(f"SABR калибровка: alpha={alpha_opt:.4f}, beta={beta:.2f}, "
              f"rho={rho_opt:.4f}, nu={nu_opt:.4f}")
        print(f"  Ошибка подгонки: {result.fun:.8f}, success={result.success}")
    
    # Строим RND из SABR: кубический сплайн по IV + Breeden-Litzenberger
    K_fine = np.linspace(strikes[0] * 0.5, strikes[-1] * 1.5, 300)
    iv_fine = np.array([sabr_vol(F, K, T, alpha_opt, beta, rho_opt, nu_opt)
                        for K in K_fine])
    iv_fine = np.clip(iv_fine, 1e-4, 5.0)
    
    K_rnd, rnd = extract_rnd_spline(K_fine, iv_fine, F, T)
    
    return {'alpha': alpha_opt, 'beta': beta, 'rho': rho_opt, 'nu': nu_opt,
            'success': result.success}, K_rnd, rnd
```

---

#### Метод 4: Gram-Charlier / Edgeworth Expansion

**Идея:** RND = стандартное нормальное + коррекция через моменты (скорость/эксцесс).

$$q(x) = \phi(x)\left[1 + \frac{\gamma_1}{3!}H_3(x) + \frac{\gamma_2}{4!}H_4(x) + \ldots\right]$$

где $H_n$ — полиномы Эрмита, $\gamma_1$ — асимметрия, $\gamma_2$ — эксцесс.

**Для цены колл-опциона:**

$$C^{GC}(K) = C^{BSM}(K) + e^{-rT} \sigma_F \left[\frac{\gamma_1}{3!} H_2(d_2)\phi(d_1) + \frac{\gamma_2}{4!}H_3(d_2)\phi(d_1)\right]$$

Этот метод менее точен для хвостов, но даёт **аналитическую связь** между IV улыбкой и моментами RND.

---

### 1.5 Расчёт моментов RND

После получения плотности $q(K)$ на сетке $\{K_i\}$:

```python
def compute_rnd_moments(K_grid, rnd_density):
    """
    Расчёт ключевых статистик RND.
    
    Returns
    -------
    dict: mean, std, skewness, excess_kurtosis, 
          quantiles, iqr, entropy, tail_probs
    """
    from scipy.stats import entropy as scipy_entropy
    
    # Нормализация
    dx = np.diff(K_grid)
    dx = np.append(dx, dx[-1])
    q = rnd_density * dx
    q = q / q.sum()  # нормируем в вероятностную меру
    
    # Моменты
    mean_rnd  = np.sum(K_grid * q)
    var_rnd   = np.sum((K_grid - mean_rnd)**2 * q)
    std_rnd   = np.sqrt(var_rnd)
    skew_rnd  = np.sum(((K_grid - mean_rnd) / std_rnd)**3 * q)
    kurt_rnd  = np.sum(((K_grid - mean_rnd) / std_rnd)**4 * q) - 3  # excess
    
    # Квантили (через CDF)
    cdf = np.cumsum(q)
    def get_quantile(p):
        idx = np.searchsorted(cdf, p)
        idx = min(idx, len(K_grid) - 1)
        return K_grid[idx]
    
    q05  = get_quantile(0.05)
    q25  = get_quantile(0.25)
    q50  = get_quantile(0.50)
    q75  = get_quantile(0.75)
    q95  = get_quantile(0.95)
    iqr  = q75 - q25
    fan  = q95 - q05
    
    # Энтропия Шеннона (в натуральных единицах)
    q_safe = np.maximum(q, 1e-12)
    entropy_rnd = -np.sum(q_safe * np.log(q_safe))
    
    # Вероятности хвостов (для ключевой ставки ЦБ РФ)
    # Вероятность роста ставки > +100bp от форварда
    prob_hike_100 = 1 - np.sum(q[K_grid < mean_rnd + 0.01])
    # Вероятность снижения > -100bp
    prob_cut_100  = np.sum(q[K_grid < mean_rnd - 0.01])
    
    return {
        'mean':     mean_rnd,
        'std':      std_rnd,
        'skewness': skew_rnd,
        'kurtosis': kurt_rnd,
        'q05':  q05, 'q25':  q25, 'q50': q50, 
        'q75':  q75, 'q95':  q95,
        'iqr':  iqr,
        'fan_chart': fan,
        'entropy':   entropy_rnd,
        'prob_hike_100bp': prob_hike_100,
        'prob_cut_100bp':  prob_cut_100,
    }
```

---

### 1.6 От RND к RWD: методы перехода

#### Почему RND ≠ реальному распределению?

Рыночные цены содержат **премию за риск**: инвесторы платят больше за страховку от плохих сценариев. Это приводит к тому, что $q(S)$ (RND) смещена относительно $p(S)$ (реальное/физическое распределение, RWD).

Связь:
$$q(S_T) = m(S_T) \cdot p(S_T), \quad \text{где } m(S_T) = \frac{dQ}{dP}(S_T)$$

$m(S_T)$ — **стохастический дисконтирующий фактор** (pricing kernel / SDF).

#### Метод 1: CRRA-трансформация (Liu et al., 2007)

**Ключевой результат работы.** При степенной функции полезности (CRRA) ядро ценообразования:

$$m(S_T) = \frac{1}{c} \left(\frac{S_T}{\mu}\right)^{-\gamma}$$

где $\gamma$ — коэффициент относительного неприятия риска (RRA), $\mu = E^P[S_T]$, $c$ — нормировочная константа.

Тогда RWD:

$$p(S_T) = \frac{q(S_T)}{m(S_T)} = \frac{q(S_T) \cdot \mu^\gamma \cdot c}{S_T^{-\gamma}}$$

Для **логнормальной RND** ($\ln S_T \sim N(\mu_Q, \sigma_Q^2)$ под $\mathbb{Q}$) результат аналитический:

$$\text{RWD: } \ln S_T \sim N(\mu_Q + \gamma\sigma_Q^2, \, \sigma_Q^2)$$

**Сдвиг среднего на $\gamma\sigma_Q^2$, дисперсия не меняется!**

Для **непараметрической RND** (сетка): трансформация на каждом узле.

```python
def rnd_to_rwd_crra(K_grid, rnd_density, gamma, mu_P=None):
    """
    Перевод RND → RWD через CRRA-трансформацию (Liu et al., 2007).
    
    Parameters
    ----------
    gamma : float, коэффициент относительного неприятия риска 
            (типичные оценки для ставок: 2-5)
    mu_P  : float, ожидаемое значение под реальной мерой P
            (если None, используем историческое среднее форварда)
    
    Returns
    -------
    rwd_density : np.ndarray, плотность RWD на той же сетке K_grid
    """
    if mu_P is None:
        # Используем RND-среднее как приближение
        moments = compute_rnd_moments(K_grid, rnd_density)
        mu_P = moments['mean']
    
    # Ядро ценообразования: m(K) ∝ (K/mu_P)^(-gamma)
    # Для процентных ставок: берём K напрямую
    pricing_kernel = (K_grid / mu_P) ** (-gamma)
    
    # RWD ∝ RND / m(K) = RND * K^gamma
    # (для ставок обратная зависимость: более высокие ставки → плохо для держателей облигаций)
    rwd_unnorm = rnd_density * (K_grid / mu_P) ** gamma
    
    # Нормализация
    area = np.trapz(rwd_unnorm, K_grid)
    if area > 0:
        rwd_density = rwd_unnorm / area
    else:
        rwd_density = rnd_density.copy()
    
    return rwd_density


def calibrate_gamma_mle(K_grid, rnd_density, realized_rates):
    """
    Оценка gamma методом максимального правдоподобия
    по историческим реализациям ставок.
    
    Parameters
    ----------
    realized_rates : array-like, исторические реализации ставки на горизонте T
    """
    from scipy.optimize import minimize_scalar
    from scipy.interpolate import interp1d
    
    K_arr = np.array(K_grid)
    q_arr = np.array(rnd_density)
    
    def neg_log_likelihood(gamma):
        rwd = rnd_to_rwd_crra(K_arr, q_arr, gamma)
        rwd_interp = interp1d(K_arr, rwd, bounds_error=False, fill_value=1e-10)
        ll = 0.0
        for r_obs in realized_rates:
            prob = rwd_interp(r_obs)
            ll += np.log(max(prob, 1e-10))
        return -ll
    
    result = minimize_scalar(neg_log_likelihood, bounds=(0.1, 10.0), method='bounded')
    return result.x  # оптимальная gamma


# ---- Нужно ли считать RWD для MPU? ----
"""
АРГУМЕНТЫ ЗА RWD:
- Для прогноза реализаций ставки — RWD точнее
- Для оценки "истинной" неопределённости без премии за риск
- Liu et al. (2007): RWD даёт более высокое правдоподобие

АРГУМЕНТЫ ПРОТИВ (для MPU достаточно RND):
- Risk premium сам является индикатором стресса
- MPU — это мера рыночной неопределённости, не статистической
- Neely (2005), Bauer et al. (2022): используют RND/IV напрямую
- KC PRU (Bundick et al., 2024): model-free, без RWD
- Ограниченность данных для оценки gamma
- Dahlhaus & Sekhposyan (2018): survey-based меры (≈ P-мера) дают схожие результаты

ВЫВОД: RND достаточно для MPU. RWD — ценное дополнение (+баллы).
"""
```

---

## РАЗДЕЛ 2. Построение MPU индекса: пошаговое руководство

### 2.1 Концептуальная основа

**Что измеряет MPU?** Неопределённость относительно будущей траектории ключевой ставки ЦБ РФ — то, что рынок «не знает» о будущей ДКП.

Иерархия мер от простого к сложному:

```
Уровень 1: ATM IV → прямая мера неопределённости (Neely, 2005)
Уровень 2: Std(RND) + Skew(RND) → учёт асимметрии (Dahlhaus & Sekhposyan, 2018)
Уровень 3: PCA по всем моментам + метрикам IV → агрегированный MPU
Уровень 4: + RWD, VRP, NLP индексы
```

---

### 2.2 Метрики для агрегации

#### Группа A: Моменты RND

| Признак | Обозначение | Интерпретация |
|---------|-------------|---------------|
| Станд. откл. RND | $\sigma^{RND}$ | Основная мера разброса ожиданий |
| Асимметрия RND | $\gamma_1^{RND}$ | Нап-равленность рисков (+ = риск роста ставки) |
| Эксцесс RND | $\kappa^{RND}$ | Толщина хвостов, «хвостовой риск» |
| Энтропия RND | $H^{RND}$ | Суммарная «размытость» распределения |
| Fan Chart ширина | $Q_{95} - Q_{05}$ | 90%-й CI для ставки |

#### Группа B: Метрики IV поверхности (без расчёта RND)

| Признак | Формула | Источник |
|---------|---------|----------|
| ATM IV | $\sigma_{ATM}$ | Neely (2005), Bauer (2012) |
| Risk Reversal 25Δ | $RR_{25} = \sigma_{25C} - \sigma_{25P}$ | Dahlhaus & Sekhposyan (2018) |
| Butterfly 25Δ | $BF_{25} = \frac{1}{2}(\sigma_{25C}+\sigma_{25P})-\sigma_{ATM}$ | Мера кривизны улыбки |
| Term Structure Slope | $\sigma_{ATM}(1Y) - \sigma_{ATM}(3M)$ | Bundick et al. (2024) |
| Model-free Variance | $MFV = \int_0^\infty \frac{2}{K^2}[C(K)+P(K)]dK$ | Bauer et al. (2022) — KC PRU метод |

#### Группа C: Реализованная волатильность (RV)

$$RV_t = \sqrt{\sum_{i=1}^{22} r_{t-i}^2}$$

где $r_t = \Delta \ln(\text{ключевая ставка}_t)$ или ежедневные изменения ставки RUONIA.

---

### 2.3 Метод KC PRU / Bauer et al. (2022): Model-Free Variance

**Главный метод** из литературы. Используется в KC PRU (Bundick et al., 2024) и Bauer et al. (2022).

**Формула model-free variance (по аналогии с VIX):**

$$MFV(T) = \frac{2e^{rT}}{T}\left[\int_0^F \frac{P(K)}{K^2}dK + \int_F^\infty \frac{C(K)}{K^2}dK\right]$$

```python
def model_free_variance(strikes, call_prices, put_prices, F, T, r=0.0):
    """
    Model-Free Variance по методологии VIX / Bauer et al. (2022).
    
    Аналог KC PRU для российского рынка.
    Измеряет неопределённость в квадрате (годовых базисных пунктах²).
    
    Returns
    -------
    MPU_MFV : float, sqrt(MFV) — в б.п. годовых (MPU в процентных пунктах)
    """
    strikes = np.array(strikes)
    call_prices = np.array(call_prices)
    put_prices  = np.array(put_prices)
    
    # Разделяем на OTM call (K > F) и OTM put (K < F)
    mask_call = strikes >= F
    mask_put  = strikes <= F
    
    K_call = strikes[mask_call]
    C_otm  = call_prices[mask_call]
    
    K_put  = strikes[mask_put]
    P_otm  = put_prices[mask_put]
    
    # Интегрирование по трапециевидному правилу
    def trapz_integral(K_arr, price_arr):
        if len(K_arr) < 2:
            return 0.0
        integrand = 2 * price_arr / K_arr**2
        return np.trapz(integrand, K_arr)
    
    integral_calls = trapz_integral(K_call, C_otm)
    integral_puts  = trapz_integral(K_put, P_otm)
    
    mfv = np.exp(r * T) / T * (integral_calls + integral_puts)
    mfv_annualized = np.sqrt(max(mfv, 0))  # в п.п. годовых (как VIX)
    
    return mfv_annualized
```

---

### 2.4 Построение MPU через PCA

**Метод PCA-агрегации** — рекомендуемый для академической работы (максимальная информативность, обоснованность весов).

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def build_mpu_pca(df_features, n_components=1, verbose=True):
    """
    Построение MPU индекса через PCA по нескольким признакам.
    
    Parameters
    ----------
    df_features : pd.DataFrame, колонки = признаки, строки = даты
    
    Returns
    -------
    mpu_series : pd.Series, нормализованный MPU индекс
    loadings   : pd.Series, нагрузки PC1 на признаки
    explained  : float, доля объяснённой дисперсии PC1
    """
    # Шаг 1: убираем пропуски
    df_clean = df_features.dropna()
    
    # Шаг 2: нормализация (z-score)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    # Шаг 3: PCA
    pca = PCA(n_components=min(n_components + 2, df_clean.shape[1]))
    pca.fit(X_scaled)
    
    # Шаг 4: PC1 — наш MPU
    pc1_scores = pca.transform(X_scaled)[:, 0]
    
    # Шаг 5: нормализация MPU в [0, 100] для интерпретируемости
    mpu_raw = pd.Series(pc1_scores, index=df_clean.index, name='MPU_PCA')
    
    # Знак: убеждаемся, что рост MPU = рост неопределённости
    # (высокие нагрузки должны быть у sigma_RND, ATM_IV)
    loadings = pd.Series(pca.components_[0], index=df_clean.columns,
                         name='PC1_loadings')
    
    # Если основные меры волатильности имеют отрицательные нагрузки — инвертируем
    key_features = [c for c in ['std_rnd', 'atm_iv', 'fan_chart', 'mfv'] 
                    if c in loadings.index]
    if key_features and loadings[key_features].mean() < 0:
        mpu_raw = -mpu_raw
        loadings = -loadings
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"MPU PCA: объяснённая дисперсия PC1 = {pca.explained_variance_ratio_[0]:.1%}")
        print(f"\nНагрузки PC1 (отсортированы):")
        print(loadings.sort_values(ascending=False).to_string())
        print(f"{'='*50}\n")
    
    return mpu_raw, loadings, pca.explained_variance_ratio_[0]


# ============================================================
# ПОЛНЫЙ PIPELINE: от IV поверхности до MPU индекса
# ============================================================

def full_mpu_pipeline(iv_data_df, rates_history):
    """
    Полный pipeline построения MPU.
    
    Parameters
    ----------
    iv_data_df : pd.DataFrame
        Колонки: date, tenor, strike, iv
        (IV поверхность для каждой исторической даты)
    rates_history : pd.Series
        Исторические ежедневные значения ключевой ставки ЦБ РФ
    
    Returns
    -------
    mpu_df : pd.DataFrame с различными MPU спецификациями
    """
    results = []
    
    for date, group in iv_data_df.groupby('date'):
        row = {'date': date}
        
        # Обрабатываем 3M тенор (можно расширить на другие)
        df_3m = group[group['tenor'] == '3M'].sort_values('strike')
        
        if len(df_3m) < 3:
            continue
        
        strikes = df_3m['strike'].values
        ivols   = df_3m['iv'].values
        F       = df_3m['forward'].values[0] if 'forward' in df_3m else np.median(strikes)
        T       = 0.25  # 3M
        
        # ---- RND ----
        K_grid, rnd = extract_rnd_spline(strikes, ivols, F, T)
        moments = compute_rnd_moments(K_grid, rnd)
        
        row['std_rnd']      = moments['std']
        row['skew_rnd']     = moments['skewness']
        row['kurt_rnd']     = moments['kurtosis']
        row['entropy_rnd']  = moments['entropy']
        row['iqr_rnd']      = moments['iqr']
        row['fan_chart']    = moments['fan_chart']
        row['prob_hike_1pct'] = moments['prob_hike_100bp']
        row['prob_cut_1pct']  = moments['prob_cut_100bp']
        
        # ---- IV метрики (прямые, без RND) ----
        # ATM IV: ближайший к форварду
        atm_idx = np.argmin(np.abs(strikes - F))
        row['atm_iv'] = ivols[atm_idx]
        
        # Risk Reversal 25Δ (грубое приближение через первый/последний страйк)
        if len(strikes) >= 5:
            row['rr25'] = ivols[-2] - ivols[1]    # OTM call IV - OTM put IV
            row['bf25'] = 0.5*(ivols[-2] + ivols[1]) - ivols[atm_idx]
        
        # ---- Реализованная волатильность ----
        past_rates = rates_history[:date]
        if len(past_rates) >= 22:
            daily_changes = past_rates.pct_change().dropna()
            rv_22d = daily_changes.tail(22).std() * np.sqrt(252)
            row['rv_22d'] = rv_22d
        
        results.append(row)
    
    df = pd.DataFrame(results).set_index('date')
    
    # ---- MPU спецификации ----
    # 1. MPU_PCA: главная компонента
    feature_cols = ['std_rnd', 'atm_iv', 'entropy_rnd', 'fan_chart', 
                    'iqr_rnd', 'kurt_rnd', 'bf25']
    feat_df = df[feature_cols].dropna()
    df['MPU_PCA'], loadings, expl = build_mpu_pca(feat_df)
    
    # 2. MPU_ATM: прямой прокси (Neely 2005, Bauer et al. 2022)
    df['MPU_ATM'] = (df['atm_iv'] - df['atm_iv'].mean()) / df['atm_iv'].std()
    
    # 3. MPU_IQR: межквартильный размах
    df['MPU_IQR'] = (df['iqr_rnd'] - df['iqr_rnd'].mean()) / df['iqr_rnd'].std()
    
    # 4. MPU_Entropy: информационная энтропия
    df['MPU_Entropy'] = (df['entropy_rnd'] - df['entropy_rnd'].mean()) / \
                         df['entropy_rnd'].std()
    
    # 5. MPU_RV: реализованная волатильность (baseline)
    df['MPU_RV'] = (df['rv_22d'] - df['rv_22d'].mean()) / df['rv_22d'].std()
    
    return df, loadings
```

---

### 2.5 Визуализация MPU

```python
def plot_mpu_comparison(mpu_df, events_dict=None):
    """
    График сравнения различных MPU спецификаций.
    
    Parameters
    ----------
    events_dict : dict, например {'2022-02-24': 'СВО', '2020-03-20': 'COVID'}
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    mpu_cols = ['MPU_PCA', 'MPU_ATM', 'MPU_IQR', 'MPU_Entropy', 'MPU_RV']
    colors   = ['navy', 'crimson', 'green', 'purple', 'orange']
    
    # Панель 1: все MPU индексы
    ax1 = axes[0]
    for col, color in zip(mpu_cols, colors):
        if col in mpu_df.columns:
            ax1.plot(mpu_df.index, mpu_df[col], label=col, color=color, 
                     alpha=0.8, lw=1.5)
    ax1.set_ylabel('Нормализованный MPU (z-score)')
    ax1.set_title('Сравнение спецификаций MPU')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.axhline(0, color='black', lw=0.5, ls='--')
    ax1.grid(alpha=0.3)
    
    # Панель 2: декомпозиция RND (std + skew + kurtosis)
    ax2 = axes[1]
    if 'std_rnd' in mpu_df.columns:
        ax2.fill_between(mpu_df.index, 
                         mpu_df['std_rnd'] * 100,  # в б.п.
                         alpha=0.4, color='blue', label='Std(RND), б.п.')
        ax2.plot(mpu_df.index, mpu_df['std_rnd'] * 100, 'b-', lw=1.5)
    ax2.set_ylabel('Std RND (б.п. годовых)')
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3)
    
    if 'skew_rnd' in mpu_df.columns:
        ax2b = ax2.twinx()
        ax2b.plot(mpu_df.index, mpu_df['skew_rnd'], 'r--', lw=1.2, 
                  label='Skew(RND)', alpha=0.7)
        ax2b.set_ylabel('Асимметрия RND', color='red')
        ax2b.axhline(0, color='red', lw=0.5, ls=':')
        ax2b.legend(loc='upper right')
    
    # Панель 3: Fan Chart (Q5-Q95 полоса)
    ax3 = axes[2]
    if all(c in mpu_df.columns for c in ['q05', 'q50', 'q95']):
        ax3.fill_between(mpu_df.index, 
                         mpu_df['q05'] * 100, mpu_df['q95'] * 100,
                         alpha=0.3, color='steelblue', label='90% доверительный интервал')
        ax3.plot(mpu_df.index, mpu_df['q50'] * 100, 'b-', lw=2, label='Медиана RND')
    ax3.set_ylabel('Ставка (%)')
    ax3.set_title('Fan Chart: прогноз RND')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Ключевые события
    if events_dict:
        for date_str, label in events_dict.items():
            try:
                date = pd.Timestamp(date_str)
                for ax in axes:
                    ax.axvline(date, color='gray', lw=1.5, ls=':', alpha=0.8)
                axes[0].text(date, axes[0].get_ylim()[1] * 0.9, label,
                             rotation=90, fontsize=7, va='top', ha='right')
            except Exception:
                pass
    
    plt.tight_layout()
    plt.savefig('mpu_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# Ключевые события для российского контекста
RU_EVENTS = {
    '2014-12-16': 'Ставка 17%',
    '2018-04-06': 'Санкции',
    '2020-03-20': 'COVID',
    '2022-02-28': 'Ставка 20%',
    '2022-09-21': 'Мобилизация',
    '2023-07-21': 'Цикл повышения 2023',
    '2024-12-20': 'Пауза ЦБ (21%)',
}
```

---

## РАЗДЕЛ 3. Эконометрика: тесты и макромодели

### 3.1 Предварительные тесты

```python
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests

def stationarity_tests(series, name='Series', max_lags=12):
    """Тесты на стационарность: ADF + KPSS."""
    print(f"\n{'='*50}")
    print(f"Тесты стационарности: {name}")
    print('='*50)
    
    # ADF: H0 = единичный корень (нестационарный)
    adf_result = adfuller(series.dropna(), maxlag=max_lags, autolag='AIC')
    print(f"ADF:  t-stat={adf_result[0]:.4f}, p-value={adf_result[1]:.4f}")
    print(f"  Крит. значения: {adf_result[4]}")
    
    # KPSS: H0 = стационарный
    kpss_result = kpss(series.dropna(), regression='c', nlags=max_lags)
    print(f"KPSS: t-stat={kpss_result[0]:.4f}, p-value={kpss_result[1]:.4f}")
    
    is_stationary = (adf_result[1] < 0.05) and (kpss_result[1] > 0.05)
    print(f"Вывод: {'СТАЦИОНАРЕН' if is_stationary else 'НЕСТАЦИОНАРЕН'}")
    return is_stationary
```

---

### 3.2 Тест причинности Грейнджера

**Гипотеза:** $MPU_t$ содержит опережающую информацию о $RV_{t+h}$.

```python
def granger_causality_tests(mpu_series, target_series, 
                              max_lag=12, target_name='RV'):
    """
    Тест Грейнджера: предсказывает ли MPU будущее значение target?
    
    H0: MPU НЕ является причиной Грейнджера для target
    """
    df = pd.DataFrame({
        'target': target_series,
        'mpu':    mpu_series
    }).dropna()
    
    print(f"\nТест Грейнджера: MPU → {target_name}")
    print("-" * 40)
    
    results = grangercausalitytests(df[['target', 'mpu']], maxlag=max_lag,
                                     verbose=False)
    
    # Выводим p-values для каждого лага
    print(f"{'Лаг':>5} | {'F-stat':>10} | {'p-value':>10} | {'Значимо':>10}")
    print("-" * 40)
    for lag in range(1, max_lag + 1):
        f_stat = results[lag][0]['ssr_ftest'][0]
        p_val  = results[lag][0]['ssr_ftest'][1]
        sig    = '***' if p_val < 0.01 else ('**' if p_val < 0.05 else 
                 ('*' if p_val < 0.1 else ''))
        print(f"{lag:>5} | {f_stat:>10.4f} | {p_val:>10.4f} | {sig:>10}")
    
    return results
```

---

### 3.3 Predictive Regression (прогностическая регрессия)

**Спецификация (Chang & Feunou, 2013; Bauer et al., 2022):**

$$RV_{t+h} = \alpha_h + \beta_h^{MPU} \cdot MPU_t + \beta_h^{RV} \cdot RV_t + \varepsilon_{t+h}$$

где $h \in \{1, 3, 6, 12\}$ месяцев.

```python
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac

def predictive_regression(mpu_series, rv_series, 
                           horizons=[1, 3, 6, 12],
                           controls=None):
    """
    Прогностическая регрессия MPU → RV на различных горизонтах.
    Использует HAC стандартные ошибки (Newey-West) для серийной корреляции.
    """
    results_table = []
    
    for h in horizons:
        # Создаём shifted переменную (future RV)
        df = pd.DataFrame({
            'rv_future': rv_series.shift(-h),
            'mpu':       mpu_series,
            'rv_lag':    rv_series,
        })
        
        if controls is not None:
            for name, series in controls.items():
                df[name] = series
        
        df = df.dropna()
        
        # Регрессоры
        X_cols = ['mpu', 'rv_lag'] + (list(controls.keys()) if controls else [])
        X = sm.add_constant(df[X_cols])
        y = df['rv_future']
        
        model = OLS(y, X).fit()
        
        # HAC стандартные ошибки (Newey-West, lags = 1.5*horizon)
        nw_lags = int(1.5 * h)
        hac_cov = cov_hac(model, nlags=nw_lags)
        model_hac = model.get_robustcov_results(cov_type='HAC', maxlags=nw_lags)
        
        results_table.append({
            'Горизонт (мес)': h,
            'β_MPU':          model_hac.params.get('mpu', np.nan),
            't-stat MPU':     model_hac.tvalues.get('mpu', np.nan),
            'p-value MPU':    model_hac.pvalues.get('mpu', np.nan),
            'R² (полная)':    model.rsquared,
            'R² (adj)':       model.rsquared_adj,
            'N':              len(y),
        })
    
    results_df = pd.DataFrame(results_table)
    print("\nPredictive Regression: MPU → RV(t+h)")
    print(results_df.to_string(index=False))
    return results_df


# HAR-RV модель (Corsi, 2009) с MPU
def har_rv_mpu(rv_series, mpu_series, h=1):
    """
    HAR-RV + MPU:
    RV(t+h) = c + β_D*RV(t) + β_W*RV^W(t) + β_M*RV^M(t) + γ*MPU(t) + ε
    """
    rv_daily   = rv_series
    rv_weekly  = rv_series.rolling(5).mean()
    rv_monthly = rv_series.rolling(22).mean()
    
    df = pd.DataFrame({
        'rv_h':        rv_series.shift(-h),
        'rv_d':        rv_daily,
        'rv_w':        rv_weekly,
        'rv_m':        rv_monthly,
        'mpu':         mpu_series,
    }).dropna()
    
    X = sm.add_constant(df[['rv_d', 'rv_w', 'rv_m', 'mpu']])
    model = OLS(df['rv_h'], X).fit(cov_type='HAC', cov_kwds={'maxlags': h + 1})
    
    print(f"\nHAR-RV-MPU (горизонт = {h} дн.):")
    print(model.summary().tables[1])
    return model
```

---

### 3.4 SVAR модель с MPU

**Порядок переменных (рекурсивная идентификация Холецкого):**

$$\mathbf{Y}_t = [MPU_t, \; r_t^{key}, \; \pi_t, \; y_t^{gap}, \; REER_t]^\prime$$

**Идентификационные ограничения:** шок MPU предположительно **predetermined** относительно ДКП переменных (рынки реагируют быстрее, чем ЦБ меняет ставку → MPU первым в рекурсии).

```python
from statsmodels.tsa.api import VAR

def run_svar_mpu(df_macro, mpu_col='MPU_PCA', maxlags=4, horizon=24):
    """
    SVAR с MPU и макропеременными.
    
    Parameters
    ----------
    df_macro : pd.DataFrame
        Колонки: MPU, ключевая ставка, инфляция (ИПЦ), разрыв выпуска, REER
    
    Returns
    -------
    irf : IRF объект statsmodels
    hd  : pd.DataFrame, историческая декомпозиция
    """
    # Убираем пропуски
    df = df_macro.dropna()
    
    # VAR модель
    model = VAR(df)
    
    # Выбор лагов по AIC
    lag_order = model.select_order(maxlags=maxlags)
    print(f"\nВыбор лагов:")
    print(lag_order.summary())
    optimal_lag = lag_order.aic
    
    # Оценка VAR
    var_result = model.fit(maxlags=optimal_lag, ic='aic', verbose=True)
    print(var_result.summary())
    
    # Тест на автокорреляцию остатков
    print("\nТест Portmanteau:")
    print(var_result.test_whiteness(nlags=optimal_lag + 4).summary())
    
    # Импульсные отклики (чол. идентификация)
    irf = var_result.irf(periods=horizon)
    
    # Визуализация IRF
    n_vars = df.shape[1]
    mpu_idx = list(df.columns).index(mpu_col)
    
    fig, axes = plt.subplots(1, n_vars, figsize=(16, 4))
    colors_irf = ['navy', 'crimson', 'green', 'purple', 'orange']
    
    for j, (var_name, color) in enumerate(zip(df.columns, colors_irf)):
        ax = axes[j]
        irf_values  = irf.irfs[:, j, mpu_idx]     # j-я переменная на шок MPU
        irf_lower   = irf.cum_effects_stderr       # приближение ДИ
        
        # Получение bootstrap CI
        try:
            irf_boot = var_result.irf(periods=horizon)
            irf_ci = irf_boot.cum_errband_mc(orth=False, repl=200, 
                                               signif=0.16, seed=42)
            lower = irf_ci[0][:, j, mpu_idx]
            upper = irf_ci[1][:, j, mpu_idx]
            ax.fill_between(range(horizon + 1), lower, upper, alpha=0.2, color=color)
        except Exception:
            pass
        
        ax.plot(irf_values, color=color, lw=2)
        ax.axhline(0, color='black', lw=0.8, ls='--')
        ax.set_title(f'→ {var_name}', fontsize=10)
        ax.set_xlabel('Месяцы')
        ax.grid(alpha=0.3)
    
    axes[0].set_ylabel('Откл. от базового сценария')
    fig.suptitle('IRF: шок MPU (1 ст. откл.) → макропеременные', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('svar_irf_mpu.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return var_result, irf


# Local Projections (Jordà, 2005) — альтернатива VAR
def local_projections(y, mpu, controls=None, horizons=range(0, 25)):
    """
    Локальные проекции Жорда для оценки IRF без ограничений VAR.
    
    Модель: y_{t+h} = α_h + β_h * MPU_t + γ_h * X_t + ε_{t+h}
    """
    beta_h = []
    ci_lower = []
    ci_upper = []
    
    for h in horizons:
        df = pd.DataFrame({'y_h': y.shift(-h), 'mpu': mpu})
        
        if controls is not None:
            for name, s in controls.items():
                df[name] = s
        
        df = df.dropna()
        X_cols = ['mpu'] + (list(controls.keys()) if controls else [])
        X = sm.add_constant(df[X_cols])
        
        # HAC стандартные ошибки
        model = OLS(df['y_h'], X).fit(
            cov_type='HAC', cov_kwds={'maxlags': h + 1}
        )
        
        beta_h.append(model.params['mpu'])
        ci_lower.append(model.conf_int(alpha=0.16).loc['mpu', 0])  # 68% CI
        ci_upper.append(model.conf_int(alpha=0.16).loc['mpu', 1])
    
    # Построение графика
    plt.figure(figsize=(10, 5))
    plt.fill_between(list(horizons), ci_lower, ci_upper, 
                     alpha=0.3, color='navy', label='68% ДИ')
    plt.plot(list(horizons), beta_h, 'navy', lw=2, marker='o', ms=4,
             label='LP: β_h (MPU → y)')
    plt.axhline(0, color='black', lw=0.8, ls='--')
    plt.xlabel('Горизонт (месяцы)')
    plt.ylabel('Коэффициент при MPU')
    plt.title('Local Projections: отклик переменной на шок MPU')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('lp_irf.png', dpi=150)
    plt.show()
    
    return pd.DataFrame({'horizon': horizons, 'beta': beta_h,
                          'ci_lower': ci_lower, 'ci_upper': ci_upper})
```

---

### 3.5 Историческая декомпозиция

**Цель:** показать, какой вклад шока MPU вносил в исторические отклонения переменных от базовой траектории.

```python
def plot_historical_decomposition(var_result, var_name, mpu_name, 
                                   events_dict=None):
    """
    Историческая декомпозиция для выбранной переменной.
    Показывает вклад шока MPU в динамику переменной.
    """
    # Историческая декомпозиция через statsmodels
    hd = var_result.fevd(periods=20)  # Forecast Error Variance Decomposition
    
    # Для полной исторической декомпозиции:
    # нужно использовать structural_irf
    # (упрощение: смотрим FEVD как proxy для важности шоков)
    
    print(f"\nFEVD для {var_name}:")
    fevd_df = pd.DataFrame(
        hd.decomp[list(var_result.names).index(var_name)],
        columns=var_result.names
    )
    print(fevd_df.tail(10).to_string())
    
    # Доля MPU шока в дисперсии переменной на горизонте 12 мес
    mpu_share = fevd_df[mpu_name].iloc[12]
    print(f"\nДоля MPU-шока в дисперсии {var_name} (12 мес): {mpu_share:.1%}")
    
    return fevd_df
```

---

## РАЗДЕЛ 4. Мастер-чеклист по всем 4 подзаданиям

### ✅ Подзадание 1: Анализ RND и построение MPU [50 + до 80 баллов]

**Минимальный набор (50 баллов):**
- [ ] Сформулирована теоретическая основа (Breeden-Litzenberger)
- [ ] Выбран и обоснован метод извлечения RND (SABR или кубический сплайн)
- [ ] RND восстановлено для каждой исторической даты
- [ ] Рассчитаны базовые моменты: $\sigma, \gamma_1, \kappa$, квантили
- [ ] Описательная статистика по всем датам (mean, std, min/max)
- [ ] Построен хотя бы один MPU индекс
- [ ] Визуализация: Fan Chart, улыбка волатильности

**Дополнительные баллы:**
- [ ] Несколько методов извлечения RND (сравнение)
- [ ] MPU через PCA (обоснование весов через нагрузки)
- [ ] Энтропийный MPU
- [ ] Model-free variance (аналог KC PRU/Bauer et al.)
- [ ] RWD через CRRA (Liu et al., 2007) + сравнение с RND
- [ ] Обоснование: нужно ли вообще RWD для MPU?
- [ ] Метрики IV поверхности без RND (ATM IV, RR25, BF25, term structure)
- [ ] NLP-индекс (пресс-релизы ЦБ РФ)
- [ ] Variance Risk Premium (VRP = IV² − RV²)

---

### ✅ Подзадание 2: Прогнозная сила [25 + до 35 баллов]

**Минимальный набор (25 баллов):**
- [ ] Рассчитана RV ключевой ставки (rolling std или простая формула)
- [ ] Тест Грейнджера MPU → RV (p-values по лагам)
- [ ] Predictive regression на h = 1, 3, 6, 12 мес
- [ ] Диаграмма scatter: MPU_t vs RV_{t+h}

**Дополнительные баллы:**
- [ ] HAR-RV-MPU модель (Corsi, 2009 + MPU)
- [ ] Прогностическая регрессия: MPU → ИПЦ, REER, ВВП разрыв
- [ ] Out-of-sample тест: R²_OS, Diebold-Mariano тест
- [ ] Сравнение MPU_PCA vs MPU_ATM vs MPU_RV в прогнозе
- [ ] Условная прогнозная сила: отдельно в кризисы vs норм. периоды
- [ ] Prequential/sequential forecasting evaluation

---

### ✅ Подзадание 3: MPU в макро-финансовой модели [25 + до 35 баллов]

**Минимальный набор (25 баллов):**
- [ ] VAR модель: [MPU, ключевая ставка, ИПЦ, разрыв выпуска, REER]
- [ ] Выбор лагов (AIC/BIC), тест на автокорреляцию
- [ ] IRF: шок MPU → ключевая ставка, ИПЦ, выпуск
- [ ] FEVD: доля MPU-шоков в дисперсии переменных

**Дополнительные баллы:**
- [ ] Local Projections (Jordà, 2005) вместо/в дополнение к VAR
- [ ] Знаковые ограничения для идентификации (sign restrictions)
- [ ] Историческая декомпозиция с маркировкой событий (2014, 2020, 2022)
- [ ] SVAR vs LP: сравнение IRF
- [ ] Нелинейный VAR (threshold VAR): разный отклик при высоком/низком MPU
- [ ] Сравнение VAR с MPU_PCA vs MPU_ATM

---

### ✅ Подзадание 4: Рекомендации по ДКП [оценивается бизнес-жюри]

**Структура раздела:**

1. **Текущий уровень MPU** (апрель 2026)
   - Где находится MPU относительно исторических квантилей?
   - Сравнение с аналогичными периодами (2015, 2022, 2023)

2. **Интерпретация компонент**
   - Что именно растёт: $\sigma^{RND}$ (разброс) или скос (асимметрия)?
   - Положительный скос = рынок ждёт повышения ставки
   - Широкий fan chart = высокая неопределённость

3. **Связь с решениями ЦБ РФ**
   - MPU обычно падает **после** заседания ЦБ (разрешение неопределённости)
   - Аналог «FOMC uncertainty cycle» из Bauer et al. (2022) для ЦБ РФ

4. **Конкретные рекомендации**
   - При высоком MPU + положительный скос: рынок ждёт удержания или повышения → ЦБ может снизить неопределённость чётким форвард-гайдансом
   - При снижающемся MPU: сигнал нормализации ожиданий, пространство для снижения ставки

5. **Ограничения модели**
   - Ликвидность российского рынка деривативов
   - Эффект санкций на ценообразование опционов
   - Risk premium vs pure uncertainty

---

## РАЗДЕЛ 5. Ключевые источники: разбор методологий

### 5.1 Bauer, Lakdawala & Mueller (2022) — Центральный источник

**Вклад:** новый model-free measure MPU на основе опционов на Eurodollar/SOFR.

**Метод (точно по тексту):**
- Инструменты: опционы на фьючерсы Eurodollar (до 2022) → SOFR (с 2022)
- Мера: **model-free conditional variance** (квадратный корень из MFV)
- Формула: аналог VIX, но для процентных ставок
- Ежедневная частота, горизонты 6, 12, 18, 24 месяца
- Путём линейной интерполяции условных дисперсий

**Ключевые результаты:**
- «FOMC uncertainty cycle»: неопределённость падает на дату заседания, потом восстанавливается
- Изменения MPU вокруг FOMC важны сами по себе (forward guidance)
- Ответ финансовых рынков на policy surprise зависит от уровня MPU

**Применение к РФ:** 
- Базовый актив: фьючерсы на RUONIA/SOFR-аналог (RUSFAR)
- Торгуются на Московской бирже секции FORTS
- Формула MFV аналогичная

---

### 5.2 Bundick, Smith & Van der Meer (2024) — KC PRU

**Вклад:** официальный индикатор ФРБ Канзас-Сити, публично доступен.

**Метод:**
- **Методология VIX** применена к опционам на процентные фьючерсы
- Одногодовой горизонт (аналог «1Y ahead uncertainty»)
- Ежедневная частота с 1989 г.
- Единицы: аннуализированные процентные пункты (как «VIX ставок»)
- KC PRU = $\sqrt{MFV(T=1Y)} \times 100$ (в б.п.)

**Преимущества:**
- Нет модельных предположений (model-free)
- Длинная история
- Легко сравнивать с реализованной волатильностью

**Связанный показатель KC PRS** (Policy Rate Skew): мера асимметрии рисков (Bundick, Doh & Smith, 2024).

---

### 5.3 Neely (2005) — Классическая работа ФРБ Сент-Луис

**Вклад:** первая систематическая работа по использованию IV для измерения неопределённости процентных ставок.

**Метод:**
- Instruments: опционы на 3-месячные Eurodollar фьючерсы
- MPU = ATM implied volatility (простейшая мера!)
- Период: 1985-2001
- Тест: IV предсказывает будущую RV (Mincer-Zarnowitz test)

**Ключевой вывод:** IV eurodollar ставок снижалась ~20 лет вместе с инфляцией и ставками (Great Moderation). Изменения IV совпадают с ключевыми новостями.

**Применение к РФ:** ATM IV опционов на RUONIA как простейший MPU (хорошая baseline мера).

---

### 5.4 Dahlhaus & Sekhposyan (2018) — «A Tale of Two Tails»

**Вклад:** Разделение MPU на «downside» (downside uncertainty: ставка выше ожиданий) и «upside» (ставка ниже ожиданий) составляющие.

**Метод:**
- **Основная мера:** предсказуемость ставки через Blue Chip Financial Forecasts
- MPU = forecast dispersion / forecast errors
- Два «хвоста» неопределённости: ошибки вверх vs вниз
- Цикличность: tightening cycles → больше downside uncertainty

**Ключевой вывод:** 
- Периоды ужесточения ДКП → downside uncertainty выше
- Симметрия нарушается: downside uncertainty снизилась (Fed более предсказуем), upside — нет
- Нельзя ограничиваться одной мерой (std); нужна асимметрия

**Применение к РФ:** 
- Risk Reversal как мера асимметрии хвостов
- Скос RND = $\gamma_1^{RND}$: рост при ожиданиях повышения ставки
- Вероятности из tail probability (MPU_downside = prob_hike, MPU_upside = prob_cut)

---

### 5.5 Chang & Feunou (2013) — Банк Канады

**Вклад:** Сравнение IV-based и RV-based мер MPU для Канады. Показали, что обе меры работают, но IV-based более опережающая.

**Ключевые находки:**
- IV содержит risk premium компонент → MPU ≠ чистая «неопределённость»
- Variance Risk Premium (VRP) = IV² − RV² = плата за страховку
- При высоком VRP: рынок платит за хеджирование → неопределённость высока

---

### 5.6 Husted, Rogers & Sun (2020) — Текстовый MPU

**Вклад:** Построение MPU на основе новостных текстов (NLP-подход), аналог Baker-Bloom-Davis (2016).

**Метод:**
- Частота упоминания слов «monetary policy», «uncertainty», «Federal Reserve» в газетах
- Нормализация на общий поток новостей
- Ежемесячная частота с 1985

**Ограничения:** только для США. Для РФ нужна адаптация (Росстат/РБК/Коммерсантъ).

**NLP-индекс для ЦБ РФ:**

```python
import re
from collections import Counter

def build_cbr_nlp_mpu(texts_dict, uncertainty_keywords=None,
                       monetary_keywords=None):
    """
    Построение NLP-MPU по текстам пресс-релизов ЦБ РФ.
    
    Parameters
    ----------
    texts_dict : dict {date: text_string}
    
    Returns
    -------
    nlp_mpu : pd.Series, нормализованный NLP MPU
    """
    if uncertainty_keywords is None:
        uncertainty_keywords = [
            'неопределённость', 'неопределенность', 'риски', 'волатильность',
            'нестабильность', 'непредсказуемость', 'неизвестность',
            'возможно', 'при необходимости', 'баланс рисков',
            'проинфляционные риски', 'дезинфляционные риски',
            'точность прогнозов', 'прогнозная неопределённость'
        ]
    
    if monetary_keywords is None:
        monetary_keywords = [
            'ключевая ставка', 'денежно-кредитная политика',
            'банк России', 'совет директоров', 'инфляция',
            'монетарная политика'
        ]
    
    results = {}
    for date, text in texts_dict.items():
        text_lower = text.lower()
        
        # Подсчёт вхождений
        n_uncertainty = sum(text_lower.count(kw) for kw in uncertainty_keywords)
        n_monetary    = sum(text_lower.count(kw) for kw in monetary_keywords)
        n_total_words = len(text_lower.split())
        
        # Нормализованный индекс
        if n_total_words > 0:
            nlp_score = (n_uncertainty / n_total_words) * 1000  # per 1000 слов
        else:
            nlp_score = 0.0
        
        results[date] = {
            'nlp_raw':        nlp_score,
            'n_uncertainty':  n_uncertainty,
            'n_monetary':     n_monetary,
        }
    
    df = pd.DataFrame(results).T
    df.index = pd.to_datetime(df.index)
    
    # Нормализация
    nlp_mpu = (df['nlp_raw'] - df['nlp_raw'].mean()) / df['nlp_raw'].std()
    nlp_mpu.name = 'MPU_NLP'
    
    return nlp_mpu
```

---

### 5.7 Liu et al. (2007) — RND → RWD

**Вклад:** Два параметрических метода перехода от RND к RWD с аналитическими формулами.

**Метод 1: CRRA-трансформация** (уже реализован выше):
- Ядро ценообразования: $m(S) = c \cdot (S/\mu)^{-\gamma}$
- Оценка $\gamma$ по MLE из исторических реализаций
- Результат для FTSE-100: $\gamma \approx 3-6$ (типичный диапазон)

**Метод 2: Статистическая калибровка** (calibration transformation):
- Подбираем монотонную функцию $g(\cdot)$ так, что $S^P = g(S^Q)$
- $g$ оценивается по maximum likelihood
- Более гибкий, но менее интерпретируемый

**Ключевой вывод:** для FTSE-100 RWD обеспечивает более высокое правдоподобие реализаций, чем простая историческая плотность.

---

### 5.8 Cooper (1999) — Тестирование методов извлечения RND

**Вклад:** BIS Workshop Materials — практический обзор методов RND с тестами качества.

**Методы тестирования RND:**
1. **In-sample fit**: RMSE между рыночными ценами и ценами из RND
2. **Density forecasting** (Berkowitz, 2001): если RND правильна, квантили реализаций ~ Uniform[0,1]
3. **Mincer-Zarnowitz test**: $\sigma^{realized} = a + b \cdot \sigma^{RND} + \varepsilon$; $H_0: a=0, b=1$

```python
from scipy.stats import kstest, uniform

def test_rnd_calibration(rnd_at_t, realized_at_t_plus_h):
    """
    Тест Беркёвица на правильность плотностных прогнозов.
    
    Если RND откалиброваны верно, pit = CDF_RND(realized) ~ Uniform[0,1].
    
    Parameters
    ----------
    rnd_at_t : list of (K_grid, rnd_density) за разные даты t
    realized_at_t_plus_h : list of float, реализации ставки за период t+h
    
    Returns
    -------
    ks_stat, ks_pval : KS тест на равномерность
    """
    pit_values = []
    for (K_grid, rnd), r_obs in zip(rnd_at_t, realized_at_t_plus_h):
        # CDF в точке реализации
        dx = np.diff(K_grid)
        dx = np.append(dx, dx[-1])
        q = rnd * dx
        q = q / q.sum()
        cdf = np.cumsum(q)
        
        # PIT = CDF(r_obs)
        idx = np.searchsorted(K_grid, r_obs)
        pit = cdf[min(idx, len(cdf)-1)]
        pit_values.append(pit)
    
    # Kolmogorov-Smirnov тест на Uniform[0,1]
    ks_stat, ks_pval = kstest(pit_values, 'uniform')
    
    print(f"\nТест калиброванности RND (Berkowitz):")
    print(f"KS статистика: {ks_stat:.4f}")
    print(f"p-value: {ks_pval:.4f}")
    print(f"Вывод: {'RND откалиброваны (не отвергаем H0)' if ks_pval > 0.05 else 'RND плохо откалиброваны'}")
    
    # Гистограмма PIT
    plt.figure(figsize=(8, 4))
    plt.hist(pit_values, bins=20, density=True, alpha=0.7, 
             color='steelblue', label='PIT значения')
    plt.axhline(1.0, color='red', ls='--', label='Равномерное')
    plt.xlabel('PIT (Probability Integral Transform)')
    plt.ylabel('Плотность')
    plt.title(f'Тест калиброванности RND\n(KS p-value = {ks_pval:.3f})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('rnd_calibration_test.png', dpi=150)
    
    return ks_stat, ks_pval, pit_values
```

---

### 5.9 Итоговая таблица: сравнение методов MPU

| Метод | Источник | Данные | Частота | Опережающий? | Интерпрет. | Сложность |
|-------|----------|--------|---------|-------------|------------|-----------|
| ATM IV | Neely (2005) | Опционы | Ежедневно | ✓✓ | Высокая | Низкая |
| Model-Free Var (KC PRU) | Bundick et al. (2024) | Опционы | Ежедневно | ✓✓ | Высокая | Средняя |
| Std(RND) | Bauer et al. (2022) | Опционы | Ежедневно | ✓✓ | Высокая | Средняя |
| PCA(RND moments) | Наш метод | Опционы | Ежедневно | ✓✓ | Средняя | Высокая |
| Entropy(RND) | Информ. теория | Опционы | Ежедневно | ✓✓ | Средняя | Средняя |
| RV ключевой ставки | Chang & Feunou (2013) | Ставки | Ежедневно | ✗ (запаздывает) | Высокая | Низкая |
| NLP (пресс-релизы) | Husted et al. (2020) | Тексты | Ежемесячно | ✓ | Высокая | Средняя |
| Survey dispersion | Dahlhaus et al. (2018) | Опросы | Ежемесячно | ✓ | Очень высокая | Низкая |
| RWD (CRRA) | Liu et al. (2007) | Опционы + история | Ежедневно | ✓✓ | Средняя | Высокая |

---

## Приложение: Данные для российского рынка

### Источники данных ЦБ РФ и Московской биржи

```python
# Данные ЦБ РФ (API)
# https://www.cbr.ru/development/DWS/

# 1. Ключевая ставка: 
# https://www.cbr.ru/statistics/avgprocstav/ или API CBR

# 2. RUONIA (Overnight):
# https://cbr.ru/hd_base/ruonia/

# 3. OФЗ кривая:
# https://www.cbr.ru/statistics/gdcurve/

# 4. Данные Московской биржи (MOEX):
# Опционы на RUSFAR (RUONIA futures) - секция FORTS
# API: https://iss.moex.com/iss/

import requests

def get_cbr_key_rate(start_date, end_date):
    """Загрузка ключевой ставки ЦБ РФ через API."""
    url = "https://cbr.ru/hd_base/KeyRate/"
    params = {
        'UniDbQuery.Posted': 'True',
        'UniDbQuery.From': start_date,
        'UniDbQuery.To':   end_date,
    }
    # Для реальной загрузки использовать cbr-xml-daily.ru API или официальный API ЦБ
    # Здесь псевдокод:
    print(f"Загрузка ключевой ставки ЦБ РФ: {start_date} – {end_date}")
    # response = requests.get(url, params=params)
    # return pd.read_html(response.text)[0]
    return None

def get_moex_options_data(ticker, date):
    """
    Загрузка опционных котировок с Московской биржи через MOEX ISS API.
    
    Тикер: RI - фьючерс РТС, RU - RUONIA, etc.
    """
    url = f"https://iss.moex.com/iss/engines/futures/markets/options/boards/ROPD/securities.json"
    params = {'date': date.strftime('%Y-%m-%d'), 'secid': ticker}
    print(f"MOEX API: загрузка опционов {ticker} за {date}")
    # response = requests.get(url, params=params)
    # data = response.json()
    # return parse_moex_options(data)
    return None
```

### Структура входных данных

```python
# Минимальный формат входных данных (CSV)
# date, tenor, strike, iv, forward, type (call/put)
# 
# 2024-01-15, 3M, 0.14, 0.185, 0.155, call
# 2024-01-15, 3M, 0.16, 0.170, 0.155, call
# 2024-01-15, 3M, 0.18, 0.175, 0.155, call
# ...

# Для российского рынка: если рыночные опционы недоступны,
# можно реконструировать IV поверхность из:
# 1. OIS/IRS котировок (RUONIA-based swaps)
# 2. Форвардных ставок из кривой ОФЗ
# 3. Процентных опционов (caps/floors) на межбанковском рынке

# Приближённая IV из волатильности форвардных ставок:
def iv_from_forward_rates(forward_rates_series, tenor_days):
    """
    Оценка ATM IV из исторической волатильности форвардных ставок.
    Используется как суррогат при недостатке рыночных данных.
    """
    log_changes = np.log(forward_rates_series / forward_rates_series.shift(1)).dropna()
    historical_vol = log_changes.rolling(window=30).std() * np.sqrt(252)
    return historical_vol * 1.1  # небольшая надбавка за risk premium
```

