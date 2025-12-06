# 11. Loss Functions

**Оценка:** ⭐⭐⭐⭐⭐ (5/5)

---

## 11.1 Standard Task-Specific Losses

### Regression Losses
| Функция | Назначение | Статус |
|---------|------------|--------|
| `masked_mse` | MSE с маской | 🟢 |
| `masked_rmse` | RMSE в оригинальных единицах | 🟢 |
| `masked_mae` | MAE устойчива к выбросам | 🟢 |
| `masked_huber` | Smooth L1 | 🟢 |

### Classification Losses
| Функция | Назначение | Статус |
|---------|------------|--------|
| `masked_cross_entropy` | BCE для вероятностей | 🟢 |
| `masked_bce_with_logits` | BCE для логитов | 🟢 |
| `masked_categorical_cross_entropy` | Мультиклассовая CE | 🟢 |

---

## 11.2 KAN-Specific Regularization ✨

| Функция | Назначение | Статус |
|---------|------------|--------|
| `l1_sparsity_loss` | L1 норма для разреженности | 🟢 |
| `l1_sparsity_gradient` | Субградиент L1 | 🟢 |
| `entropy_regularization` | Штраф за энтропию | 🟢 |
| `smoothness_penalty` | Штраф за вторую производную | 🟢 |
| `smoothness_gradient` | Градиент smoothness | 🟢 |
| `kan_combined_loss` | L_total = L_pred + λ₁L₁ + λ₂H + λ₃L_smooth | 🟢 |

---

## 11.3 Physics-Informed & Symbolic Regression

| Функция | Назначение | Статус |
|---------|------------|--------|
| `pde_residual_loss` | Residual loss для PDE | 🟢 |
| `r_squared` | R² для symbolic regression | 🟢 |

---

## 11.4 PyTorch Cross-Entropy Parity Tests

| Тест | Что проверяет | Оценка |
|------|---------------|--------|
| `test_cross_entropy_pytorch_perfect_prediction` | BCE pred=[0.9,0.1] vs PyTorch | 🟢 |
| `test_cross_entropy_pytorch_confident_wrong` | BCE pred=[0.1,0.9] vs PyTorch | 🟢 |
| `test_cross_entropy_pytorch_uncertain` | BCE pred=[0.5,0.5] = ln(2) | 🟢 |
| `test_cross_entropy_pytorch_multiclass` | BCE 4 classes | 🟢 |
| `test_cross_entropy_pytorch_soft_targets` | BCE soft labels | 🟢 |
| `test_cross_entropy_gradient_direction` | grad sign correctness | 🟢 |
| `test_cross_entropy_with_mask` | CE mask support | 🟢 |
| `test_cross_entropy_numerical_stability` | No NaN/Inf near 0,1 | 🟢 |

---

## 11.5 Другие тесты

| Тест | Что проверяет | Оценка |
|------|---------------|--------|
| `test_masked_mse` | MSE без маски | 🟢 |
| `test_rmse_vs_mse` | RMSE = √MSE | 🟢 |
| `test_mae_robust_to_outliers` | MAE < MSE для выбросов | 🟢 |
| `test_bce_logits_gradient` | BCE grad = sigmoid - target | 🟢 |
| `test_l1_gradient` | L1 grad = sign/n | 🟢 |
| `test_entropy_uniform` | High entropy для uniform | 🟢 |
| `test_entropy_concentrated` | Low entropy для concentrated | 🟢 |
| `test_smoothness_linear` | Smooth=0 для линейных | 🟢 |
| `test_r_squared_perfect` | R²=1 для perfect | 🟢 |
| `test_kan_combined_basic` | Combined loss finite | 🟢 |
| `test_huber_loss` | Huber < MSE для выбросов | 🟢 |

---

## Выводы

| Аспект | Статус |
|--------|--------|
| Regression losses | 🟢 Полное (MSE, RMSE, MAE, Huber) |
| Classification losses | 🟢 Полное (BCE, CE) |
| KAN regularization | 🟢 Полное (L1, Entropy, Smoothness) |
| Combined losses | 🟢 Тестировано |
| Physics-informed | 🟢 Базовое |
| Symbolic regression | 🟢 Тестировано |

**Оценка честности тестов:** ⭐⭐⭐⭐⭐ (5/5)
- ✅ Все формулы проверены численно
- ✅ Свойства (MAE robustness, entropy ordering) тестируются
- ✅ PyTorch parity — 8 тестов с точностью 1e-5
- ✅ Gradient формулы проверены

---

## Мертвые зоны

| Область | Риск | Причина |
|---------|------|----------|
| ~~PyTorch parity~~ | ~~🟡~~ | ✅ **ЗАКРЫТО** — 8 тестов |
| Numerical stability extreme | 🟡 Средний | log(ε), exp(big) частично покрыто |
| Training loop integration | 🟡 Средний | kan_combined_loss требует manual wiring |
| GPU loss functions | 🔴 Высокий | Loss вычисляется на CPU |
