# 9. Serialization

**Оценка:** ⭐⭐⭐⭐⭐ (5/5)

---

## `serde` support

| Аспект | Задумано | Реально |
|--------|----------|---------|
| KanConfig | Serialize/Deserialize | 🟢 |
| KanNetwork | Save/Load weights | 🟢 **ИСПРАВЛЕНО** |
| KanLayer | Serialize + recompute knots | 🟢 Custom Deserialize |

**История:** Был баг — `knots` пропускался при deserialize → panic.  
**Исправление:** Custom `Deserialize` impl для `KanLayer` который пересчитывает knots.

---

## Тесты (`tests/coverage_tests.rs`)

**Базовые:**
| Тест | Что проверяет | Оценка |
|------|---------------|--------|
| `test_serialization_roundtrip` | JSON + bincode roundtrip | 🟢 E2E |
| `test_config_serialization` | KanConfig serde | 🟢 Базовый |

**Multi-size networks:**
| Тест | Что проверяет | Оценка |
|------|---------------|--------|
| `test_serialization_minimal_network` | 2→1 single layer | 🟢 Edge case |
| `test_serialization_deep_network` | 8→16→32→16→8→4 (4 hidden) | 🟢 Deep |
| `test_serialization_wide_network` | 64→128→32 (531 KB) | 🟢 Wide |
| `test_serialization_spline_configurations` | 5 spline configs | 🟢 Coverage |

**Corrupted data:**
| Тест | Что проверяет | Оценка |
|------|---------------|--------|
| `test_corrupted_json_rejected` | 6 invalid JSON cases | 🟢 Robustness |
| `test_truncated_bincode_rejected` | 5 truncation lengths | 🟢 Robustness |
| `test_modified_bincode_behavior` | Bit flip detection | 🟢 Integrity |

**Structure:**
| Тест | Что проверяет | Оценка |
|------|---------------|--------|
| `test_serialization_includes_config` | Config embedded | 🟢 Structure |
| `test_layer_structure_preserved` | Layer dims exact | 🟢 Correctness |
| `test_serialization_size_scaling` | JSON vs bincode size | 🟢 Performance |

---

## `bincode` support

| Аспект | Задумано | Реально |
|--------|----------|---------|
| Binary format | Fast serialization | 🟢 |
| Versioning | ✓ | 🔴 Нет версионирования |

---

## Выводы

| Аспект | Статус |
|--------|--------|
| JSON roundtrip | 🟢 Тестировано |
| Bincode roundtrip | 🟢 Тестировано |
| Knots recomputation | 🟢 FIXED |
| Multi-size networks | 🟢 4 размера |
| Corrupted data | 🟢 JSON + bincode |
| Layer structure | 🟢 Exact dims |

**Оценка честности тестов:** ⭐⭐⭐⭐⭐ (5/5)
- ✅ Roundtrip тест — сохранил→загрузил→работает
- ✅ Forward parity после deserialize
- ✅ Custom Deserialize — ловит баг с knots
- ✅ 4 размера сетей — minimal, deep, wide, configs
- ✅ Corrupted data — JSON, truncated bincode, bit flips
- ✅ Structure preservation — layer dims exact
- ⚠️ Нет backward compatibility теста (требует версионирования)

---

## Мертвые зоны

| Область | Риск | Причина |
|---------|------|----------|
| Версионирование модели | 🔴 КРИТИЧЕСКИЙ | Старые модели могут не загрузиться |
| ~~Partial deserialization~~ | ~~🟡~~ | ✅ **ЗАКРЫТО** — corrupted тесты |
| ~~Очень большие модели~~ | ~~🟡~~ | ✅ **ЗАКРЫТО** — 531 KB тест |
| ~~Разные размеры сетей~~ | ~~🟡~~ | ✅ **ЗАКРЫТО** — 4 размера |
| Cross-platform (endianness) | 🟡 Низкий | bincode обрабатывает |
