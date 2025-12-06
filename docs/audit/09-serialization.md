# 9. Serialization

**Оценка:** ⭐⭐⭐ (3/5)

**⚠️ ПРОБЛЕМА:** Все тесты под `#[cfg(feature = "serde")]`:
- По умолчанию `cargo test` НЕ запускает эти тесты
- CI также НЕ запускает serde тесты
- Нужно: `cargo test --features serde`

---

## 9.1 `serde` support

| Аспект | Задумано | Реально |
|--------|----------|---------|
| KanConfig | Serialize/Deserialize | 🟢 |
| KanNetwork | Save/Load weights | 🟢 **ИСПРАВЛЕНО** |
| KanLayer | Serialize + recompute knots | 🟢 Custom Deserialize |

**История:** Был баг — `knots` пропускался при deserialize → panic.  
**Исправление:** Custom `Deserialize` impl для `KanLayer` который пересчитывает knots.

---

## 9.2 Тесты (`tests/coverage_tests.rs`)

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

## 9.3 `bincode` support

| Аспект | Задумано | Реально |
|--------|----------|---------|
| Binary format | Fast serialization | 🟢 |
| Versioning | ✓ | 🔴 Нет версионирования |

---

## 9.4 Выводы

| Аспект | Статус |
|--------|--------|
| JSON roundtrip | 🟡 **Тесты под feature flag** |
| Bincode roundtrip | 🟡 **Тесты под feature flag** |
| Knots recomputation | 🟢 FIXED |
| Multi-size networks | 🟡 **Тесты под feature flag** |
| Corrupted data | 🟡 **Тесты под feature flag** |
| Layer structure | 🟡 **Тесты под feature flag** |

**Оценка честности тестов:** ⭐⭐ (2/5)
- ❌ **Все тесты под `#[cfg(feature = "serde")]`**
- ❌ **CI не запускает serde тесты**
- ❌ **Регрессии могут оставаться незамеченными**
- ✅ Если запустить вручную — тесты полные
- ⚠️ Нет backward compatibility теста (требует версионирования)

---

## 9.5 Мертвые зоны

| Область | Риск | Причина |
|---------|------|----------|
| Версионирование модели | 🔴 КРИТИЧЕСКИЙ | Старые модели могут не загрузиться |
| ~~Partial deserialization~~ | ~~🟡~~ | ✅ **ЗАКРЫТО** — corrupted тесты |
| ~~Очень большие модели~~ | ~~🟡~~ | ✅ **ЗАКРЫТО** — 531 KB тест |
| ~~Разные размеры сетей~~ | ~~🟡~~ | ✅ **ЗАКРЫТО** — 4 размера |
| Cross-platform (endianness) | 🟡 Низкий | bincode обрабатывает |

---

## 9.6 Место для оптимизации

| Область | Тип | Сложность | Описание |
|---------|-----|-----------|----------|
| Model versioning | 🔧 Feature | 🟡 Средняя | Версия в заголовке, migration при загрузке |
| ONNX export | 🔧 Feature | 🔴 Высокая | Экспорт в ONNX для inference в других фреймворках |
| Streaming serialization | 🚀 Perf | 🟡 Средняя | Загрузка частями для больших моделей |
| Compression (zstd) | 🚀 Perf | 🟢 Низкая | Сжатие для уменьшения размера файлов |
| Checkpointing | 🔧 Feature | 🟡 Средняя | Сохранение optimizer state для resume training |
