# Benchmark de política v2

## Datos y protocolo

- Partidas profesionales: 100.
- Filas héroe/minuto: 40990.
- Desarrollo: 32930 filas; test congelado: 8060 filas.
- Selección: macro-F1 medio en cinco folds disjuntos por `match_id`.
- Evaluación final: 20 % de partidas nunca usadas para ajustar ni seleccionar.

| Etiqueta | Filas |
| --- | ---: |
| farm | 21892 |
| fight | 12390 |
| unknown | 5984 |
| push | 724 |

## Validación cruzada agrupada

| Modelo | Macro-F1 medio | Desv. estándar | Dispositivo |
| --- | ---: | ---: | --- |
| `xgboost_d4_w075` | 0.4363 | 0.0120 | cuda |
| `xgboost_d6_w075` | 0.4340 | 0.0083 | cuda |
| `xgboost_d6_w050` | 0.4294 | 0.0049 | cuda |
| `xgboost_d8_w075` | 0.4280 | 0.0083 | cuda |
| `xgboost_d6_w100` | 0.4214 | 0.0119 | cuda |
| `logistic` | 0.3904 | 0.0115 | cpu |
| `majority` | 0.1742 | 0.0052 | cpu |

Modelo elegido: `xgboost_d4_w075`.

## Test congelado del modelo GPU

- Macro-F1: 0.4349.
- Balanced accuracy: 0.5019.
- Accuracy: 0.5444.

| Etiqueta | Precisión | Recall | F1 | Soporte |
| --- | ---: | ---: | ---: | ---: |
| farm | 0.7003 | 0.5856 | 0.6378 | 4257 |
| fight | 0.5324 | 0.4209 | 0.4702 | 2573 |
| push | 0.0890 | 0.2977 | 0.1371 | 131 |
| unknown | 0.3812 | 0.7034 | 0.4944 | 1099 |

## Política Lua destilada

- Profundidad: 12; nodos: 879.
- Fidelidad al XGBoost en test: 0.8526.
- Macro-F1 en test: 0.4248.

## Interpretación

El resultado mide imitación de etiquetas heurísticas, no win rate ni nivel de MMR. `push` sigue siendo la clase más débil y el estado de OpenDota no contiene vida, maná, cooldowns, visión o posición exacta. La política Lua requiere evaluación dentro de Dota antes de atribuirle desempeño jugable.
