# Benchmark de política v3

## Datos y protocolo

- Partidas profesionales: 500.
- Filas héroe/minuto: 210270.
- Desarrollo: 168200 filas; test congelado: 42070 filas.
- Selección XGBoost: macro-F1 medio en cinco folds disjuntos por `match_id`.
- Evaluación final: 20 % de partidas nunca usadas para ajustar ni seleccionar.

| Etiqueta | Filas |
| --- | ---: |
| farm | 110882 |
| fight | 63861 |
| unknown | 31781 |
| push | 3746 |

## Validación cruzada agrupada

| Modelo | Macro-F1 medio | Desv. estándar | Dispositivo |
| --- | ---: | ---: | --- |
| `xgboost_d6_w050` | 0.4556 | 0.0035 | cuda |
| `xgboost_d8_w075` | 0.4543 | 0.0024 | cuda |
| `xgboost_d6_w075` | 0.4531 | 0.0030 | cuda |
| `xgboost_d4_w075` | 0.4513 | 0.0015 | cuda |
| `xgboost_d6_w100` | 0.4199 | 0.0007 | cuda |
| `logistic` | 0.3960 | 0.0004 | cpu |
| `majority` | 0.1726 | 0.0023 | cpu |

Modelo elegido: `xgboost_d6_w050`.
Sesgos calibrados con predicciones out-of-fold: farm=1, fight=1, push=1.5, unknown=1.
Macro-F1 out-of-fold tras calibración: 0.4616.

## Test congelado del baseline XGBoost

- Macro-F1: 0.4533.
- Balanced accuracy: 0.4908.
- Accuracy: 0.5766.

| Etiqueta | Precisión | Recall | F1 | Soporte |
| --- | ---: | ---: | ---: | ---: |
| farm | 0.6943 | 0.6578 | 0.6756 | 22237 |
| fight | 0.5658 | 0.4187 | 0.4812 | 12874 |
| push | 0.1162 | 0.2330 | 0.1551 | 734 |
| unknown | 0.4068 | 0.6537 | 0.5015 | 6225 |

## Política recurrente portable

- Arquitectura: GRU causal de dos capas, 96 unidades ocultas.
- Entrenamiento: cuda en NVIDIA GeForce GTX 1660 Ti, 34.7 segundos.
- Selección de época: validación fija del 20 % de partidas; el test no participa.
- Macro-F1 de validación: 0.4705.
- Macro-F1 en test: 0.4591.
- Paridad Lua/PyTorch: 5000/5000 (1.0000).
- Tamaño Lua autocontenido: 2.19 MB.

## Interpretación

El resultado mide imitación de etiquetas heurísticas, no win rate ni nivel de MMR. `push` sigue siendo la clase más débil y el estado de OpenDota no contiene vida, maná, cooldowns, visión o posición exacta. La GRU mejora la imitación secuencial, pero su exportación Lua solo tiene paridad offline; no se atribuye habilidad jugable sin una evaluación del runtime.
