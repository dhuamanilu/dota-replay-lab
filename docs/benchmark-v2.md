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
Sesgos calibrados con predicciones out-of-fold: farm=1, fight=1, push=0.75, unknown=0.75.
Macro-F1 out-of-fold tras calibración: 0.4448.

## Test congelado del modelo GPU

- Macro-F1: 0.4372.
- Balanced accuracy: 0.4683.
- Accuracy: 0.5640.

| Etiqueta | Precisión | Recall | F1 | Soporte |
| --- | ---: | ---: | ---: | ---: |
| farm | 0.6850 | 0.6310 | 0.6569 | 4257 |
| fight | 0.5221 | 0.4508 | 0.4838 | 2573 |
| push | 0.0861 | 0.1756 | 0.1156 | 131 |
| unknown | 0.4103 | 0.6160 | 0.4925 | 1099 |

## Política Lua destilada

- Estrategia: teacher_raw; profundidad: 12; nodos: 879.
- Fidelidad al XGBoost en test: 0.8448.
- Macro-F1 en test: 0.4248.

## Interpretación

El resultado mide imitación de etiquetas heurísticas, no win rate ni nivel de MMR. `push` sigue siendo la clase más débil y el estado de OpenDota no contiene vida, maná, cooldowns, visión o posición exacta. La política Lua requiere evaluación dentro de Dota antes de atribuirle desempeño jugable.
