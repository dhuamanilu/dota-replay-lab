# Benchmark de política v2

## Datos y protocolo

- Partidas profesionales: 500.
- Filas héroe/minuto: 210150.
- Desarrollo: 166980 filas; test congelado: 43170 filas.
- Selección: macro-F1 medio en cinco folds disjuntos por `match_id`.
- Evaluación final: 20 % de partidas nunca usadas para ajustar ni seleccionar.

| Etiqueta | Filas |
| --- | ---: |
| farm | 110975 |
| fight | 63607 |
| unknown | 31805 |
| push | 3763 |

## Validación cruzada agrupada

| Modelo | Macro-F1 medio | Desv. estándar | Dispositivo |
| --- | ---: | ---: | --- |
| `xgboost_d6_w050` | 0.4567 | 0.0017 | cuda |
| `xgboost_d8_w075` | 0.4535 | 0.0022 | cuda |
| `xgboost_d6_w075` | 0.4521 | 0.0023 | cuda |
| `xgboost_d4_w075` | 0.4490 | 0.0027 | cuda |
| `xgboost_d6_w100` | 0.4176 | 0.0026 | cuda |
| `logistic` | 0.3950 | 0.0010 | cpu |
| `majority` | 0.1719 | 0.0009 | cpu |

Modelo elegido: `xgboost_d6_w050`.
Sesgos calibrados con predicciones out-of-fold: farm=1, fight=1, push=1, unknown=0.75.
Macro-F1 out-of-fold tras calibración: 0.4573.

## Test congelado del modelo GPU

- Macro-F1: 0.4488.
- Balanced accuracy: 0.4531.
- Accuracy: 0.6003.

| Etiqueta | Precisión | Recall | F1 | Soporte |
| --- | ---: | ---: | ---: | ---: |
| farm | 0.6908 | 0.7162 | 0.7032 | 23463 |
| fight | 0.5323 | 0.4342 | 0.4783 | 12347 |
| push | 0.1266 | 0.1057 | 0.1152 | 766 |
| unknown | 0.4512 | 0.5564 | 0.4983 | 6594 |

## Política Lua destilada

- Estrategia: teacher_calibrated; profundidad: 12; nodos: 1699.
- Fidelidad al XGBoost en test: 0.9133.
- Macro-F1 en test: 0.4420.

## Interpretación

El resultado mide imitación de etiquetas heurísticas, no win rate ni nivel de MMR. `push` sigue siendo la clase más débil y el estado de OpenDota no contiene vida, maná, cooldowns, visión o posición exacta. La política Lua requiere evaluación dentro de Dota antes de atribuirle desempeño jugable.
