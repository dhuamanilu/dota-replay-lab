# Política secuencial GRU

La política v3 usa una GRU causal pequeña en lugar de tratar cada minuto como una fila independiente. Cada secuencia contiene el historial completo de un héroe dentro de una partida; el estado del futuro nunca entra en el paso actual. Los `match_id` se separan antes de normalizar: 60 % para ajuste, 20 % para seleccionar la época y 20 % como test congelado.

## Arquitectura

- Embedding de héroe de 16 dimensiones y embedding de equipo de 2.
- Dos capas GRU de 96 unidades con dropout 0,2 durante entrenamiento.
- LayerNorm y cabeza `96 → 64 → 4` para `farm`, `fight`, `push` y `unknown`.
- Cross-entropy con pesos de clase elevados a 0,5, AdamW, clipping de gradiente y parada temprana.
- 414.574 bytes en el checkpoint PyTorch y 2,19 MB en el Lua autocontenido.

En la GTX 1660 Ti, PyTorch 2.13.0 con CUDA 13.0 entrenó la configuración elegida en 34,7 segundos y seleccionó la época 15. La multiplicación 4.096 × 4.096 usada para comprobar CUDA tardó 0,0737 segundos.

## Resultado congelado

| Modelo | Selección/validación macro-F1 | Test macro-F1 | Accuracy | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| XGBoost canónico | 0,4556 (CV de 5 folds) | 0,4533 | 0,5766 | 0,4908 |
| GRU causal | 0,4705 (validación fija) | 0,4591 | 0,5957 | 0,4900 |

La GRU mejora 0,0058 de macro-F1 y 0,0191 de accuracy en el mismo test congelado. El F1 de `push` es 0,1541; sigue siendo bajo y no justifica afirmar buena toma de torres por sí solo.

Se evaluó además una ablación de denies. Con las señales presentes, XGBoost obtuvo 0,4564 de CV y 0,4492 en test; al anularlas obtuvo 0,4567 y 0,4526. Por esa evidencia, el dataset v3 conserva denies para análisis y telemetría, pero la política canónica no los usa como entrada aprendida. El runtime sí puede ejecutar denies mediante su regla táctica local.

## Reproducción

```powershell
python -m pip install -e ".[ml,dev,dl]"
python -m pip install --upgrade torch==2.13.0+cu130 `
  --index-url https://download.pytorch.org/whl/cu130
$env:PYTHONPATH='src'
python -m dota_replay_lab.train_sequence_policy --device cuda
python -m dota_replay_lab.export_sequence_lua --parity-rows 5000
python -m pytest -q
```

La exportación implementa las ecuaciones GRU, LayerNorm y las capas lineales directamente en Lua. En 5.000 checkpoints congelados, las 5.000 clases coincidieron con PyTorch. Esto demuestra paridad numérica offline, no win rate ni compatibilidad nueva dentro del cliente de Dota.
