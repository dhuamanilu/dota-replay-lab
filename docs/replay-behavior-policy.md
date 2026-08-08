# Política de comportamiento por segundo

Esta política auxiliar aprende de órdenes reales extraídas de replays
profesionales. Su objetivo es estimar por separado la probabilidad de que el
jugador emita una orden de movimiento, ataque o lanzamiento durante el segundo
actual. No confunde este objetivo con win rate ni con reinforcement learning.

## Causalidad y split

Las entradas incluyen economía y posición actuales, movimiento ya observado,
acción del segundo anterior y contexto espacial simultáneo: distancia al aliado
y enemigo más cercanos y conteos cercanos. Nunca se incluyen las órdenes que se
están prediciendo. Las partidas completas se separan 60/20/20 antes de ajustar
la normalización; los umbrales de cada acción se eligen solo con validación.

La red usa embeddings de héroe/equipo y un MLP `96 → 48 → 3`, con BCE
multi-label. Se compara contra persistencia, un baseline fuerte que simplemente
repite la presencia o ausencia de cada orden en el segundo anterior.

```powershell
python -m dota_replay_lab.train_replay_behavior --device cuda
```

## Resultado inicial de 10 partidas

Sobre 250.310 filas, la época 17 obtuvo macro-F1 0,6227 en validación y 0,6144
en las dos partidas de test congeladas. Persistencia obtuvo 0,5795 en ese mismo
test. Por acción:

| Señal | MLP causal | Persistencia |
| --- | ---: | ---: |
| `move` | 0,8748 | 0,8395 |
| `attack` | 0,5798 | 0,5600 |
| `cast` | 0,3888 | 0,3391 |

La mejora inicial es consistente entre las tres salidas, pero diez partidas no
bastan para promover el modelo al runtime. Además, una orden `attack` no revela
por sí sola si el objetivo era un creep o un héroe, y los eventos disponibles no
identifican el objetivo de cada orden. El checkpoint queda como candidato hasta
superar una auditoría con más partidas y una prueba de integración separada.
