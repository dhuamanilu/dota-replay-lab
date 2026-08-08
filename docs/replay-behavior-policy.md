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

## Resultado auditado de 20 partidas

Sobre 499.230 filas, la época 22 obtuvo macro-F1 0,6225 en validación y 0,6378
en cuatro partidas de test completamente separadas. Persistencia obtuvo 0,6023
en ese mismo test. Por acción:

| Señal | MLP causal | Persistencia |
| --- | ---: | ---: |
| `move` | 0,8844 | 0,8564 |
| `attack` | 0,5961 | 0,5671 |
| `cast` | 0,4330 | 0,3835 |

El candidato superó al baseline en cada una de las cuatro partidas de test. Un
bootstrap de 5.000 muestras, re-muestreando partidas completas, estimó delta
medio +0,0366, IC 95 % `[+0,0317, +0,0416]` y probabilidad positiva 1,0.

Esta evidencia permite aceptar el checkpoint como mejor imitador offline de
órdenes, pero no promoverlo al runtime principal. Una orden `attack` no revela
por sí sola si el objetivo era un creep o un héroe, y cuatro partidas de test no
miden win rate. La integración queda condicionada al clasificador de combate y
a una evaluación separada del bot ejecutable.
