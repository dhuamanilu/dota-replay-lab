# Offline RL sobre repeticiones

Este experimento introduce recompensas causales inspiradas en OpenAI Five sin afirmar que un
replay sea un simulador contrafactual. La fila del minuto `m` contiene el estado anterior a la
decisión y la fila contigua `m + 1` contiene los contadores observados del resultado. Las
transiciones finales, discontinuas o pertenecientes a otro héroe se invalidan.

## Recompensa

La recompensa disponible combina oro (`0,006`), experiencia (`0,002`), last hits (`0,40`),
denies (`0,15`) y kills (`1,0`). `team_spirit` interpola recompensa individual y media aliada;
después se resta la media rival del mismo `match_id` y minuto para obtener una señal zero-sum.
En el corpus original hay 205.270 transiciones válidas de 210.270 filas y la media zero-sum
residual es inferior a `2e-16`.

La ponderación por ventaja calcula mediana y MAD robusta por acción exclusivamente en partidas
de entrenamiento. El peso es `exp(advantage / beta)`, limitado a `[0,25, 4]`. Esto es imitación
ponderada por resultados observados, no PPO ni una estimación de qué habría ocurrido al elegir
otra acción.

## Ablaciones

Todos los candidatos parten del checkpoint GRU congelado y se seleccionan por validación.

| Candidato | Test original macro-F1 | Audit externo macro-F1 |
| --- | ---: | ---: |
| GRU vigente | 0,4591 | 0,4610 |
| Continuación supervisada | 0,4604 | 0,4628 |
| Ventaja ponderada, `beta=8` | 0,4608 | 0,4605 |
| Ventaja + cabeza auxiliar, peso `0,02` | 0,4570 | no promovido |
| Ventaja + cabeza auxiliar, peso `0,10` | 0,4575 | no promovido |
| GRU desde cero con 580 partidas | 0,4652 interno | 0,4578 |

El audit externo contiene 100 partidas y 38.010 filas sin solapamiento con las 500 partidas
originales ni con las 80 de expansión. La continuación supervisada mejora `0,00175` sobre el
audit, pero un bootstrap de 5.000 remuestreos agrupados por partida produce IC 95 %
`[-0,00237, 0,00575]` y probabilidad de mejora `0,8058`. Como el intervalo incluye cero, ningún
candidato reemplaza la política Lua vigente.

## Resultado

Se conservan como infraestructura experimental:

- construcción causal de recompensas y retornos;
- recompensa zero-sum y `team_spirit`;
- ponderación conservadora por ventaja;
- predicción auxiliar de recompensa desde el estado recurrente;
- inicialización desde un checkpoint local confiable;
- manifiestos excluyentes, combinación de corpus y reanudación sólo desde caché.

El siguiente paso válido requiere más estados causales —idealmente posición, vida, maná,
cooldowns y acciones desde `.dem`— o un entorno ejecutable que permita observar consecuencias
contrafactuales. No se debe llamar self-play a optimizar únicamente sobre transiciones fijas.
