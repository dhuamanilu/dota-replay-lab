# Anticipación causal de combate

Las órdenes `attack` mezclan last hits, denies, estructuras y héroes. Para evitar
usar esa señal ambigua como si toda ella fuera una pelea, el extractor agrega el
combat log y normaliza los nombres de héroes. El objetivo `engage` indica daño
infligido a otro héroe y `threat` daño recibido por el héroe observado.

El modelo ve el estado hasta el segundo `t` y predice si habrá alguna de esas
señales entre `t+1` y `t+5`. Sus entradas son economía, posición, movimiento y
acción previos, contexto espacial de los diez héroes y daño reciente. El split,
la selección de umbrales y el bootstrap son por partida.

```powershell
python -m dota_replay_lab.parse_replay_corpus `
  --manifest artifacts/corpora/pro-matches-v1.json `
  --count 20 `
  --output-dir artifacts/replay-combat-trajectories
python -m dota_replay_lab.train_replay_combat --horizon 5
```

Se probó primero el horizonte de un segundo. Obtuvo macro-F1 0,6587 frente a
0,6575 de persistencia, pero el IC agrupado cruzó cero; ese candidato se rechazó.
Con cinco segundos y 20 partidas, el modelo completo obtuvo macro-F1 0,6597
frente a 0,5548. Ganó en las cuatro partidas de test; el bootstrap agrupado dio
delta medio +0,1065 e IC 95 % `[+0,0919, +0,1215]`.

Se hizo después una ablación estricta de campos no disponibles en el runtime:
XP, net worth y daño reciente. Con las 21 señales restantes obtuvo 0,6195 frente
a 0,5548, ganó las cuatro partidas y mantuvo delta +0,0707 con IC 95 %
`[+0,0405, +0,0948]`. Este es el candidato que se exporta.

`export_combat_lua` destila los 300 árboles a
`bots/replay_combat_policy.lua`. En 1.000 estados, las clases coincidieron
1.000/1.000 y el error máximo de probabilidad fue `1,11e-16`. El adaptador solo
permite una pelea oportunista si `engage_probability >= 0,90` y ya existe un
enemigo a 900 unidades. En validación, ese umbral priorizó precisión sobre recall;
la retirada por poca vida sigue teniendo prioridad absoluta.

Esta integración está respaldada para anticipación offline y paridad numérica,
no para win rate: todavía requiere una sesión autónoma futura dentro del juego.
