# Self-play coordinado y reproducible

Esta etapa incorpora reinforcement learning real al proyecto, pero con una frontera
importante: el agente se autoentrena dentro de un simulador causal calibrado con
repeticiones, no dentro del motor completo de Dota 2. Por tanto, los resultados de
esta página son victorias en el *surrogate*; no son win rate, MMR ni prueba de nivel
dentro del cliente.

## Qué aprende

La política empieza con los pesos de la GRU de imitación histórica y conserva sus
13 características causales. Cada minuto observa cinco héroes aliados y elige una
composición conjunta de `farm`, `fight`, `push` y `unknown`. Hay 56 composiciones
posibles para cinco héroes. Una cabeza central selecciona la composición y las
preferencias individuales de la GRU asignan los roles a los héroes.

El optimizador es PPO recurrente, con secuencias completas, crítico de equipo,
recompensa zero-sum, oponentes guionados y snapshots anteriores de la propia
política. Un término de imitación sobre replays evita olvidar por completo la
conducta profesional inicial. La distribución final mezcla 75 % de la cabeza
aprendida con 25 % del prior histórico de composiciones para mantener soporte en
acciones poco frecuentes.

Esto no aprende las reglas desde píxeles. Las reglas de transición están codificadas
en `selfplay_env.py`; los ritmos de oro, experiencia, last hits, kills y push se
ajustan usando solo las partidas del split de entrenamiento. El split de test se
reserva para auditar calibración y retención.

## Correcciones y candidatos rechazados

El primer simulador usaba incrementos de recursos condicionados por la etiqueta
observada. Una política podía explotar ese efecto posterior eligiendo pelea o push
casi siempre y obteniendo 100 % artificial. Ese resultado fue rechazado.

La versión promovida ancla la disponibilidad base de recursos a las filas de farm.
Pelear, pushear o quedar `unknown` aplican costes de oportunidad explícitos; la
ganancia de pelea depende del rival y el push convierte la tasa histórica por
jugador en una oportunidad de equipo con retornos sublineales. También se
rechazaron dos PPO independientes: uno olvidó la imitación histórica y otro, al
regularizarse, dejó de superar rivales balanceados y agresivos.

## Entrenamiento promovido

Comando principal ejecutado en la GTX 1660 Ti:

```powershell
$env:PYTHONPATH='src'
python -m dota_replay_lab.train_team_selfplay `
  --device cuda --iterations 60 --environments 128 `
  --max-minutes 45 --evaluation-games 1000
```

El entrenamiento tardó 53,0 s. En la evaluación primaria obtuvo:

| Rival | Partidas | Victoria | IC Wilson 95 % |
| --- | ---: | ---: | ---: |
| GRU de imitación original | 1.000 | 89,7 % | 87,66–91,43 % |
| Guion balanceado | 1.000 | 79,2 % | 76,57–81,60 % |
| Guion agresivo | 1.000 | 63,5 % | 60,47–66,43 % |
| Guion de farm | 1.000 | 98,5 % | 97,54–99,09 % |

El control simétrico fue 50,4 % con IC 47,31–53,49 %. El macro-F1 congelado
sobre replays cambió de 0,45906 a 0,45695 (`-0,00211`), dentro del límite de
retención de `-0,01`.

La auditoría independiente repitió 400 partidas en cada una de cinco semillas
(2.000 por rival). Dio 87,60 % contra imitación, 78,80 % contra balanceado,
63,15 % contra agresivo y 98,70 % contra farm. El control simétrico fue 49,95 %.
Todos los límites inferiores Wilson superaron 50 % y la diversidad de acciones
cumplió el mínimo derivado de los replays.

## Exportación e integración

```powershell
python -m dota_replay_lab.audit_team_selfplay --device cuda
python -m dota_replay_lab.export_team_selfplay_lua --parity-rows 250
```

El exportador se niega a escribir una política si falla la auditoría multisemilla.
La versión incluida en `bots/team_selfplay_policy.lua` coincidió en los 250 planes
de equipo comparados con PyTorch; el error máximo de probabilidad fue
`2,94e-7`. `bots/team_selfplay_policy.metrics.json` conserva entrenamiento,
auditoría y paridad.

`bot_generic.lua` consulta los cinco miembros mediante la API de Valve, construye
un plan común por minuto y usa la acción correspondiente a cada héroe. El ejemplo
oficial instalado con Dota confirma la forma `GetTeamMember(index)`. Todas las
consultas, carga y predicción están protegidas: si falta un miembro o falla el
módulo, se mantiene la GRU individual anterior. La prioridad de supervivencia y
el predictor conservador de combate no cambian.

Esta ruta se verificó con pruebas Python, ejecución Lua bajo `lupa`, paridad
PyTorch/Lua y mocks de la API. No se abrió ni controló el cliente para desarrollarla,
y todavía no se presenta como validación real dentro de una partida de Dota.
