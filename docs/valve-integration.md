# Integración experimental con bots de Valve

`bots/bot_generic.lua` carga el árbol destilado de `bots/decision_policy.lua`, construye un estado al comienzo de cada minuto y mantiene la acción elegida durante ese minuto. La carga, la predicción, las consultas y las órdenes están protegidas con `pcall`; un fallo se degrada de forma segura y queda visible en telemetría.

## Conducta del runtime

- `farm`: ataca al creep de línea enemigo con menos vida; si no hay uno, avanza hacia el frente de la línea asignada.
- `fight`: ataca al héroe enemigo con menos vida dentro de 1.800 unidades.
- `push`: ataca una torre cercana y, si no existe, vuelve a farm.
- `unknown`: retrocede al Ancient si hay una amenaza cercana; en otro caso farmea.
- Supervivencia: con 25 % de vida o menos y un enemigo cercano, o con 15 % de vida o menos en cualquier caso, una retirada al Ancient tiene prioridad sobre la política.

Las acciones realmente ejecutadas en el minuto anterior alimentan `previous_fight`, `previous_push` y `previous_farm`. Esto reemplaza los ceros permanentes y aproxima la semántica histórica sin confundir una predicción con un comportamiento observado.

## Telemetría

Cada línea empieza con `DRL_TELEMETRY` y contiene JSON con versión de esquema, tiempo de juego, identidad del bot, evento y sus campos asociados. El esquema 2 añadió oro, nivel, XP necesaria para subir, last hits, kills y muertes. El esquema 3 añadió muestras de acción, tiempo vivo observado y tiempo inactivo acumulado. El esquema 4 añade denies y su cambio por minuto. Las órdenes repetidas se muestrean como máximo una vez cada cinco segundos.

El resumen también cuenta ataques dirigidos a unidades cuyo nombre contiene `tower`. Es un proxy de intención de push, no daño exacto a torres.

```powershell
$env:PYTHONPATH='src'
python -m dota_replay_lab.valve_telemetry `
  'E:\SteamLibrary\steamapps\common\dota 2 beta\game\dota\console.<match_id>.log' `
  --output artifacts/valve/session-summary.json
```

## Validación real realizada

Entorno: Dota 2 build `24621316`, GeForce GTX 1660 Ti de 6 GiB, driver `591.86`, runtime CUDA `13.1`, Python `3.12.10` y XGBoost `3.4.0`. Una prueba CUDA de 250.000 × 32 características y 120 árboles tardó 1,206 segundos sin advertencias.

| Partida | Cambio validado | Evidencia |
| --- | --- | --- |
| `8935368914` | Carga inicial | 9 políticas cargadas, 27 decisiones, 0 errores; reveló que los bots no emitían órdenes y que `GetXP`/`GetKills` no correspondían al runtime usado. |
| `8935385035` | Movimiento y órdenes | 755 órdenes durante 243,734 s observados: 319 ataques y 436 movimientos; 0 errores de telemetría. Los bots salieron de la fuente y alcanzaron sus líneas. |
| `8935400352` | Identidad y contadores | 9 políticas, 36 decisiones y 1.026 órdenes durante 312,767 s; 0 errores. Máximos observados: 11 last hits, nivel 4, 1.400 de oro y 2 muertes por bot. |

En la tercera sesión hubo 29 decisiones `farm` y 7 `unknown`; los nueve bots registraron entre 1 y 11 last hits. La partida se interrumpió deliberadamente tras el minuto 3, por lo que no existe resultado final ni win rate. Tampoco se instrumentó daño exacto a torres. El tiempo inactivo del esquema 3 está cubierto por pruebas offline, pero la sesión destinada a validarlo dentro del juego fue cancelada y no se presenta como evidencia real.

## Diferencia de dominio pendiente

`GetCurrentXP` pertenece a la API de entidades del servidor, no a `CDOTA_Bot_Script`; por eso `experience` y `experience_change` permanecen explícitamente ausentes. Se registran nivel y XP necesaria para el siguiente nivel, pero no se inventa XP acumulada.

Las ventajas de oro y experiencia por equipo siguen en cero porque no se ha validado una lectura equivalente para ambos equipos desde la API del bot. `previous_*` ya procede de órdenes observadas, aunque sigue siendo una aproximación a las señales agregadas de OpenDota. El benchmark offline mide imitación de etiquetas heurísticas, no habilidad, MMR ni probabilidad de victoria.

La ejecución dentro de Dota ya confirmó carga, `Think()`, movimiento, ataque, degradaciones y contadores con la política de árbol anterior. La GRU portable posterior tiene paridad exacta con PyTorch y ejecuta el adaptador bajo `lupa`, pero no se presenta como validada dentro del cliente. Las mejoras posteriores se desarrollan con replays y pruebas automatizadas; no requieren controlar manualmente Dota.

## Predictor de combate por replay

El adaptador carga también `replay_combat_policy.lua`. El modelo reducido usa
solo señales que la API puede leer y anticipa combate a cinco segundos. Para
limitar falsos positivos, únicamente abre una pelea oportunista cuando
`engage_probability >= 0,90` y ya existe un enemigo a 900 unidades. La retirada
por poca vida conserva prioridad absoluta; un fallo del predictor desactiva esta
ruta sin afectar la política principal.

La mejora offline y la paridad exacta sklearn/Lua están documentadas en
`replay-combat-policy.md`. Todavía no existe evidencia de win rate para esta
integración y no se presenta como validada dentro del cliente.

## Coordinador de equipo por self-play

El adaptador intenta cargar `team_selfplay_policy.lua` antes de recurrir a la GRU
individual. Al comienzo de cada minuto consulta `GetTeamMember(1..5)`, construye
cinco estados y obtiene un único plan coordinado. Cada bot usa la posición que le
corresponde dentro de ese plan. La misma semilla determinista por minuto y equipo
mantiene la selección alineada entre runtimes; incluso un bot muerto avanza el
estado recurrente antes de omitir órdenes.

Las lecturas de oro, last hits y kills están protegidas con `pcall`. Experiencia y
ventajas de equipo siguen en cero porque aún no existe una lectura validada que
reproduzca su semántica histórica. Si falta un miembro, no se reconoce al bot
propio, falla una consulta crítica o la predicción devuelve una acción inválida,
el adaptador vuelve a la política individual. Retirada por supervivencia y consejo
conservador de combate conservan prioridad sobre el plan aprendido.

La política se entrenó y auditó sin abrir el cliente. La integración está cubierta
por un runtime Lua simulado y el módulo exportado tiene paridad PyTorch/Lua, pero
aún no se afirma que el coordinador haya sido validado en una partida real. Véase
`self-play.md` para resultados, límites y reproducción.
