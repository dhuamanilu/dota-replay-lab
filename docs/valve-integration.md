# Integración experimental con bots de Valve

`bots/bot_generic.lua` carga el árbol destilado de `bots/decision_policy.lua`, construye un estado al comienzo de cada minuto y mantiene la acción elegida durante ese minuto. La carga, la predicción, las consultas y las órdenes están protegidas con `pcall`; un fallo se degrada a `unknown` y queda visible en telemetría.

## Traducción de acciones

- `farm`: ataca al creep de línea enemigo con menos vida dentro de 1400 unidades; si no hay uno, avanza hacia el frente de su línea asignada.
- `fight`: ataca al héroe enemigo con menos vida dentro de 1800 unidades.
- `push`: ataca una torre cercana; si no existe, vuelve a farm.
- `unknown`: retrocede al Ancient si hay un enemigo muy cerca; en otro caso farmea.

Las lecturas de contadores usan `pcall`: una función no disponible no detiene todo el script. Los cambios de oro, XP, last hits y kills se calculan entre checkpoints locales.

## Instalación local y captura

La carpeta versionada `bots/` debe copiarse a la instalación activa de Dota:

```text
<SteamLibrary>/steamapps/common/dota 2 beta/game/dota/scripts/vscripts/bots
```

Si ya existe, se debe respaldar antes de reemplazarla. Inicia Dota con `-console -condebug`, crea un lobby con servidor **Local Host** y selecciona **Local Dev Script** para ambos equipos. `-condebug` escribe la consola en `game/dota/console.log`.

Cada línea del adaptador empieza con `DRL_TELEMETRY` y contiene JSON con versión de esquema, tiempo de juego, minuto, evento y, según corresponda, acción, features disponibles/faltantes, fallback, error u orden emitida. Las órdenes repetidas se limitan a una muestra cada cinco segundos para evitar inundar el log.

Resume una captura de forma reproducible con:

```powershell
$env:PYTHONPATH='src'
python -m dota_replay_lab.valve_telemetry `
  'E:\SteamLibrary\steamapps\common\dota 2 beta\game\dota\console.log' `
  --output artifacts/valve/session-summary.json
```

## Diferencia de dominio pendiente

OpenDota y la API de bots no exponen exactamente el mismo estado. El adaptador usa cero para ventajas de equipo porque todavía no hay un cálculo equivalente validado en vivo. También usa cero para `previous_*`: la versión anterior colocaba allí la acción elegida, que no equivale al evento observado usado durante entrenamiento. Esas seis features se enumeran como faltantes en cada decisión. Por tanto, el benchmark offline no mide el desempeño dentro del juego.

La sintaxis Lua y los fallbacks están validados offline con `luaparser` y `lupa`. Aún se debe seleccionar **Local Dev Script** en un lobby y ejecutar partidas controladas; eso no puede sustituirse por una afirmación basada solo en tests Python.
