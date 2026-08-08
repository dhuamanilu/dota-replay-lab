# Integración experimental con bots de Valve

`bots/bot_generic.lua` carga el árbol destilado de `bots/decision_policy.lua`, construye un estado al comienzo de cada minuto y mantiene la acción elegida durante ese minuto.

## Traducción de acciones

- `farm`: ataca al creep de línea enemigo con menos vida dentro de 1400 unidades.
- `fight`: ataca al héroe enemigo con menos vida dentro de 1800 unidades.
- `push`: ataca una torre cercana; si no existe, vuelve a farm.
- `unknown`: retrocede al Ancient si hay un enemigo muy cerca; en otro caso farmea.

Las lecturas de contadores usan `pcall`: una función no disponible no detiene todo el script. Los cambios de oro, XP, last hits y kills se calculan entre checkpoints locales.

## Diferencia de dominio pendiente

OpenDota y la API de bots no exponen exactamente el mismo estado. El adaptador usa cero para ventajas de equipo porque todavía no hay un cálculo equivalente validado en vivo. `previous_*` representa la acción previa elegida por la política, mientras que durante el entrenamiento provenía de eventos observados. Por tanto, el benchmark offline no mide el desempeño dentro del juego.

La sintaxis Lua está validada estáticamente. Falta copiar la carpeta `bots/` a la ubicación de scripts local, seleccionar **Local Dev Script** en un lobby y ejecutar partidas controladas. Ese paso requiere Dota 2 instalado y no debe sustituirse por una afirmación basada solo en tests Python.
