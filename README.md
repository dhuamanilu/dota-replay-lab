# Dota Replay Lab

Laboratorio open source para aprender a construir modelos que toman decisiones en Dota 2 a partir de replays públicos.

El primer objetivo no es crear un bot completo. Es construir una base verificable que responda, con datos reales, una pregunta sencilla:

> En un momento de una partida, ¿un héroe debería pelear, retirarse, farmear o empujar?

## Ruta de aprendizaje

1. **Datos** — descargar y entender una partida pública.
2. **Estado** — representar qué puede conocer un jugador en cada instante.
3. **Decisión** — etiquetar una decisión acotada y entrenar un primer modelo local.
4. **Evaluación** — medir el modelo con partidas que no vio durante el entrenamiento.
5. **Control** — conectar decisiones limitadas a un bot de Dota mediante la API de Valve.

Los detalles de cada etapa están en [docs/learning-path.md](docs/learning-path.md).

## Principios

- Dota primero: el proyecto trabaja con partidas reales desde el comienzo.
- Reproducible: código, configuraciones y resultados versionados.
- Local primero: la RTX 4050 Laptop se usa para los experimentos iniciales.
- Medible: ninguna afirmación de nivel se hace sin una evaluación publicada.
- Abierto: el repositorio, pesos de modelos entrenados y documentación se publicarán bajo licencia MIT.

## Estado actual

Base inicial creada. El primer entregable, el explorador de replays, ya permite descargar una partida pública y generar un resumen legible.

## Primer experimento

Con Python 3.9 o superior y el código fuente en `PYTHONPATH`:

```powershell
$env:PYTHONPATH='src'
python -m dota_replay_lab.fetch_match --latest-pro
```

El comando descarga la partida profesional más reciente que OpenDota pueda entregar, guarda su respuesta sin modificar y crea un resumen Markdown dentro de `artifacts/matches/`. Para analizar una partida concreta:

```powershell
python -m dota_replay_lab.fetch_match 1234567890
```

El identificador anterior es solo un ejemplo; sustitúyelo por un `match_id` real de OpenDota.

Para pausar la partida y comparar los diez héroes en un minuto concreto:

```powershell
python -m dota_replay_lab.inspect_state 8934279386 --minute 12
```

## Primer dataset de decisiones

Para congelar un corpus profesional reproducible y crear una fila CSV por héroe/minuto:

```powershell
$env:PYTHONPATH='src'
python -m dota_replay_lab.collect_matches --count 500
python -m dota_replay_lab.build_dataset --manifest artifacts/corpora/pro-matches-v1.json
```

El resultado predeterminado es `artifacts/datasets/decision-labels-v3.csv`. Conserva las reglas de etiqueta v2 y añade denies acumulados y por minuto al estado observable. Las entradas describen el estado al inicio del minuto y la etiqueta describe el intervalo posterior, evitando filtrar el futuro al modelo. Las etiquetas son heurísticas transparentes (`fight`, `push`, `farm` o `unknown`), no la intención verdadera del jugador. `retreat` queda reservada y no se asigna sin evidencia de posición o movimiento. Consulta [las reglas y sesgos de v2](docs/decision-labels-v2.md).

Para entrenar y evaluar baselines sin mezclar minutos de una misma partida entre conjuntos:

```powershell
python -m pip install -e ".[ml]"
$env:PYTHONPATH='src'
python -m dota_replay_lab.train_policy --device auto
```

El entrenamiento compara mayoría, regresión logística y XGBoost. Selecciona por macro-F1 medio en cinco folds agrupados por partida y evalúa una sola vez sobre 20 % de partidas de test completamente separadas.

Para ejecutar la política en el estado de un héroe y obtener probabilidades para el minuto siguiente:

```powershell
python -m dota_replay_lab.predict_policy 8934279386 --minute 12 --player-slot 0
```

Para destilar el modelo GPU a un árbol Lua autocontenido que pueda incorporarse a un bot de Valve:

```powershell
python -m dota_replay_lab.export_lua_policy
```

La política Lua generada se guarda en `bots/decision_policy.lua`. Su sintaxis puede verificarse sin Dota con `python -c "from luaparser import ast; ast.parse(open('bots/decision_policy.lua', encoding='utf-8').read())"` después de instalar `.[dev]`.

`bots/bot_generic.lua` contiene el primer adaptador experimental para la API de bots. Sus acciones, degradaciones seguras y diferencias pendientes respecto del estado de OpenDota están documentadas en [docs/valve-integration.md](docs/valve-integration.md).

Los resultados completos y el protocolo de evaluación están publicados en [docs/benchmark-v2.md](docs/benchmark-v2.md). Se regeneran con `python -m dota_replay_lab.benchmark_report`.
