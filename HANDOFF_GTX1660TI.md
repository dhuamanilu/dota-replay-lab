# Prompt maestro de handoff — PC con GTX 1660 Ti

Pega desde **PROMPT MAESTRO** en una nueva tarea de Codex abierta dentro del clon de este repositorio.

## PROMPT MAESTRO

```text
Actúa como coimplementador autónomo principal de Dota Replay Lab. No adoptes un rol de profesor y no te detengas en explicaciones o planes si puedes ejecutar trabajo verificable. Tu objetivo es producir la mejor versión funcional y medible del bot que permita esta PC con GTX 1660 Ti, continuando hasta encontrar un bloqueo genuino que requiera Dota 2 instalado, una acción humana dentro del juego, credenciales o hardware externo.

Repositorio canónico:
https://github.com/dhuamanilu/dota-replay-lab

Reglas de trabajo:

1. Inspecciona el repositorio y el estado real de la PC antes de modificar nada.
2. Conserva cambios ajenos y no inventes resultados. Distingue probado offline, probado dentro de Dota y pendiente.
3. Usa commits pequeños y regulares con Conventional Commits: feat:, fix:, test:, docs:, refactor:, perf:, chore:.
4. Antes de cada push ejecuta las pruebas relevantes. Al terminar, push, PR y merge preservando los commits; no hagas squash.
5. Usa CPU y GPU de forma intensiva cuando aporte evidencia, pero mide primero la GPU con nvidia-smi y una ejecución XGBoost CUDA. No supongas que CUDA funciona por el nombre de la tarjeta.
6. No atribuyas MMR, nivel de juego o win rate a métricas offline.
7. Avanza autónomamente. Solo detente si agotaste alternativas seguras y el siguiente paso exige interacción humana o un recurso realmente ausente.

Estado heredado confirmado:

- Corpus: 500 partidas profesionales de OpenDota, 210,150 filas héroe/minuto.
- Modelo elegido: xgboost_d6_w050.
- Validación cruzada agrupada por match_id: macro-F1 0.4567 ± 0.0017.
- Predicciones OOF calibradas: macro-F1 0.4573.
- Test congelado: macro-F1 0.4488, balanced accuracy 0.4531, accuracy 0.6003.
- F1 en test: farm 0.7032, fight 0.4783, push 0.1152, unknown 0.4983.
- Política Lua portable: teacher_calibrated_d12, 1,699 nodos, profundidad 12, fidelidad al teacher 0.9133, macro-F1 0.4420.
- La suite heredada tiene 30 pruebas y CI en Python 3.9 y 3.12.
- El adaptador Valve todavía no se ha ejecutado dentro de Dota 2.

Archivos clave:

- bots/bot_generic.lua: runtime experimental para la API de bots de Valve.
- bots/decision_policy.lua: política portable autocontenida y versionada.
- bots/decision_policy.metrics.json: métricas de la destilación.
- docs/benchmark-v2.md: benchmark y protocolo completos.
- docs/valve-integration.md: acciones, degradaciones y brecha de dominio.
- src/dota_replay_lab/collect_matches.py: adquisición paginada con reintentos.
- src/dota_replay_lab/build_dataset.py: dataset héroe/minuto.
- src/dota_replay_lab/train_policy.py: selección y evaluación agrupada.
- src/dota_replay_lab/export_lua_policy.py: destilación a Lua.
- src/dota_replay_lab/benchmark_report.py: regeneración del benchmark.

Arranque obligatorio en la nueva PC (PowerShell):

git clone https://github.com/dhuamanilu/dota-replay-lab.git
cd dota-replay-lab
python --version
nvidia-smi
python -m pip install -e ".[ml,dev]"
$env:PYTHONPATH='src'
python -m pytest -q
python -c "import xgboost as xgb; print(xgb.__version__)"

Verifica con una prueba corta que XGBoost puede entrenar con device='cuda'. Si falla, diagnostica driver, versión del paquete, memoria y compatibilidad antes de caer a CPU. Registra versiones, GPU detectada, VRAM y tiempo medido.

Los artefactos de entrenamiento bajo artifacts/ están ignorados por Git. La política Lua necesaria para ejecutar el bot sí está versionada. Si se desea reproducir exactamente el entrenamiento sin volver a descargar, copia artifacts/ desde la PC anterior por un medio separado. Si no existe, regenera así:

$env:PYTHONPATH='src'
python -m dota_replay_lab.collect_matches --count 500
python -m dota_replay_lab.build_dataset --manifest artifacts/corpora/pro-matches-v1.json
python -m dota_replay_lab.train_policy --device auto --cv-folds 5
python -m dota_replay_lab.export_lua_policy
python -m dota_replay_lab.benchmark_report
python -m pytest -q

Prioridad 1 — cerrar la integración real con Dota:

1. Localiza todas las bibliotecas de Steam y confirma appmanifest_570.acf, dota2.exe y la carpeta efectiva de scripts de bots. No asumas una ruta fija.
2. Copia el contenido de bots/ a la ubicación que Dota 2 use para Local Dev Script, conservando una copia del original si reemplazas archivos existentes.
3. Crea un lobby local controlado, selecciona Local Dev Script y ejecuta una partida reproducible.
4. Inspecciona consola y logs. Corrige nombres o firmas reales de la API Valve, errores Lua y rutas de carga.
5. Confirma con evidencia que decision_policy.lua carga, Think() se ejecuta y las cuatro acciones tienen degradaciones seguras.
6. Añade telemetría local estructurada sin datos sensibles: timestamps, acción seleccionada, features disponibles, fallback, errores y órdenes emitidas.
7. Ejecuta varias partidas controladas contra bots estándar o un escenario espejo. Mide al menos: arranques sin error, minutos sin crash, tiempo inactivo, distribución de acciones, fallbacks, last hits, muertes, daño a torres y resultado. Conserva configuración y semillas cuando existan.
8. Convierte cada fallo reproducible en una prueba offline cuando sea posible.

Prioridad 2 — mejorar con evidencia:

- Corrige primero la brecha entre features OpenDota y la API en vivo. Hoy las ventajas de equipo se envían como cero y previous_* representa la última acción elegida, no el evento observado usado en entrenamiento.
- Revisa la clase push, cuyo F1 es 0.1152, pero no la optimices aislada si empeora seguridad o conducta en partida.
- Evalúa el árbol Lua contra el teacher y contra trazas reales después de cada cambio.
- Solo después de estabilizar el runtime considera nuevas features, más datos, búsqueda de hiperparámetros, imitation learning secuencial o RL.

Limitaciones que debes conservar explícitas:

- Las etiquetas son heurísticas; no describen la intención real del profesional.
- El estado por minuto de OpenDota no contiene vida, maná, cooldowns, visión ni posición exacta.
- El test congelado mide imitación de etiquetas, no habilidad jugable.
- Una sintaxis Lua válida no prueba compatibilidad con el runtime de Dota.

Criterio de finalización:

Entrega código, pruebas, telemetría y benchmark reproducible. Publica commits convencionales, abre PR, espera CI y fusiona sin squash. Si quedas bloqueado, documenta el comando exacto, error, evidencia revisada y la única intervención necesaria; no marques como verificado lo pendiente.
```

## Resultado mínimo esperado en esa PC

El primer hito no es reentrenar por reentrenar: es ejecutar la política versionada dentro de Dota, descubrir incompatibilidades reales del runtime y producir telemetría de partidas controladas. La GPU se usa después para iteraciones que mejoren una métrica vinculada a esa evidencia.
