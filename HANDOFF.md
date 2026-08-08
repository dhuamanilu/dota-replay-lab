# Handoff — Dota Replay Lab

## Rol esperado

Actúa como profesor y coimplementador. El usuario quiere aprender mientras se construye: antes de cada cambio explica, de forma breve y en español, la idea de alto nivel, qué se va a medir y qué limita el resultado. Luego implementa, ejecuta pruebas y publica los cambios al repositorio. No entregues solo código ni saltes directo a una propuesta abstracta.

## Objetivo del proyecto

Construir un proyecto open source que enseñe y experimente con decisiones de IA en **Dota 2**. La inspiración es OpenAI Five, pero el objetivo inicial no es prometer un bot de nivel profesional. El camino debe ser medible y local:

1. Extraer datos reales de partidas de Dota.
2. Convertirlos en estados y decisiones que un modelo pueda aprender.
3. Entrenar modelos pequeños en la GPU local.
4. Evaluarlos con datos no vistos.
5. Solo después conectar una política limitada a Dota mediante la API de bots de Valve.

El usuario prefiere empezar en Dota desde el primer día. Un mini-MOBA solo se aceptaría más adelante como banco de pruebas de RL, no como producto principal.

## Repositorio y entorno

- Repositorio público: https://github.com/dhuamanilu/dota-replay-lab
- Checkout local: `C:\Users\Usuario\Documents\Codex\2026-08-07\realtime-voice-chat\outputs\dota-replay-lab`
- Rama: `main`
- GitHub autenticado como `dhuamanilu`; Git configurado como Diego Alonso Huamani Luque.
- GPU verificada: NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB VRAM.
- Python disponible: 3.9. Por compatibilidad, el proyecto declara Python >= 3.9.
- Steam está instalado, pero Dota 2 no se encontró instalado en las bibliotecas actuales. No bloquea el análisis con OpenDota; sí bloqueará el control dentro del juego y el parseo local de replays completos.

## Lo que ya está construido y verificado

### 1. Explorador de partidas públicas

El comando usa OpenDota:

```powershell
$env:PYTHONPATH='src'
python -m dota_replay_lab.fetch_match --latest-pro
```

También acepta un `match_id`:

```powershell
python -m dota_replay_lab.fetch_match 8934279386
```

Produce, dentro de `artifacts/matches/`:

- `<match_id>.json`: respuesta cruda de OpenDota.
- `<match_id>.md`: informe Markdown con resultado, héroes, jugadores, K/D/A, GPM, XPM, línea de tiempo y eventos.
- `<match_id>.advantages.svg`: gráfico de ventaja de oro y experiencia por minuto.

### 2. Traducción de héroes

El endpoint de una partida entrega `hero_id` numérico, no siempre el nombre. El proyecto consulta `constants/heroes` y une ambos datos. Esta corrección fue necesaria porque el primer informe mostraba `unknown hero`.

### 3. Línea de tiempo de equipo

Se usan `radiant_gold_adv`, `radiant_xp_adv` y `objectives` de OpenDota. Los valores positivos favorecen a Radiant; el informe lo explica de forma explícita.

### 4. Estado por héroe a nivel de minuto

Comando:

```powershell
$env:PYTHONPATH='src'
python -m dota_replay_lab.inspect_state 8934279386 --minute 12
```

Produce `<match_id>.minute-<n>.md` con cada héroe, equipo, oro, XP, last hits, cambios del último minuto, ventaja de equipo desde su perspectiva y kills en el último minuto.

Ejemplo ya generado:

`artifacts/matches/8934279386.minute-12.md`

### 5. Pruebas

Ejecutar desde la raíz del repo:

```powershell
$env:PYTHONPATH='src'
python -m pytest -q
```

La última verificación pasó con 5 pruebas.

## Archivos principales

- `src/dota_replay_lab/opendota.py`: cliente HTTP sin dependencias de OpenDota.
- `src/dota_replay_lab/fetch_match.py`: descarga e informe de una partida.
- `src/dota_replay_lab/summary.py`: resumen final de partida.
- `src/dota_replay_lab/timeline.py`: línea de tiempo y SVG de ventajas.
- `src/dota_replay_lab/hero_state.py`: representación de estado por héroe y minuto.
- `src/dota_replay_lab/inspect_state.py`: CLI para generar una tabla de estados.
- `tests/`: pruebas unitarias actuales.
- `docs/learning-path.md`: ruta pedagógica original.

## Decisiones importantes y límites actuales

- No afirmar que el sistema ve el mismo estado que un jugador. Los datos actuales son agregados por minuto y no contienen vida, maná, cooldowns, visión ni posición exacta.
- No afirmar equivalencia de MMR/medalla todavía.
- Los bots de Workshop chinos como Open Hyper AI son scripts de reglas, no RL entrenado. No son una base para un modelo de aprendizaje; sirven como referencia de integración con Dota.
- OpenAI Five no liberó pesos ni código reproducible. Sí publicó papers, arquitectura, PPO, autojuego, reward shaping y la interfaz de alto nivel.
- La RTX 4050 de 6 GB puede entrenar modelos pequeños o demos de RL locales. No es razonable prometer un OpenAI Five completo ni nivel Ancient antes de tener un benchmark real.
- El modo de voz de ChatGPT puede ocultar la selección exacta de modelo/esfuerzo; no asumir el modelo solo por la conversación.

## Siguiente bloque recomendado: etiquetas de decisión

La próxima pieza debe convertir estados por minuto en un dataset inicial de acciones de alto nivel. Mantenerlo transparente y pequeño:

1. Definir etiquetas provisionales: `farm`, `fight`, `push`, `retreat`.
2. Empezar con reglas heurísticas documentadas, no fingir que son verdad absoluta.
   - `fight`: kill propia o participación en teamfight reciente.
   - `push`: destrucción de edificio u objetivo cercano.
   - `farm`: aumento de last hits/oro sin fight ni push.
   - `retreat`: solo si el dato disponible permite una señal defendible; si no, dejar la etiqueta como `unknown` en vez de inventarla.
3. Construir un comando que procese varias partidas y emita CSV o JSONL versionado con una fila por héroe/minuto.
4. Añadir pruebas para conflictos de etiquetas y una explicación pedagógica del sesgo de estas reglas.
5. Solo después preparar el primer clasificador supervisado. No pasar a PPO antes de tener datos, etiquetas y un baseline medible.

## Estilo de colaboración

- Responder en español claro y directo.
- Dar al usuario enlaces a los artefactos generados.
- Antes de un hito, explicar el concepto con una analogía corta si ayuda.
- Hacer cambios reales, probarlos y subirlos al repositorio en commits convencionales separados.
- Preservar la rama `main` limpia y no exponer secretos.
