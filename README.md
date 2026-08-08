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

Base inicial creada. El siguiente entregable es el explorador de replays: descarga una partida pública, extrae sus eventos principales y genera un resumen legible.
