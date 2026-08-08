# Ruta de aprendizaje

Cada etapa deja un artefacto que se puede inspeccionar antes de avanzar.

## Etapa 1 — Leer una partida

**Pregunta:** ¿qué hechos contiene un replay de Dota 2?

Construiremos un comando que consulte una partida pública y produzca un resumen: jugadores, héroes, resultado, duración y eventos relevantes. Aprenderemos la diferencia entre datos de resultado y estado por instante.

**Salida:** un resumen reproducible de una partida.

## Etapa 2 — Construir estados y decisiones

**Pregunta:** ¿qué datos necesita una política para decidir?

Convertiremos los eventos en vectores por héroe: vida, maná, oro, nivel, posición, aliados y enemigos visibles. La primera etiqueta será una decisión de alto nivel: pelear, retirarse, farmear o empujar.

**Salida:** conjunto de datos versionado con su diccionario de variables.

## Etapa 3 — Primer modelo local

**Pregunta:** ¿puede un modelo pequeño anticipar una decisión humana?

Entrenaremos un clasificador simple en la GPU local. Esta etapa no intenta controlar el juego: verifica que los datos, las etiquetas y la evaluación son correctos.

**Salida:** pesos, métricas y ejemplos de aciertos y errores.

## Etapa 4 — Política de aprendizaje por refuerzo

**Pregunta:** ¿una política mejora al optimizar una recompensa explícita?

Usaremos un entorno de prueba rápido para validar PPO y autojuego. El objetivo es aprender el ciclo de RL sin depender aún de la velocidad del motor de Dota.

**Salida:** curvas de aprendizaje y partidas reproducibles.

## Etapa 5 — Dota controlado

**Pregunta:** ¿podemos aplicar una decisión limitada dentro de Dota real?

Con Dota 2 instalado, se creará una integración mínima con la API de bots de Valve. Empezaremos con un héroe y una sola capacidad, por ejemplo retirada o last hit.

**Salida:** demostración local dentro del juego.
