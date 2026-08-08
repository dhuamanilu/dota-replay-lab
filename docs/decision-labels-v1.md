# Etiquetas de decisión v1

Este dataset no contiene la intención verdadera del jugador. Es una primera aproximación auditable: cada fila representa a un héroe en un minuto y conserva tanto la etiqueta elegida como todas las señales que compitieron por ella.

## Reglas

La prioridad es `fight > push > farm > unknown`.

- `fight`: el héroe consiguió una kill o tuvo actividad registrada en una teamfight que cruza el minuto (daño, curación, muerte o kill).
- `push`: el `player_slot` del héroe figura como autor de una destrucción de edificio, Roshan o miniboss durante el minuto.
- `farm`: aumentaron sus last hits y no ganó una regla de mayor prioridad.
- `unknown`: ninguna regla tuvo evidencia suficiente.
- `retreat`: forma parte del vocabulario conceptual, pero v1 no la asigna. Sin posiciones, vida, dirección de movimiento o visión, hacerlo sería inventar la etiqueta.

La columna `signals` conserva conflictos como `fight+push+farm`; `label` contiene la decisión tomada por prioridad. `rules_version=v1` permite comparar futuras correcciones sin mezclar definiciones.

## Sesgos conocidos

- Una teamfight de OpenDota es una ventana agregada; no demuestra la intención táctica del jugador. Si la ventana cruza dos minutos, su actividad puede aportar señal a ambos porque no conocemos el segundo exacto de cada acción interna.
- Curar durante esa ventana se interpreta como participación, aunque la curación pueda no haber sido decisiva.
- `push` solo se atribuye cuando OpenDota incluye `player_slot`. Una torre caída por creeps queda sin autor en vez de repartirse entre todo el equipo.
- Subir last hits es una señal estrecha de farm. No captura stacking, zoning, regeneración o desplazamiento hacia una zona de farmeo.
- La prioridad resuelve el conflicto para entrenar un clasificador de una sola clase, pero `signals` debe usarse para medir cuántas filas son ambiguas.

Estas etiquetas sirven para construir y criticar un baseline; no son ground truth ni una evaluación de MMR.
