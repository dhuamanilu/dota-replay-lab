# Trayectorias de replays crudos

Los JSON de OpenDota sirven para el estado por minuto, pero no contienen cada
orden del jugador. Esta etapa usa el `replay_url` ya guardado en cada match,
descarga el demo de Valve y lo procesa localmente. No inicia Dota ni envía
entradas al cliente.

## Parser reproducible

La validación inicial usó `odota/parser` en el commit
`02b78c6ca0010b5ed64a800c23212040c28d35ce`, compilado con Java 21. El servicio
local escucha de forma predeterminada en `http://localhost:5600`. El endpoint
raíz recibe el `.dem` mediante POST y devuelve JSONL; `/blob` no es equivalente,
pues devuelve un resumen agregado.

Valve puede entregar BZip2 o Zstandard aunque la URL termine en `.bz2`. El
descargador identifica el formato por bytes mágicos, descarga de forma atómica,
valida la cabecera `PBDEMS2` y nunca confía en la extensión.

```powershell
python -m pip install -e ".[replay]"
python -m dota_replay_lab.parse_replay_corpus `
  --manifest artifacts/corpora/pro-matches-v1.json `
  --count 10
```

Cada `*.seconds.csv` conserva una fila por héroe y segundo con posición, equipo,
vida, nivel, economía acumulada, movimiento desde el segundo anterior y conteos
de órdenes `move`, `attack`, `cast` y `hold`. El split de cualquier experimento
posterior debe hacerse por `match_id`, nunca por filas.

El extractor también normaliza los nombres de unidades del combat log y conserva
`hero_damage_dealt` y `hero_damage_received` por segundo. En una integración real
de 20.570 filas encontró 1.183 segundos con daño de héroe infligido y 2.922 con
daño recibido. Esto permite distinguir combate contra héroes sin asumir que toda
orden `attack` era una pelea.

Los `.dem`, comprimidos y eventos JSONL son temporales. Después de validar que
la trayectoria contiene intervalos, el comando elimina únicamente esos archivos
generados y conserva el CSV compacto. `--keep-events` permite retener el JSONL
para investigar nuevos campos.

## Primera comprobación real

Cinco partidas profesionales produjeron 129.040 filas, diez slots distintos por
partida y cobertura continua desde el segundo cero hasta el final. El 83,03 % de
las filas contenía al menos una orden. Esta comprobación valida la extracción;
no demuestra todavía una mejora de win rate ni autoriza reemplazar la política.
