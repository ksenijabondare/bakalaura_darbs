# Anomaliju noteikšana ūdens patēriņa datos
Anomāliju detektēšanas prototipa izstrāde un pārbaude ūdens patēriņa datiem.

## Projekta struktūra
**Svarīgi:** Projektā izmantotie dati ir konfidenciāli. Mapēs `data/raw`, `data/preprocessed` un `data/labeled` esošie dati ir noņemti atbilstoši datu aizsardzības prasībām.
data/raw - Neapstrādāti ūdens patēriņa dati (CSV)\n
data/preprocessed - Dienas agregētas datu kopas\n
data/labeled - Datu kopas ar anomāliju iezīmēm (sintētiskas noplūdes)\n\n

src/anomaly_detection_algorythms/pot.py - Peaks-Over-Threshold algoritms\n
src/anomaly_detection_algorythms/dspot.py - DSPOT algoritms\n
src/prepare_data.py - Datu ielāde\n
src/labeling_dataset.py - Anomāliju atzīmēšana\n
src/visualizations.py - vizualizācijas rīki\n\n

tests - testu un testu novērtēšanas faili\n
test_outputs - algoritma izvaddati: iezīmētas datu kopas ar anomālijam, novērtēšanas apkopojums, vizualizācijas\n


## Nepieciešamas bibliotēkas
Python 3.10+
numpy
pandas
matplotlib
scipy
