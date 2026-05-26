# Anomaliju noteikšana ūdens patēriņa datos
Anomāliju detektēšanas prototipa izstrāde un pārbaude ūdens patēriņa datiem.

## Projekta struktūra
**Svarīgi:** Projektā izmantotie dati ir konfidenciāli. Mapēs `data/raw`, `data/preprocessed` un `data/labeled` esošie dati ir noņemti atbilstoši datu aizsardzības prasībām.
data/raw - Neapstrādāti ūdens patēriņa dati (CSV)
data/preprocessed - Dienas agregētas datu kopas
data/labeled - Datu kopas ar anomāliju iezīmēm (sintētiskas noplūdes)

src/anomaly_detection_algorythms/pot.py - Peaks-Over-Threshold algoritms
src/anomaly_detection_algorythms/dspot.py - DSPOT algoritms
src/prepare_data.py - Datu ielāde
src/labeling_dataset.py - Anomāliju atzīmēšana
src/visualizations.py - vizualizācijas rīki

tests - testu un testu novērtēšanas faili
test_outputs - algoritma izvaddati: iezīmētas datu kopas ar anomālijam, novērtēšanas apkopojums, vizualizācijas


## Nepieciešamas bibliotēkas
Python 3.10+
numpy
pandas
matplotlib
scipy
