# Anomaliju noteikšana ūdens patēriņa datos
Anomāliju detektēšanas prototipa izstrāde un pārbaude ūdens patēriņa datiem.

## Projekta struktūra
**Svarīgi:** Projektā izmantotie dati ir konfidenciāli. Mapēs `data/raw`, `data/preprocessed` un `data/labeled` esošie dati ir noņemti atbilstoši datu aizsardzības prasībām.<br><br>
data/raw - Neapstrādāti ūdens patēriņa dati (CSV)<br>
data/preprocessed - Dienas agregētas datu kopas<br>
data/labeled - Datu kopas ar anomāliju iezīmēm (sintētiskas noplūdes)<br>

src/anomaly_detection_algorythms/pot.py - Peaks-Over-Threshold algoritms<br>
src/anomaly_detection_algorythms/dspot.py - DSPOT algoritms<br>
src/prepare_data.py - Datu ielāde<br>
src/labeling_dataset.py - Anomāliju atzīmēšana<br>
src/visualizations.py - vizualizācijas rīki<br><br>

tests - testu un testu novērtēšanas faili<br>
test_outputs - algoritma izvaddati: iezīmētas datu kopas ar anomālijam, novērtēšanas apkopojums, vizualizācijas<br>


## Nepieciešamas bibliotēkas
- Python 3.10+
- numpy
- pandas
- matplotlib
- scipy
