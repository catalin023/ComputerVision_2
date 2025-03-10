# Proiect de Detectare și Clasificare a Fețelor

Acest proiect este împărțit în două task-uri principale: detectarea fețelor și clasificarea acestora pe baza unor personaje specifice. Mai jos sunt detaliile fiecărui task.

## Task 1: Detectarea Fețelor

### 1. Extragerea Patch-urilor pentru Fiecare Personaj
- **Conversia imaginii în grayscale**: Transformăm imaginile în tonuri de gri pentru a simplifica procesarea.
- **Augmentarea datelor**: Pentru fiecare patch, extragem patch-uri suplimentare cu mici deviații pe axe și scalare.
- **Flip la patch**: Aplicăm flip pe patch-uri pentru a dubla numărul de exemple.

### 2. Extragerea Patch-urilor Negative
- **Generarea de patch-uri aleatoare**: Generăm patch-uri din imagini care nu intersectează fața cu mai mult de 15%.
- **Patch-uri din interiorul feței**: Extragem patch-uri din interiorul feței care conțin mici părți ale acesteia.

### 3. Calculul Descriptorilor HOG
- **Dimensiunea celulei**: 8px.
- **4 celule per block**: Pentru diferite dimensiuni ale fețelor:
  - Dad: (64, 96)
  - Deedee: (128, 64)
  - Dexter și Mom: (64, 64)

### 4. Antrenarea SVM-urilor
- Antrenăm 3 SVM-uri pentru fiecare tip de față cu diferiți parametri și salvăm modelul cu cea mai bună performanță.

### 5. Detectarea Facială
- Folosim un sliding window de raport cu dimensiunea fețelor, scalându-le pentru a detecta fețe de dimensiuni variate.

### 6. Non-Maximal Suppression
- Pentru detectiile găsite, aplicăm non-maximal suppression pentru a rămâne cu detectiile cu scorul maxim.

## Task 2: Clasificarea Fețelor

### 1. Antrenarea SVM-urilor One vs All
- Antrenăm 4 SVM-uri one vs all, fiecare pentru un personaj specific.
- Exemplele negative sunt celelalte personaje.

### 2. Utilizarea Descriptorilor HOG
- Folosim descriptorii HOG pentru antrenarea SVM-urilor.

### 3. Clasificarea
- Fiecare detectie facială este transmisă ca input modelelor și clasificată pe baza scorului maxim.

## Concluzie
Acest proiect oferă o soluție completă pentru detectarea și clasificarea fețelor, folosind tehnici avansate de preprocesare a imaginilor și învățare automată.
