# Detectarea şi recunoaşterea facială a personajelor din serialul de desene animate Laboratorul lui Dexter
Task1:
1. Extragem patch-uri corespunzatoare pentru fiecare personaj
• Convertim imaginea in grayscale
• Augmentarea datelor: pentru fiecare patch, mai luam si patchuri
cu mici deviatii pe axe si scalare
• Facem flip la patch pentru dublarea exempleror
2. Extragem patchuri negative
• Generam patchuri random din imagini, dar care sa nu
intersecteze fata cu mai mult de 15%
• Extragem deasemenea, pathcuri din interiorul fetei care cuprinde
mici parti ale ei
3. Calculam descriptorii HOG cu dimenisunea celulei 8px si 4 celule per
block pentru diferite dimensiuni ale fetelor
• Dad - (64, 96)
• Deedee - (128, 64)
• Dexter si Mom - (64, 64)
4. Antrenam 3 SVM-uri pentru fiecare tip de fata cu diferiti parametri si
salvam pe aceea cu performanta cea mai buna
5. Pentru detectarea faciala, folosim un sliding window de ratio cu
dimensiunea fetelor scalandu-le pe pentru a detecta fete de dimensiuni
variate
6. Pentru detectiile gasite folosim non maximal supresion pentru a
ramane cu detectiile cu scorul maximTask2:
1. Antrenam 4 SVM one vs all, fiecare SVM va fi antrenat pentru un
personaj anumit, iar ca exemple negative vor fi restul personajelor
2. Folosim deasemena descriptorii HOG pentru antrenarea SVM-urilor
3. Pentru clasificare, fiecare detectie faciala va fi transmisa ca input
modelelor si clasificata dupa scorul maxim
