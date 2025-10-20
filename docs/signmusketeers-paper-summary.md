# SignMusketeers paper review

## Overview
- El artículo introduce SignMusketeers, un enfoque multi-stream para traducción de
  lengua de señas a inglés que prioriza las regiones relevantes (manos, rostro y
  pose corporal) del video.
- La motivación principal es reducir drásticamente el cómputo del preentrenamiento
  respecto al método SSVP-SLT, manteniendo un rendimiento competitivo en How2Sign.

## Pipeline propuesto
- El flujo consta de dos fases: (1) preentrenamiento auto-supervisado de dos
  extractores DINOv2-Small especializados en rostro y manos; (2) traducción
  supervisada con un codificador adicional de pose y un decodificador T5.1.1-Base.
- Cada frame se procesa con detectores para recortar manos y rostro, además de un
  estimador MediaPipe para la pose del torso. Los rasgos de los tres canales se
  concatenan y proyectan antes de alimentar al modelo T5.

## Datos y preparación
- Se utilizan los conjuntos YouTube-ASL (∼700 horas, 600k clips) y How2Sign
  (31,128/1,741/2,322 clips para train/val/test).
- Para el preentrenamiento se muestrean 1M recortes de manos y 1M de rostro desde
  YouTube-ASL; la pose se resume en 14 coordenadas MediaPipe normalizadas.
- En la fase supervisada se emplea stride 2 por clip y se respeta la división
  estándar de How2Sign para evaluación.

## Entrenamiento y cómputo
- El preentrenamiento se ejecuta en 8 GPUs A6000 Ada con lotes de 128 por GPU,
  tasa base 2×10⁻⁴, 5 pseudo-épocas y ajuste de KoLeo ϵ=10⁻⁴ para estabilidad.
- La fase supervisada usa T5.1.1-Base, lotes de 128 distribuidos en 8 GPUs y un
  fine-tuning adicional de 5k pasos en How2Sign tras entrenar en YouTube-ASL.
- Las tablas 7 y 8 recopilan hiperparámetros exhaustivos para reproducir ambas
  etapas.

## Resultados principales
- Frente a Uthus et al. (2023), SignMusketeers mejora BLEU en todos los
  escenarios, logrando +1.9 BLEU en el plan YT→H2S y +1.2 BLEU en H2S-only.
- Comparado con SSVP-SLT, queda 0.4 BLEU por debajo en YT→H2S, pero usa menos
  del 3 % del cómputo y órdenes de magnitud menos cuadros (1.2M vs. ≥42M).
- En H2S-only la brecha con SSVP-SLT es de ~9 BLEU, resaltando la necesidad de
  datos adicionales para métodos multi-stream.

## Eficiencia y análisis
- La Figura 2 ilustra que el método alcanza BLEU 14.1 con unas 600 GPU-horas
  totales, frente a >18k GPU-horas de SSVP-SLT para resultados similares.
- Los autores argumentan que las características multi-stream requieren más
  supervisión para aprender relaciones temporales que ya aporta el preentrenamiento
  masivo de SSVP-SLT.
- Experimentos de ablation muestran que los tres canales (manos, rostro, pose)
  contribuyen y que la pose aporta robustez en escenarios con datos limitados.

## Consideraciones de privacidad y ética
- Para proteger la identidad se difuminan zonas del rostro salvo ojos y boca,
  conservando rasgos lingüísticos críticos sin degradar la calidad.
- El trabajo destaca que reducir los requisitos de cómputo facilita que equipos
  con recursos modestos participen en investigación de accesibilidad para la
  comunidad Sorda.

## Limitaciones y trabajo futuro
- La calidad depende de detectores externos; fallos de MediaPipe o de detección
  de manos/rostro pueden degradar la traducción.
- No se reportan evaluaciones humanas ni pruebas fuera de ASL/How2Sign, lo que
  limita la generalización a otras lenguas de señas.
- Se sugiere explorar modelos multimodales adicionales (p. ej., CLIP) si hubiera
  cómputo disponible para cerrar la brecha restante con SSVP-SLT.

## Recursos complementarios
- El apéndice provee los índices de MediaPipe para reproducir el vector de pose y
  servir como referencia anatómica.
- También se listan configuraciones detalladas de entrenamiento (Tablas 7 y 8) y
  los hiperparámetros de DINOv2 e iBOT utilizados.
- El artículo incluye enlaces a la web del proyecto y a correos de contacto en la
  primera página para consultas adicionales.
