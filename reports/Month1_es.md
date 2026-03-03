# Informe de Progreso — Mes 1

## Proyecto: Modeling News-Driven Liquidity Dynamics and Information Propagation in Commodities Markets

**Autor:** Enrique Favila Martínez  
**Institución:** Radboud University — Maestría en Inteligencia Artificial  
**Empresa anfitriona:** Hammer Market Intelligence  
**Supervisora académica:** Dra. Lejla Batina  
**Periodo reportado:** Mes 1 de 6

---

## 1. Introducción

El presente informe documenta los avances, decisiones metodológicas y hallazgos preliminares correspondientes al primer mes de desarrollo del proyecto de tesis. Durante este periodo se llevaron a cabo actividades de exploración de datos, evaluación de herramientas de procesamiento de lenguaje natural, selección del commodity de estudio y construcción de un pipeline inicial de adquisición y análisis de datos. Las decisiones tomadas en esta fase condicionan el diseño metodológico del resto del proyecto y se documentan aquí con el fin de garantizar la trazabilidad y reproducibilidad de la investigación.

---

## 2. Exploración de Modelos de Análisis de Sentimiento

### 2.1 Modelos evaluados

Una de las primeras tareas del proyecto consistió en evaluar modelos de lenguaje de gran escala (Large Language Models, LLMs) con capacidad para clasificar el sentimiento de noticias financieras. El objetivo fue determinar qué herramienta resulta más adecuada para extraer señales de sentimiento a partir de texto no estructurado en el contexto de mercados de commodities.

Se evaluaron los siguientes modelos:

- **FinBERT** (Araci, 2019): modelo basado en BERT y ajustado fino (fine-tuned) sobre corpus financieros. Clasifica el sentimiento en tres categorías: positivo, neutral y negativo. Su entrenamiento incluye datos del periodo COVID-19, lo que amplía su cobertura de eventos de alta volatilidad informativa.

- **SEC-BERT-finetuned-finance-classification**: variante ajustada sobre documentos financieros regulatorios, con clasificación en las categorías bullish, neutral y bearish. Su enfoque en documentos formales lo distingue de FinBERT en términos de dominio de entrenamiento.

- **Modelos de propósito general**: se incluyeron en la evaluación modelos conversacionales de gran escala como ChatGPT (OpenAI) y DeepSeek, aplicados directamente sobre las mismas noticias mediante instrucciones de clasificación de sentimiento.

### 2.2 Metodología de evaluación

Las noticias utilizadas para la evaluación correspondieron a temas de agricultura y aranceles comerciales internacionales, en el contexto del commodity inicialmente seleccionado (azúcar, véase Sección 3). Las noticias provenían de fuentes diversas con distintos niveles de objetividad y orientación editorial, lo que permite caracterizar el comportamiento de los modelos ante texto con sesgos potenciales.

### 2.3 Resultados y observaciones

Los resultados obtenidos fueron heterogéneos entre modelos. Cada herramienta produjo clasificaciones distintas ante las mismas noticias, con algunos modelos mostrando tendencia hacia categorías neutras y otros con clasificaciones más polarizadas. Esta dispersión constituye una observación metodológica relevante: la elección del modelo de sentimiento no es una decisión técnica secundaria, sino que afecta directamente las etiquetas con las que se alimentan los modelos posteriores y, por tanto, las conclusiones del análisis. En ausencia de un estándar consolidado para la clasificación de sentimiento en noticias de commodities, la selección y validación del modelo de NLP representa una contribución metodológica propia de este proyecto.

Esta observación está respaldada por la literatura, que señala que la extracción de sentimiento a partir de texto financiero varía significativamente en función del modelo y del dominio de entrenamiento (Araci, 2019; Malo et al., 2014).

---

## 3. Selección del Commodity: del Azúcar al Petróleo

### 3.1 Commodity inicial: azúcar

El plan original del proyecto contemplaba el azúcar como commodity de estudio. Con este fin, se realizó una búsqueda de fuentes de datos en plataformas especializadas, incluyendo TradingMap, UN Comtrade y FAOSTAT. Si bien estas fuentes proveen datos de producción, comercio internacional y precios, la exploración reveló limitaciones estructurales que comprometen la viabilidad del proyecto con este commodity.

Las principales limitaciones identificadas fueron:

- **Baja variabilidad temporal:** los precios del azúcar presentan cambios poco frecuentes y de baja amplitud en comparación con commodities energéticos.
- **Estacionalidad predecible:** las fluctuaciones de precio están fuertemente determinadas por ciclos de cosecha, lo que reduce el componente de sorpresa informativa que es central para este proyecto.
- **Escasez de eventos exógenos de alto impacto:** a diferencia de los commodities energéticos, el azúcar rara vez experimenta shocks abruptos derivados de noticias geopolíticas o decisiones de política internacional, que son precisamente el tipo de eventos que el proyecto busca modelar.

Estas características hacen del azúcar un commodity con escaso material para estudiar la propagación de información a través de noticias.

### 3.2 Commodity seleccionado: petróleo crudo WTI

Sobre la base de los criterios anteriores, se tomó la decisión de pivotar hacia el petróleo crudo West Texas Intermediate (WTI), negociado en el NYMEX bajo el ticker CL. Esta decisión se fundamenta en las siguientes razones:

- **Alta sensibilidad a eventos externos:** el precio del WTI reacciona de forma inmediata y mensurable a eventos geopolíticos — conflictos armados, sanciones internacionales, tensiones entre países productores —, así como a decisiones de la OPEP y a publicaciones de datos macroeconómicos como los inventarios semanales de la EIA. El concepto de riesgo geopolítico (geopolitical risk, GPR) está bien documentado en la literatura de mercados energéticos y constituye uno de los principales canales mediante los cuales las noticias de texto se traducen en movimientos de precio y liquidez.
- **Abundancia de literatura académica:** existe una amplia base de estudios previos sobre la relación entre noticias y precios del petróleo, lo que permite contextualizar y comparar resultados.
- **Disponibilidad de datos de alta frecuencia:** a diferencia del azúcar, el WTI cuenta con datos intradía accesibles a través de múltiples fuentes públicas.
- **Flexibilidad para extensiones:** la estructura del mercado energético permite, en una fase posterior, extender el análisis a commodities relacionados como Brent, gas natural o gasolina, en el marco del estudio de spillovers (RQ3).

---

## 4. Exploración y Selección de Fuentes de Datos

### 4.1 OilPriceAPI

Se evaluó OilPriceAPI como fuente principal de datos de precio intradía para WTI. La plataforma ofrece precios en tiempo real, datos históricos, curvas de futuros e inventarios. Sin embargo, la exploración técnica reveló una limitación estructural en el plan gratuito: un techo artificial de 100 registros por solicitud y 500 registros en datos históricos horarios para un periodo de 30 días. Esta restricción hace inviable la construcción de un dataset de entrenamiento de tamaño suficiente. En consecuencia, OilPriceAPI fue descartada como fuente principal de datos de precio.

### 4.2 yfinance — fuente adoptada

Como alternativa, se adoptó la librería yfinance (Python), que permite acceder a datos históricos del contrato de futuros WTI (ticker CL=F) con resolución horaria y cobertura de aproximadamente dos años. Los datos incluyen las columnas Open, High, Low, Close y Volume (OHLCV), lo que resuelve directamente el problema de la variable de liquidez. El dataset obtenido comprende **11,205 registros horarios** sin restricciones de volumen ni costos asociados.

### 4.3 EIA — fuente de eventos estructurados

Para la representación de noticias, se adoptó el Weekly Petroleum Status Report de la EIA, disponible de forma gratuita a través de su API pública. Este reporte, publicado cada miércoles a las 10:30 AM ET, informa el nivel de inventarios comerciales de petróleo crudo en los Estados Unidos. Se descargaron **321 observaciones semanales** con cobertura 2020–2026. Cada publicación constituye un evento de noticia estructurado con timestamp exacto, lo que permite construir ventanas de evento con precisión horaria.

---

## 5. Construcción de Variables y Pipeline Inicial

### 5.1 Variables de liquidez

Dado que los datos disponibles no incluyen bid-ask spread ni profundidad de order book, la liquidez se aproxima mediante las siguientes métricas derivadas de los datos OHLCV:

- **Volumen horario (log-transformado):** medida directa de actividad de trading. Variable dependiente principal.
- **Rango de precios (High − Low):** proxy de volatilidad intradía, equivalente al estimador de Parkinson (1980).
- **Ratio de Amihud (2002):** cociente entre el retorno absoluto y el volumen, que mide el impacto de precio por unidad de volumen negociada.

### 5.2 Ventanas de evento

Se construyeron ventanas de evento de ±4 horas alrededor de cada reporte EIA, resultando en un dataset de **993 registros distribuidos en 101 eventos**. Cada registro incluye las métricas de liquidez, la dirección del shock de inventarios (bearish si los inventarios suben, bullish si bajan) y la hora relativa al reporte.

---

## 6. Resultados Preliminares

### 6.1 Análisis descriptivo

El análisis visual de las ventanas de evento revela dos patrones relevantes:

1. El price range alcanza su valor máximo en la hora inmediatamente anterior al reporte (hora −1), lo que sugiere anticipación del mercado antes de la publicación oficial.
2. El volumen de trading es consistentemente mayor en eventos bearish que en eventos bullish a lo largo de toda la ventana de observación.

### 6.2 Baseline estadístico — Asimetría en volumen

Se estimaron tres modelos de regresión lineal con el log-volumen como variable dependiente y la dirección del shock, la magnitud del shock y su interacción como predictores:

| Modelo                    | Variable    | Coeficiente | p-valor |
| ------------------------- | ----------- | ----------- | ------- |
| M1 — Solo dirección       | is_bearish  | 0.210       | 0.030   |
| M2 — Dirección + Magnitud | is_bearish  | 0.217       | 0.030   |
| M2 — Dirección + Magnitud | shock_size  | −0.014      | 0.782   |
| M3 — Con interacción      | is_bearish  | 0.216       | 0.031   |
| M3 — Con interacción      | interacción | 0.029       | 0.770   |

El efecto de la dirección bearish sobre el volumen es estadísticamente significativo (p < 0.05) y robusto a la inclusión de controles por magnitud e interacción. El coeficiente de 0.21 equivale a aproximadamente un **23% más de volumen de trading** en eventos bearish respecto a eventos bullish en la ventana de reacción inmediata (horas 0 a +2). La magnitud del shock y su interacción con la dirección no resultan significativas, lo que sugiere que el efecto asimétrico es independiente del tamaño del cambio en inventarios.

Estos resultados constituyen el **baseline estadístico de la tesis** y representan la primera evidencia preliminar en respuesta a RQ1. El R² bajo (0.024) es esperado en este contexto e indica que factores contextuales adicionales — que serán capturados mediante modelos de NLP en fases posteriores — moderan la respuesta del mercado.

---

## 7. Conclusiones del Mes 1

El primer mes de trabajo permitió establecer las bases empíricas y metodológicas del proyecto. Las decisiones clave tomadas son:

- El commodity de estudio es **WTI Crude Oil**, con datos horarios OHLCV obtenidos de yfinance.
- La variable de liquidez principal es el **volumen horario log-transformado**, complementado con price range y ratio de Amihud.
- La fuente inicial de eventos de noticias son los **reportes semanales EIA**, con timestamps exactos y clasificación de dirección del shock.
- Existe evidencia estadística preliminar de **asimetría en la respuesta de liquidez** ante noticias bearish vs bullish (p = 0.03), lo que valida empíricamente la RQ1 y justifica el modelado con técnicas de IA en fases posteriores.

La fase siguiente se concentrará en incorporar texto libre de noticias como input para modelos de NLP (FinBERT), completar el feature engineering y avanzar hacia el modelado formal de asimetría y estructura temporal.

---

## Referencias

- Amihud, Y. (2002). Illiquidity and stock returns: Cross-section and time-series effects. _Journal of Financial Markets_, 5(1), 31–56.
- Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. _ACL Workshop on Financial Technology and NLP_.
- Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. _Journal of the Association for Information Science and Technology_, 65(4), 782–796.
- Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. _Journal of Business_, 53(1), 61–65.
- Smales, L. A. (2015). Asymmetric volatility response to news in commodity markets. _Journal of International Financial Markets, Institutions and Money_, 34, 130–149.
- Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. _The Journal of Finance_, 62(3), 1139–1168.
