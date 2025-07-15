# Sistema de Análisis de Datos Académicos

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Universidad](https://img.shields.io/badge/universidad-UTP-red.svg)](https://utp.ac.pa)

## 📋 Descripción

El Sistema de Análisis de Datos Académicos es una aplicación Python desarrollada para la Universidad Tecnológica de Panamá que permite realizar análisis integral de datos de rendimiento estudiantil. El sistema proporciona capacidades avanzadas de procesamiento de datos, análisis estadístico, visualización y generación de reportes en formato APA.

## 🚀 Características Principales

- ✅ **Carga y validación de datos** - Soporte para archivos CSV y Excel
- 🧹 **Limpieza y preprocesamiento** - Manejo automático de valores faltantes y estandarización
- 📊 **Análisis estadístico avanzado** - Cálculos descriptivos y inferenciales
- 📈 **Visualizaciones interactivas** - Gráficos con Matplotlib y Seaborn
- 📑 **Reportes profesionales** - Generación automática de reportes PDF estilo APA
- ⚠️ **Análisis de riesgos** - Identificación de estudiantes en riesgo académico
- 🔍 **Análisis demográfico** - Segmentación por diferentes variables
- 🎯 **Prioridades de intervención** - Recomendaciones basadas en datos

## 📁 Estructura del Proyecto

```
programming-iii/
├── main.py                 # Archivo principal del sistema
├── config.py               # Configuración global
├── requirements.txt        # Dependencias del proyecto
├── README.md              # Este archivo
├── ARCHITECTURE.md        # Documentación de arquitectura
├── xAPI-Edu-Data.csv     # Dataset de ejemplo
├── src/
│   ├── data/              # Módulos de datos
│   │   ├── data_loader.py    # Carga de datos
│   │   └── data_cleaner.py   # Limpieza de datos
│   ├── analysis/          # Módulos de análisis
│   │   ├── statistics.py     # Análisis estadístico
│   │   ├── grouping.py       # Análisis por grupos
│   │   └── risk_analysis.py  # Análisis de riesgos
│   ├── visualization/     # Módulos de visualización
│   │   └── charts.py         # Generación de gráficos
│   ├── reports/           # Módulos de reportes
│   │   ├── apa_report.py     # Generación de reportes APA
│   │   └── apa_report_generator.py
│   └── utils/             # Utilidades
│       └── helpers.py        # Funciones auxiliares
└── output/                # Archivos de salida
    ├── charts/           # Gráficos generados
    └── reports/          # Reportes generados
```

## 🛠️ Instalación

### Prerequisitos
- Python 3.12 o superior
- pip o uv (gestor de paquetes)

### Instalación con uv (recomendado)
```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd programming-iii

# Instalar uv si no lo tienes
pip install uv

# Crear entorno virtual e instalar dependencias
uv sync
```

### Instalación con pip
```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd programming-iii

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## 🔧 Configuración

El archivo `config.py` contiene todas las configuraciones importantes:

```python
# Rutas de archivos
DATA_FILE_PATH = "xAPI-Edu-Data.csv"
OUTPUT_DIR = "output"

# Umbrales de rendimiento
MINIMUM_PASSING_GRADE = 60
EXCELLENT_GRADE = 85
RISK_ABSENCE_THRESHOLD = 7

# Configuración de reportes
REPORT_TITLE = "Academic Performance Analysis Report"
REPORT_AUTHOR = "Programming III Team"
INSTITUTION = "Universidad Tecnológica de Panamá"
```

## 🚀 Uso

### Ejecución básica
```bash
python main.py
```

### Uso programático
```python
from main import run_complete_analysis

# Ejecutar análisis completo
results = run_complete_analysis(
    data_file="mi_archivo.csv",
    generate_charts=True,
    generate_report=True,
    verbose=True
)

# Verificar resultados
if results['success']:
    print(f"Análisis completado exitosamente!")
    print(f"Reportes generados en: {results['report_file']}")
```

## 📊 Tipos de Análisis

### 1. Análisis Estadístico
- Medidas de tendencia central (media, mediana, moda)
- Medidas de dispersión (desviación estándar, varianza)
- Distribuciones de frecuencia
- Análisis de correlación

### 2. Análisis Demográfico
- Rendimiento por género
- Análisis por nacionalidad
- Comparaciones por nivel académico
- Segmentación por semestre

### 3. Análisis de Participación
- Métricas de engagement
- Interacciones en clase
- Uso de recursos educativos
- Participación en discusiones

### 4. Análisis de Riesgos
- Identificación de estudiantes en riesgo
- Análisis de ausentismo
- Patrones de bajo rendimiento
- Prioridades de intervención

## 📈 Visualizaciones

El sistema genera automáticamente los siguientes tipos de gráficos:

- **Distribución de calificaciones** - Histogramas y gráficos de barras
- **Análisis de correlación** - Heatmaps y matrices de correlación
- **Comparaciones demográficas** - Boxplots y gráficos de violin
- **Tendencias temporales** - Gráficos de líneas y series temporales
- **Análisis de riesgos** - Gráficos de dispersión y clasificación

## 📋 Reportes

Los reportes generados incluyen:

- **Resumen ejecutivo** - Hallazgos principales
- **Análisis estadístico detallado** - Métricas y pruebas
- **Visualizaciones integradas** - Gráficos incrustados
- **Recomendaciones** - Sugerencias basadas en datos
- **Formato APA** - Estilo académico profesional

## 🔍 Formato de Datos

El sistema espera datos en formato CSV con las siguientes columnas:

```csv
gender,NationalITy,StageID,GradeID,SectionID,Topic,Semester,Relation,raisedhands,VisITedResources,AnnouncementsView,Discussion,ParentAnsweringSurvey,ParentschoolSatisfaction,StudentAbsenceDays,Class
```

### Descripción de columnas:
- **gender**: Género del estudiante (M/F)
- **NationalITy**: Nacionalidad
- **StageID**: Nivel académico
- **GradeID**: Grado
- **Topic**: Materia
- **raisedhands**: Número de veces que levantó la mano
- **VisITedResources**: Recursos visitados
- **AnnouncementsView**: Anuncios visualizados
- **Discussion**: Participación en discusiones
- **StudentAbsenceDays**: Días de ausencia
- **Class**: Nivel de rendimiento (L/M/H)

## 🤝 Contribución

Para contribuir al proyecto:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crea un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 👥 Autores

- **Programming III Team** - Universidad Tecnológica de Panamá
- **Institución**: Universidad Tecnológica de Panamá

## 📞 Soporte

Para soporte técnico o preguntas:
- Email: soporte@utp.ac.pa
- Documentación: [ARCHITECTURE.md](ARCHITECTURE.md)
- Issues: [GitHub Issues](https://github.com/usuario/programming-iii/issues)

## 🚨 Problemas Conocidos

- Los archivos Excel con caracteres especiales pueden causar errores de codificación
- El sistema requiere al menos 10 filas de datos para análisis significativo
- Algunos gráficos pueden tardar en generarse con datasets grandes (>10,000 filas)

## 🔄 Actualizaciones Futuras

- [ ] Soporte para bases de datos SQL
- [ ] Interfaz web con Flask/Django
- [ ] Análisis predictivo con machine learning
- [ ] Exportación a más formatos (Word, PowerPoint)
- [ ] Integración con APIs de sistemas educativos
- [ ] Análisis en tiempo real

---

**Desarrollado con ❤️ por el equipo de Programming III de la Universidad Tecnológica de Panamá**