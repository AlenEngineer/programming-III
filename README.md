# Sistema de Análisis de Datos Académicos

Esta es la versión completamente traducida al español del Sistema de Análisis de Datos Académicos.

## 🌐 Características

### Interfaz Principal
- Todos los mensajes de usuario en español
- Funciones principales traducidas
- Variables de configuración en español
- Directorios de salida en español

### Funciones Principales
- `ejecutar_analisis_completo()` - Pipeline completo de análisis
- `imprimir_resumen_analisis()` - Resumen de resultados
- `configurar_logging()` - Configuración de logging

### Configuración en Español
- `TITULO_REPORTE` - Título del reporte
- `AUTOR_REPORTE` - Autor del reporte
- `INSTITUCION` - Institución académica
- `COLUMNAS_NUMERICAS` - Columnas numéricas
- `COLUMNAS_CATEGORICAS` - Columnas categóricas
- `MAPEO_RENDIMIENTO` - Mapeo de rendimiento académico

### Directorios de Salida
- `salida/` - Directorio principal de salida
- `salida/graficos/` - Gráficos de visualización
- `salida/reportes/` - Reportes académicos en PDF

## 🚀 Uso del Sistema

### Ejecutar Análisis Completo
```bash
uv python main.py
```

### Importar en Python
```python
from main import ejecutar_analisis_completo

# Ejecutar análisis con configuración personalizada
resultados = ejecutar_analisis_completo(
    archivo_datos="mi_archivo.csv",
    generar_graficos=True,
    generar_reporte=True,
    verboso=True
)
```

## 📊 Salidas del Sistema

### Gráficos Generados
1. **grade_distribution.png** - Distribución de calificaciones
2. **subject_comparison.png** - Comparación por materias
3. **semester_trends.png** - Tendencias por semestre
4. **performance_scatter.png** - Dispersión de rendimiento
5. **attendance_heatmap.png** - Mapa de calor de asistencia
6. **demographic_analysis.png** - Análisis demográfico
7. **correlation_matrix.png** - Matriz de correlación

### Reporte PDF
- Reporte académico completo en formato APA
- Incluye estadísticas, análisis de riesgo y visualizaciones
- Guardado en `salida/reportes/`

## 🔧 Compatibilidad

El sistema mantiene **compatibilidad completa** con el código inglés existente mediante aliases:
- Funciones en inglés siguen funcionando
- Variables de configuración en inglés disponibles
- Importaciones existentes no requieren cambios

## 🎯 Verificación del Sistema

El sistema ha sido completamente probado:
- ✅ 480 registros de estudiantes procesados
- ✅ 7 gráficos de visualización generados
- ✅ Reporte PDF académico creado
- ✅ Análisis estadístico completo
- ✅ Identificación de estudiantes en riesgo
- ✅ Pipeline completo ejecutado exitosamente

## 📁 Estructura de Archivos

```
salida/
├── graficos/
│   ├── attendance_heatmap.png
│   ├── correlation_matrix.png
│   ├── demographic_analysis.png
│   ├── grade_distribution.png
│   ├── performance_scatter.png
│   ├── semester_trends.png
│   └── subject_comparison.png
└── reportes/
    └── academic_analysis_report_YYYYMMDD_HHMMSS.pdf
```

¡El sistema está listo para uso en producción en entornos académicos de habla hispana!
