# Construcción de la base de datos

Para la construcción de la base de datos preliminar se tienen en cuenta, hasta ahora, 3 fuentes de datos pincipales.

- **Casos confirmados de Dengue:** Casos confirmados de Dengue y Dengue Grave obtenidos por el registro en SIVIGILA y consolidado que dispone el Ministerio de Salud y Protección Social a través de los cubos de información del SISPRO. Se obtienen datos historicos con periodicidad mensual de 2007 a 2024, la agregación es a nivel municipal.
- **Coordenadas:** Para la obtención de coordenadas por municipio se utiliza el recurso del DANE con la codificación DIVIPOLA que se obtiene en https://geoportal.dane.gov.co/servicios/descarga-y-metadatos/datos-geoestadisticos/?cod=112
- **Variables Climaticas:** Obtenidas de la base de datos historica mensual de WorldClim, quienes funcionan como un repositorio que consolida los reportes del clima de estaciones en todo el mundo, al generar datos espaciales del clima utilizan técnicas de interpolación para generar superficies continuas (splines de placa delgada) apoyándose en datos de elevación y distancia a la costa para estimar el clima en zonas sin estaciones.

    - **Splines de Placa Delgada (Thin Plate Splines):** Utilizan un software llamado ANUSPLIN. Este algoritmo ajusta superficies suaves a los datos de las estaciones. Imagina estirar una lámina de goma flexible que debe tocar la "cabeza" de cada estación meteorológica.
    - **Uso de Covariables (La Clave):** No interpolan solo basándose en la distancia entre estaciones. Usan variables auxiliares que se conocen con mucha precisión satelital:
    - **Elevación (DEM):** Es el factor más importante. Como la temperatura disminuye con la altura, WorldClim usa datos de elevación (como SRTM) para corregir la interpolación. Esto permite que el modelo "sepa" que la cima de una montaña debe ser más fría que el valle, aunque no haya una estación en la cima.
    - **Distancia a la costa:** Ayuda a moderar las temperaturas en zonas costeras.
    - **Datos Satelitales (MODIS):** En versiones recientes, utilizan datos de temperatura superficial terrestre obtenidos por satélite como covariable para mejorar la precisión en áreas con pocas estaciones físicas.


# Diccionario de datos

- **CodigoMunicipio (DANE):** El codigo DIVIPOLA del municipio.
- **Departamento (SIVIGILA):** El departamento al que pertenece el municipio.
- **Municipio (SIVIGILA):** El nombre del municipio.
- **TipoMunicipio (DANE):** Tipo de municipio ('municipio', 'isla', 'area no municipalizada')
- **Longitud (DANE):** La longitud del municipio.
- **Latitud (DANE):** La latitud del municipio.
- **Year (SIVIGILA):** Año de los registros.
- **Mes (SIVIGILA):** Mes de los registros.
- **CasosConfirmados (SIVIGILA):** Número de casos confirmados de Dengue y Dengue Grave.
- **Elevacion (WorldClim):** Elevación en metros sobre el nivel del mar.
- **Precipitacion (WorldClim):** Precipitación total en mm.
- **TempMax (WorldClim):** Temperatura máxima en grados Celsius.
- **TempMin (WorldClim):** Temperatura mínima en grados Celsius.

### Instrucciones para construcción de la base de datos a partir de notebook BaseDengue.ipynb

Se debe crear un entorno en python con la versión 3.12 e instalar las librerías contenidas en el archivo requirements.txt, este archivo también contiene los comandos para llevar a cabo toda la instalación siempre y cuando se tenga conda o miniconda instalado.


1. El archico principal corresponde al *BaseDengue.csv*, que corresponde a la data extraida del SISPRO. Debe estar en la carpeta data.
2. El archivo secundario del DANE que contiene las coordenadas, codigo y tipo de municipio se obtiene al descargar en el link mencionado más arriba en la tabla *Listado completos de Codificación Divipola*. También se debe almacenar en la carpeta data.
3. La data de WorldClim se obtiene a partir de multiples archivos, 3 archivos comprimidos por cada variable y se alamacenan en data\worldclim, allí se descomprimen cada uno en su carpeta correspondiente.
4. Con los archivos ubicados basta con ejecutar todo el notebook y se genera el resultado en data con el nombre de dengue_data_v2.csv