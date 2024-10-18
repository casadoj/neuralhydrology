# Catchment attributes

The computation of catchment statistics is done using the tool `catchment_statistics` from the repository `lisflood-utilities`:

	```from lisfloodutilities import catchment_statistics```

The computations are weighted by the pixel area (which changes with latitude) using the static map _pixarea.nc_.

## Static maps

The file _glofas_static_maps.csv_ contains catchment attributes extracted from the GloFAS static maps. The statistics used for each static maps are controlled in the configuration file _attributes.yml_.

### Geomorphology

* From the elevation map mean, standard deviation, minimum and maximum are computed.
* The mean and standard deviation of the gradient.
* The maximum of the upstream area represents the catchment area.

### Land use

The proportion of the catchment area dedicated to the different land uses (forest, irrigated, other, rice, water and sealed) is computed as the mean of the respective static map.

The raw land use maps will be later on used to weigh other static maps discretised by land use.

### Crop coefficient

A weighted average and standard deviation of the crop coefficient is computed by crossing the different crop coefficient maps with the land use maps.

### Streams

These attributes include the mean of the bankful depth and width, the channel gradient and the Manning coefficient, and the total length of channels.

### Soil properties

In this case, the six variables (ksat, lambda, genua, soildepth, thetas, thetar) are discretised in the trhee soil layers and the different land uses. The catchment attributes are the weighted average of each of the variables for each soil layer; the weighing is based on the proportion of each of the land uses.

### LAI

The LAI maps are 10-daily timesteps for each land use. The catchment attributes include monthly and annual averages weighted by the proportion of each of the land uses.

### Reservoirs and lakes

In both cases the attributes are the count of reservoirs/lakes in the catchment and the total sum of the main characteristic of that type of water body: storage in the case of reservoirs and area in the case of lakes.


## Model parameters

The file _glofas_model_paraemeters.csv_ contains the catchment mean values of each of the 14 LISFLOOD model parameters. Since the model parameters are constant across the headwater cathcments, the mean, median, minimum or maximum would result in the same value.