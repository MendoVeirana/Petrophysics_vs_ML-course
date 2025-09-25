
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import gaussian_filter

from shapely.geometry import Point
import matplotlib.path
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def Fu(water, clay, por, wc, solid_ec, dry_ec, sat_ec, s=1, w=2):
    """
    Calculate the soil bulk real electrical conductivity using the Fu model and return

    This is a volumetric mixing model that takes into account various soil properties 
    such as clay content, porosity, and water content. 
    It was exhaustively validated using several soil samples [1]. Reported R2 = 0.98

    Parameters
    ----------
    water : array_like
        Soil volumetric water content [m**3/m**3].
    clay : array_like
        Soil clay content [g/g]*100.
    por: array_like
        Soil porosity [m**3/m**3].
    wc : array_like
        Soil water real electrical conductivity [S/m].
    solid_ec : array_like
        Soil solid real electrical conductivity [S/m].
    dry_ec : array_like
        Soil bulk real electrical conductivity at zero water content [S/m].
    sat_ec : array_like
        Soil bulk real electrical conductivity at saturation water content [S/m].
    s : float, optional
        Phase exponent of the solid, default is 1.
    w : float, optional
        Phase exponent of the water, default is 2.

    Returns
    -------
    array_like
        The estimated bulk electrical conductivity [S/m].

    Notes
    -----
    The method uses default values for s and w, which are 1 and 2 respectively, 
    but can be modified if necessary. Three different forms of the model are used 
    depending on the soil data availability. The soil electrical conductivity of solid surfaces 
    is calculated as in [1] using the formula of Doussan and Ruy (2009) [2]

    References
    ----------
    .. [1] Yongwei Fu, Robert Horton, Tusheng Ren, J.L. Heitman,
    A general form of Archie's model for estimating bulk soil electrical conductivity,
    Journal of Hydrology, Volume 597, 2021, 126160, ISSN 0022-1694, https://doi.org/10.1016/j.jhydrol.2021.126160.
    .. [2] Doussan, C., and Ruy, S. (2009), 
    Prediction of unsaturated soil hydraulic conductivity with electrical conductivity, 
    Water Resour. Res., 45, W10408, doi:10.1029/2008WR007309.

    Example
    -------
    >>> Fu(0.3, 30, 1.3, 2.65, 0.3, 0, np.nan, np.nan)
    0.072626

    """
    d = 0.6539
    e = 0.0183
    surf_ec = (d*clay/(100-clay))+e # Soil electrical conductivity of solid surfaces

    if np.isnan(dry_ec) & np.isnan(sat_ec):
        bulk_ec = solid_ec*(1-por)**s + (water**(w-1))*(por*surf_ec) + wc*water**w

    elif ~(np.isnan(dry_ec)) & ~(np.isnan(sat_ec)):
        bulk_ec = dry_ec + ((dry_ec-sat_ec)/(por**w) - surf_ec)*water**w + (water**(w-1))*(por*surf_ec)

    elif ~(np.isnan(dry_ec)) & np.isnan(sat_ec):
        sat_ec = dry_ec + (wc+surf_ec)*por**w
        bulk_ec = dry_ec + ((dry_ec-sat_ec)/(por**w) - surf_ec)*water**w + (water**(w-1))*(por*surf_ec)

    return bulk_ec



#Predicting water content, clay content and bulk density with Linde et al. (2006)
# -------------------------------------------------------------------------------------
def linde(water, por, sand, clay, wc, pdn=2.65, m=1.5, n=2):
    """        
        Parameters
        ----------
        vwc: float
            volumetric water content [%]
        
        por: float
            Soil porosity [m**3/m**3].

        clay: float
            Soil volumetric clay content [%]

        wc: float
            Soil water real electrical conductivity [mS/m]

        pdn: float
            particle density [g/cm3]

        m: float
            cementation exponent [-]

        n: float
            saturation exponent [-]

        Returns
        -------
        bulk_ec: float
            Soil bulk real electrical conductivity [mS/m]
    """  

    sat_w = (water/100)/por # water saturation
    f_form = por**(-m) # formation factor
    
    water_ec = wc/1000

    silt = 100 - clay - sand

    radius_clay = 0.002/2000
    radius_silt = 0.025/2000
    radius_sand = 0.75/2000

    solid_ec = 1*(10**-7) # Solid electrical conductivity
    clay_ec= 3*(solid_ec/radius_clay)  # clay electrical conductivity
    silt_ec = 3*(solid_ec/radius_silt) # Silt electrical conductivity
    sand_ec = 3*(solid_ec/radius_sand) # Sand electrical conductivity

    surf_ec = np.average([clay_ec * (clay / 100), 
                            sand_ec * (sand / 100), 
                            silt_ec * (silt / 100)], axis=0)
    
    bulk_ec = (((sat_w**n)*water_ec) 
               + (f_form - 1)*(surf_ec))/f_form 
    
    return bulk_ec*1000


def export_grid(grid_in, filename='georaster'):
    """
    Interpolate scatter data to regular grid through selected interpolation 
    method (with scipy.interpolate for simple interpolation).

    Parameters
    ----------
    grid_in : np.array
        Array of interpolated and masked grid.

    filename : str, optional
        Name of the GeoTIFF (.tif) file (standard = 'gridded').

    y : np.array
        Cartesian GPS y-coordinates.

    z : np.array
        Data points to interpolate.
    
    """

    #   Get grid properties
    cell_size = grid_in['cell_size']
    extent = grid_in['extent']
    transform = from_origin(extent['x_min'], extent['y_min'], 
                                cell_size, -cell_size)

    #   Prepare rasterio grid
    grid_exp = grid_in['grid']
    grid_exp[np.isnan(grid_exp)] = -99999
    grid_exp = grid_exp.astype(rasterio.float32)
    nx, ny = grid_exp.shape
    grid_exp = np.flip(grid_exp, axis=0)

    
    #   Create an empty grid with correct name and coordinate system
    with rasterio.open(
        filename + '.tif',
        mode='w',
        driver='GTiff',
        height=nx,
        width=ny,
        count=1,
        dtype=str(grid_exp.dtype),
        crs='EPSG:31370', #Lambert 1972 coordinates
        transform=transform,
        nodata=-99999
    ) as dst:
        dst.write(grid_exp, 1)

    # Open the GeoTIFF file in read/write mode to flip the image vertically
    with rasterio.open(filename + '.tif', mode='r+') as dst:
        data = dst.read()
        dst.write(data[0, ::-1], 1)

def interpolate(x, y, z, cell_size, blank, method='nearest', smooth_s = 0):
    """
    Interpolate scatter data to regular grid through selected interpolation 
    method (with scipy.interpolate for simple interpolation).

    The output of this function is a Numpy array that holds the interpolated 
    data (accessed via `datagrid['grid']` in the cell below), alongside 
    the grid's cell size (`datagrid['cell_size']`) and the grid extent 
    (`datagrid['extent']`). The cell size and extent are needed to allow 
    exporting the interpolated data efficiently to a GeoTIF that can be 
    opened in any GIS software such as QGIS. 

    Parameters
    ----------
    x : np.array
        Cartesian GPS x-coordinates.

    y : np.array
        Cartesian GPS y-coordinates.

    z : np.array
        Data points to interpolate.

    cell_size : float
        Grid cell size (m).

    method : str, optional
        Scipy interpolation method ('nearest', 'linear', 'cubic' or 'IDW')

    smooth_s : float, optional
        Smoothing factor to apply a Gaussian filter on the interpolated grid.
        If 0, no smoothing is performed. (Applying smoothing can result 
        in a loss of detail in the interpolated grid.)

    blank : object
        A blank object to mask (clip )interpolation beyond survey bounds.

    Returns
    -------
    grid : np.array
        Array of interpolated and masked grid containing:
        - the interpolated grid (grid['grid'])
        - the grid cell size (grid['cell_size'])
        - the grid extent (grid['extent'])

    """

    # Define an evenly spaced grid over which the dataset values have to be 
    # interpolated
    x_min = x.min()
    x_max = x.max() + cell_size
    y_min = y.min()
    y_max = y.max() + cell_size
    x_vector = np.arange(x_min, x_max, cell_size)
    y_vector = np.arange(y_min, y_max, cell_size)
    extent = (x_vector[0], x_vector[-1], y_vector[0], y_vector[-1])

    xx, yy = np.meshgrid(x_vector, y_vector)
    nx, ny = xx.shape
    coords = np.concatenate((xx.ravel()[np.newaxis].T, 
                        yy.ravel()[np.newaxis].T), 
                        axis=1)

    # Create a mask to blank grid outside surveyed area
    boolean = np.zeros_like(xx)
    boundaries = np.vstack(blank.loc[0, 'geometry'].exterior.coords.xy).T
    bound = boundaries.copy()
    boolean += matplotlib.path.Path(
        bound).contains_points(coords).reshape((nx, ny))
    boolean = np.where(boolean >= 1, True, False)
    mask = np.where(boolean == False, np.nan, 1)
    binary = np.where(boolean == False, 0, 1)
    
    # Fast (and sloppy) interpolation (scipy.interpolate)
    if method in ['nearest','cubic', 'linear']:
        # Interpolate 
        data_grid = griddata(
            np.vstack((x, y)).T, z, (xx, yy), method=method
            ) * mask
    else:
        print('define interpolation method')
    
    if smooth_s > 0:
        data_grid = gaussian_filter(data_grid, sigma=smooth_s)

    # Create a structured array with additional fields for coordinates and cell size
    dtype = [
        ('grid', data_grid.dtype, data_grid.shape),
        ('cell_size', float),
        ('extent', [
            ('x_min', float), 
            ('x_max', float), 
            ('y_min', float), 
            ('y_max', float)
            ])
    ]
    grid = np.array((data_grid, cell_size, extent), dtype=dtype)
    
    return grid