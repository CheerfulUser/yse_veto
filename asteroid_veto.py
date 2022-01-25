from astropy.coordinates import SkyCoord  # High-level coordinates
import astropy.units as u
#pympc.update_catalogue()
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.utils.data import download_file
from copy import deepcopy

def _query_solar_system_objects(ra, dec, times, radius=10/60, location='807',
                                cache=False):
    """Returns a list of asteroids/comets given a position and time.
    This function relies on The Virtual Observatory Sky Body Tracker (SkyBot)
    service which can be found at http://vo.imcce.fr/webservices/skybot/
     Geert's magic code

    Parameters
    ----------
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    times : array of float
        Times in Julian Date.
    radius : float
        Search radius in degrees.
    location : str
        Spacecraft location. Options include `'kepler'` and `'tess'`.
    cache : bool
        Whether to cache the search result. Default is True.
    Returns
    -------
    result : `pandas.DataFrame`
        DataFrame containing the list of known solar system objects at the
        requested time and location.
    """
    url = 'http://vo.imcce.fr/webservices/skybot/skybotconesearch_query.php?'
    url += '-mime=text&'
    url += '-ra={}&'.format(ra)
    url += '-dec={}&'.format(dec)
    url += '-bd={}&'.format(radius)
    url += '-loc={}&'.format(location)

    df = None
    times = np.atleast_1d(times)
    for time in tqdm(times, desc='Querying for SSOs'):
        url_queried = url + 'EPOCH={}'.format(time)
        response = download_file(url_queried, cache=cache)
        if open(response).read(10) == '# Flag: -1':  # error code detected?
            raise IOError("SkyBot Solar System query failed.\n"
                          "URL used:\n" + url_queried + "\n"
                          "Response received:\n" + open(response).read())
        res = pd.read_csv(response, delimiter='|', skiprows=2)
        if len(res) > 0:
            res['epoch'] = time
            res.rename({'# Num ':'Num', ' Name ':'Name', ' Class ':'Class', ' Mv ':'Mv'}, inplace=True, axis='columns')
            res = res[['Num', 'Name', 'Class', 'Mv', 'epoch']].reset_index(drop=True)
            if df is None:
                df = res
            else:
                df = df.append(res)
    if df is not None:
        df.reset_index(drop=True)
    return df


def check_asteroids(coord,epoch):
	"""
	Find the closest asteroid to the given coordinates at the given epoch
	
	------
	Inputs
	------
		coord : SkyCoord
			coordinates of the target
		epoch : astropy Time
			epoch of the observation 

	-------
	Returns
	-------
		ast : dataframe
			datframe containing the information of the closest asteroid.
			The seperation is given by the sep column which is measured in 
			arcsec.

	"""
    q = _query_solar_system_objects(coord.ra.deg,coord.dec.deg,epoch.jd)
    sep = []
    for i in range(len(q)):
        name = q['Name'][i].strip(' ')
        obs = Horizons(id=name, location='807', epochs=epoch.jd)
        eph = obs.ephemerides().to_pandas()
        ast = SkyCoord(eph.RA[0],eph.DEC[0],unit=u.deg)
        sep += [ast.separation(c).deg*60**2]
    closest = np.argmin(sep)
    ast = deepcopy(q).iloc[closest]
    ast['sep'] = sep[closest]
    return ast
