import pandas as pd
import numpy as np

#server path
ps1_path = '/ifs/cs/projects/armin1/data/rridden/yse/cat/ps1/'
gaia_path = '/ifs/cs/projects/armin1/data/rridden/yse/cat/gaia/'

#local test path
#ps1_path = '/Users/rridden/Documents/work/code/yse/cats/ps1/total_field/'
#gaia_path = '/Users/rridden/Documents/work/code/yse/cats/gaia/total_field/'


def dist_calc(ra,dec,table):
	"""
	Calculates the distance from each input target to sources in the catalogues


	------
	Inputs
	------
		ra : array 
			array of RA for targets 
		dec : array 
			array of Dec for targets 
		table : dataframe
			source catalogue
	-------
	Returns
	-------
		dist : array
			array containing the distance from each target to each source 
			in the catalogues in degrees.
	"""
	dra = ra[:,np.newaxis] - table['ra'].values[np.newaxis,:]
	ddec = dec[:,np.newaxis] - table['dec'].values[np.newaxis,:]
	dist = np.sqrt(dra**2 + ddec**2)
	return dist


def chip_clip(table,chip):
	"""
	limits the input table to sources that are on the input chip

	------
	Inputs
	------
		table : dataframe 
			catalogue 
		chip : int
			DECam chip number
	"""
	chips = table['ampl'].values

	ind = chips == chip

	return table.iloc[ind]

def ps1_gal_clip(table,ps1_gal_cut,ps1_mag_lim):
	"""
	Clips the ps1 source catalogue to remove galaxies by 
	cutting on the ipsf-ikron difference.

	------
	Inputs
	------
		table : dataframe
			ps1 source catalogue 
		ps1_gal_cut : float
			ipsf-ikron limit to distinguish between stars and galaxies
		ps1_mag_lim : float
			magnitude limit to cut the PS1 table
	-------
	Returns
	-------
		tab : dataframe
			truncated version of the input dataframe
	"""
	ipsf = table['iMeanPSFMag'].values
	ikron =table['iMeanKronMag'].values

	rpsf = table['rMeanPSFMag'].values

	ind = ((ipsf - ikron) < ps1_gal_cut) & (rpsf < ps1_mag_lim)

	tab = table.iloc[ind]
	return tab

def close_to_star(dist,mags,faint_mag,bright_mag,rad):
	"""
	Calculates if the sources are close to a star given the limits

	------
	Inputs
	------
		dist : array
			array of distances from the sources to the catalogue stars
		mags : array
			magnitudes of catalogue stars 
		faint_mag : float
			all stars must be brighter than this value
		bright_mag : float
			all stars must be fainter than this value
		rad : float
			radius in arcsec that defines if a source is close 

	-------
	Returns
	-------
		close : boolean array
			True values indicate the source is within the specified distance limit.
	"""
	ind = (mags <= faint_mag) & (mags >= bright_mag)

	d = dist[:,ind]

	close = d < (rad / 60**2)
	close = np.sum(close,axis=1) > 0
	return close 

def star_veto(ra,dec,field,chip=None,ps1_gal_cut=0.05,ps1_mag_lim=20,
			  faint_maglim=15, faint_bad_rad=2,
			  bright_maglim=12, bright_bad_rad=10,
			  saturated_bad_rad=50):
	"""
	Simple catalogue check to veto sources that are close to stars. Limits and radii can
	be fine tuned to match the data.

	------
	Inputs
	------
		ra : list, float, array
			ra of targets to check
		dec : list, float, array
			dec of targets to check
		field : str
			string representing the YSE field to search. Case sensative.
		chip : int
			DECam chip to narrow down the field. This is optional 
		ps1_gal_cut : float 
			The PS1 ipsf-ikron cut for seperating stars from galaxies.
			https://outerspace.stsci.edu/display/PANSTARRS/How+to+separate+stars+and+galaxies
				Default value: 0.05
			
		ps1_mag_lim : float
			Magnitude limit for the PS1 catalogue. The galaxy cut starts to become quite 
			contaminated at faint magnitudes, should be okay for sources brighter than 20/19
				Default value: 20
		faint_maglim : float
			Limit to distinguish faint stars from bright stars.
				Default value: 15
		faint_bad_rad : float
			Radius in arcsec to determine if the object is close to a faint star.
				Default value: 2 arcsec
		bright_maglim : float
			Limit to distinguish bright stars from saturated stars.
				Default value: 12
		bright_bad_rad : float
			Radius in arcsec to determine if the object is close to a bright star.
				Default value: 10 arcsec
		saturated_bad_rad : float
			Radius in arcsec to determine if the object is close to a saturated star.
				Default value: 50 arcsec

	-------
	Returns
	-------
	close_any : boolean array
		A boolean array where True values indicate the source is close to a 
		star given the imposed conditions 
	"""

	if (type(ra) == float) | (type(ra) == int) | (type(ra) == np.float64):
		ra = [ra]
		dec = [dec]
	ra = np.array(ra)
	dec = np.array(dec)
	gaia = pd.read_csv(gaia_path + field + '_gaia.csv')
	ps1 = pd.read_csv(ps1_path + field + '_ps1.csv')
	if chip is not None:
		gaia = chip_clip(gaia,chip)
		ps1 = chip_clip(ps1,chip)

	ps1 = ps1_gal_clip(ps1, ps1_gal_cut, ps1_mag_lim)

	dist_ps1 = dist_calc(ra, dec, ps1)
	dist_gaia = dist_calc(ra, dec, gaia)

	faint_close_ps1 = close_to_star(dist_ps1, ps1['rMeanPSFMag'].values, ps1_mag_lim, 
									faint_maglim, faint_bad_rad)

	
	bright_close_ps1 = close_to_star(dist_ps1, ps1['rMeanPSFMag'].values, faint_maglim, 
									 bright_maglim, bright_bad_rad)	

	bright_close_gaia = close_to_star(dist_gaia, gaia['Gmag'].values, faint_maglim, 
									 bright_maglim, bright_bad_rad)	

	saturated_close_gaia = close_to_star(dist_gaia, gaia['Gmag'].values, bright_maglim, 
										 0, saturated_bad_rad)	

	close_any = faint_close_ps1 | bright_close_ps1 | bright_close_gaia | saturated_close_gaia

	return close_any