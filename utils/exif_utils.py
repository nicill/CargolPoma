import os
import sys
import pathlib
import exifread as ef
import csv
from random import randrange


def _convert_to_degress(value):
	"""
	Helper function to convert the GPS coordinates stored in the EXIF to degress in float format
	:param value:
	:type value: exifread.utils.Ratio
	:rtype: float
	"""
	d = float(value.values[0].num) / float(value.values[0].den)
	m = float(value.values[1].num) / float(value.values[1].den)
	s = float(value.values[2].num) / float(value.values[2].den)

	return d + (m / 60.0) + (s / 3600.0)


def get_gps(filepath):
	"""""
	returns gps data if present other wise returns empty dictionary
	"""
	with open(filepath, 'rb') as f:
		tags = ef.process_file(f)
		latitude = tags.get('GPS GPSLatitude')
		latitude_ref = tags.get('GPS GPSLatitudeRef')
		longitude = tags.get('GPS GPSLongitude')
		longitude_ref = tags.get('GPS GPSLongitudeRef')
		if latitude:
			lat_value = _convert_to_degress(latitude)
			if latitude_ref.values != 'N':
				lat_value = -lat_value
		else:
			return {}
		if longitude:
			lon_value = _convert_to_degress(longitude)
			if longitude_ref.values != 'E':
				lon_value = -lon_value
		else:
			return {}
		return {'latitude': lat_value, 'longitude': lon_value}


def main(argv):

	PATH = pathlib.Path(argv[1])

	with open("exif_info.csv", "w", newline="") as file:
		writer = csv.writer(file)
		writer.writerow(["filename", "GNSS Latitude", "GNSS Longitude", "Grapes found"])

		for filename in os.listdir(PATH):
			# Open image file for reading (binary mode)
			# f = open(PATH / filename, 'rb')

			gps = get_gps(PATH / filename)
			writer.writerow([filename, gps['latitude'], gps['longitude'], randrange(6)])


if __name__ == '__main__':
	main(sys.argv)